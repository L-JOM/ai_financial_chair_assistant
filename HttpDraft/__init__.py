import os, json, csv, time, logging
from pathlib import Path
import azure.functions as func
from importlib.metadata import version, PackageNotFoundError



# ---------- logging ----------
log = logging.getLogger("HttpDraft")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log.setLevel(logging.INFO)

CSV_BLOB_URL = os.getenv("CSV_BLOB_URL")  # already set in Function App settings

def pv(n: str) -> str:
    try:
        return version(n)
    except PackageNotFoundError:
        return "NOT INSTALLED"

log.info(
    "pkgs -> langchain=%s, langchain-openai=%s, openai=%s, faiss-cpu=%s",
    pv("langchain"), pv("langchain-openai"), pv("openai"), pv("faiss-cpu")
)

log.info("Available packages: langchain-openai=%s, openai=%s", pv("langchain-openai"), pv("openai"))


# ---------- paths ----------
#BASE_DIR = Path(__file__).resolve().parent.parent  # repo root when deployed
#CSV_PATH = Path(os.getenv("CSV_PATH", str(BASE_DIR / "src" / "emails.csv")))

# ---------- helpers ----------
def _json_response(payload: dict, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(json.dumps(payload), mimetype="application/json", status_code=status)

def build_db():
    rows = []
    data = read_csv_from_blob()
    if not data:
        log.warning("No rows found in blob CSV")
        return None

    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document

        for r in data:
            body_in = (r.get("body (incoming)") or "").strip()
            if not body_in:
                continue
            sender = (r.get("sender") or "").strip()
            subject = (r.get("subject") or "").strip()
            reply_out = (r.get("your_reply (outgoing)") or "").strip()
            content = f"From: {sender}\nSubject: {subject}\nBody: {body_in}"
            rows.append(Document(page_content=content, metadata={"reply": reply_out}))

        if not rows:
            log.warning("Blob CSV had no usable rows")
            return None

        api_key = os.getenv("OPENAI_API_KEY")
        emb = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")
        vs = FAISS.from_documents(rows, emb)
        log.info("FAISS built with %d rows", len(rows))
        return vs

    except Exception:
        log.exception("Failed to build FAISS from blob CSV")
        return None


import urllib.request

def read_csv_from_blob():
    if not CSV_BLOB_URL:
        log.error("CSV_BLOB_URL is not set")
        return []
    try:
        with urllib.request.urlopen(CSV_BLOB_URL) as response:
            content = response.read().decode("utf-8").splitlines()
            reader = csv.DictReader(content)
            rows = list(reader)
            log.info("Blob CSV: headers=%s rows=%d", reader.fieldnames, len(rows))
            return rows
    except Exception as e:
        log.exception("Failed to read CSV from blob: %s", e)
        return []



# ---------- function entry ----------
def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    inv = getattr(context, "invocation_id", "unknown")
    log.info("[%s] LLM email draft triggered", inv)

    # Parse JSON
    try:
        data = req.get_json()
    except Exception as e:
        log.exception("[%s] Invalid JSON", inv)
        return _json_response({"error": "invalid_json", "detail": str(e)}, 400)

    subject = (data.get("subject") or "").strip()
    body    = (data.get("body") or "").strip()
    sender  = (data.get("sender") or "").strip()
    log.info("[%s] Input lens - sender:%d subject:%d body:%d", inv, len(sender), len(subject), len(body))

    # Retrieve examples
    # Retrieve examples (build vector store per request)
    examples = ""
    try:
        vs = build_db()  # build from blob each time
        if vs and body:
            t0 = time.time()
            docs = vs.similarity_search(body, k=3)
            dt = time.time() - t0
            examples = "\n\n".join(
                f"Email: {d.page_content}\nReply: {d.metadata.get('reply','')}" for d in docs
            )
            log.info("[%s] Retrieved %d examples in %.3fs", inv, len(docs), dt)
        else:
            log.info("[%s] No vector store available; skipping retrieval.", inv)
    except Exception:
        log.exception("[%s] Retrieval failed (continuing without examples)", inv)


    # Build prompt
    prompt = f"""
You are drafting an email reply on behalf of Jurmain Mitchell (Finance Chair, Region 3 NSBE).
You are an assistant of Jurmain drafting professional emails for NSBE corporate relations.

The user received an email from {sender} with subject: "{subject}".

New Email:
{body}

Here are Jurmain's past replies to similar emails:
{examples}

IMPORTANT: Study Jurmain's writing style, tone, and availability patterns, and the context of his replies from the examples above. 
- Copy his phrasing patterns 
- Use similar availability times and days that he typically mentions
- Match his informal but professional tone
- Use his previous answers in the examples to respond to the emails, but cater to the sender and subject.
Write a reply that sounds  like Jurmain wrote it himself.
Return ONLY the email text.
""".strip()

    # Call LLM (lazy import; fallback to OpenAI SDK)
    api_key = os.getenv("OPENAI_API_KEY")
    model   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        log.error("[%s] OPENAI_API_KEY missing", inv)
        return _json_response({"error": "missing_openai_api_key"}, 500)

    reply_text = None
    # Try langchain_openai first
    try:
        from langchain_openai import ChatOpenAI  # lazy import to avoid freezing None
        t0 = time.time()
        llm = ChatOpenAI(openai_api_key=api_key, model=model, temperature=0.3)
        reply_text = llm.invoke(prompt).content.strip()
        log.info("[%s] LLM call ok (langchain) in %.2fs", inv, time.time() - t0)
    except Exception:
        log.exception("[%s] langchain-openai path failed; falling back to OpenAI SDK", inv)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            reply_text = (resp.choices[0].message.content or "").strip()
            log.info("[%s] LLM call ok (openai sdk) in %.2fs", inv, time.time() - t0)
        except Exception as e2:
            log.exception("[%s] OpenAI SDK fallback failed", inv)
            return _json_response({"error": "llm_failed", "detail": str(e2)}, 500)

    return _json_response({
        "reply_subject": f"Re: {subject}" if subject else "Re:",
        "reply_body": reply_text,
        "examples_used": examples
    }, 200)
