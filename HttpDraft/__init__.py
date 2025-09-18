import os, json, csv, time, logging
from pathlib import Path
import azure.functions as func
from importlib.metadata import version, PackageNotFoundError




# ---------- logging ----------
log = logging.getLogger("HttpDraft")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log.setLevel(logging.INFO)



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
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root when deployed
CSV_PATH = Path(os.getenv("CSV_PATH", str(BASE_DIR / "src" / "emails.csv")))

# ---------- helpers ----------
def _json_response(payload: dict, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(json.dumps(payload), mimetype="application/json", status_code=status)

def build_db(path: Path = CSV_PATH):
    """
    Build a FAISS vector store from CSV. Tolerant:
    - If CSV missing/empty or deps not installed, returns None and logs why.
    - Expects headers: sender, subject, body (incoming), your_reply (outgoing)
    """
    log.info("CSV_PATH resolved: %s exists=%s", path, path.exists())
    if not path.exists():
        log.warning("emails.csv not found; continuing without examples.")
        return None

    # Lazy-import retrieval deps so missing packages don't crash the app
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document
    except Exception:
        log.exception("Retrieval dependencies not importable; retrieval disabled.")
        return None

    # Read CSV
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            log.info("CSV headers: %s", headers)

            for r in reader:
                body_in = (r.get("body (incoming)") or "").strip()
                if not body_in:
                    continue
                sender = (r.get("sender") or "").strip()
                subject = (r.get("subject") or "").strip()
                reply_out = (r.get("your_reply (outgoing)") or "").strip()
                content = f"From: {sender}\nSubject: {subject}\nBody: {body_in}"
                rows.append(Document(page_content=content, metadata={"reply": reply_out}))
    except Exception:
        log.exception("Failed reading CSV")
        return None

    if not rows:
        log.warning("emails.csv had no usable rows; retrieval disabled.")
        return None

    # Build embeddings + FAISS
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log.warning("OPENAI_API_KEY not set; retrieval/LLM may fail.")
        emb = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")
        t0 = time.time()
        vs = FAISS.from_documents(rows, emb)
        log.info("FAISS built with %d rows in %.2fs", len(rows), time.time() - t0)
        return vs
    except Exception:
        log.exception("Failed to build FAISS; retrieval disabled.")
        return None

DB = build_db()

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
    examples = ""
    try:
        if DB and body:
            t0 = time.time()
            docs = DB.similarity_search(body, k=3)
            dt = time.time() - t0
            examples = "\n\n".join(
                f"Email: {d.page_content}\nReply: {d.metadata.get('reply','')}" for d in docs
            )
            log.info("[%s] Retrieved %d examples in %.3fs", inv, len(docs), dt)
        else:
            if not DB:
                log.info("[%s] No DB available; skipping retrieval.", inv)
    except Exception:
        log.exception("[%s] Retrieval failed (continuing without examples)", inv)

    # Build prompt
    prompt = f"""
You are drafting an email reply on behalf of Jurmain Mitchell (Finance Chair, Region 3 NSBE).
You are an assistant of Jurmain drafting professional emails for NSBE corporate relations.
The user received an email from {sender} with subject: "{subject}".

New Email:
{body}

Here are Jurmain's past replies to emails like these:
{examples}

Write a reply to the new email that is consistent with Jurmain’s style of writing.
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
