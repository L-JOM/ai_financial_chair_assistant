import logging, os, json, csv, time
import azure.functions as func
from pathlib import Path

# Optional: tighten log format a bit (Functions will still route to App Insights)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("HttpDraft")

# ----- Paths / constants
BASE_DIR = Path(__file__).resolve().parent.parent   # repo root when deployed
CSV_PATH = BASE_DIR / "src" / "emails.csv"          # adjust if you move it

# ----- Lazy imports (so import errors don't kill the process)
def _safe_imports():
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document
        return ChatOpenAI, OpenAIEmbeddings, FAISS, Document
    except Exception as e:
        log.exception("Dependency import failed")
        return None, None, None, None


ChatOpenAI, OpenAIEmbeddings, FAISS, Document = _safe_imports()

# ----- Build vector store (tolerant & logged)
def build_db(path: Path = CSV_PATH):
    if not (ChatOpenAI and OpenAIEmbeddings and FAISS and Document):
        log.warning("LangChain/OpenAI not available; retrieval disabled.")
        return None
    log.info(f"CSV_PATH resolved: {path} exists={path.exists()} abs={path}")

    if not path.exists():
        log.warning("emails.csv not found; continuing without examples.")
        return None

    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            log.info(f"CSV headers: {headers}")

            # Expecting: sender, subject, body (incoming), your_reply (outgoing)
            for r in reader:
                body_in = (r.get("body (incoming)") or "").strip()
                if not body_in:
                    continue
                sender = (r.get("sender") or "").strip()
                subject = (r.get("subject") or "").strip()
                reply_out = (r.get("your_reply (outgoing)") or "").strip()
                content = f"From: {sender}\nSubject: {subject}\nBody: {body_in}"
                rows.append(Document(page_content=content, metadata={"reply": reply_out}))
    except Exception as e:
        log.exception("Failed reading CSV")
        return None

    if not rows:
        log.warning("emails.csv had no usable rows; retrieval disabled.")
        return None

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log.warning("OPENAI_API_KEY not set; retrieval/LLM may fail.")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")
        t0 = time.time()
        vs = FAISS.from_documents(rows, embeddings)
        log.info(f"FAISS built with {len(rows)} rows in {time.time()-t0:.2f}s")
        return vs
    except Exception:
        log.exception("Failed to build FAISS; continuing without retrieval.")
        return None

DB = build_db()

def _json_response(payload: dict, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(json.dumps(payload), mimetype="application/json", status_code=status)

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    inv = getattr(context, "invocation_id", "unknown")
    log.info(f"[{inv}] LLM email draft triggered")

    # Parse JSON safely
    try:
        data = req.get_json()
    except Exception as e:
        log.exception(f"[{inv}] Invalid JSON")
        return _json_response({"error": "invalid_json", "detail": str(e)}, 400)

    subject = (data.get("subject") or "").strip()
    body    = (data.get("body") or "").strip()
    sender  = (data.get("sender") or "").strip()

    log.info(f"[{inv}] Input lens - sender:{len(sender)} subject:{len(subject)} body:{len(body)}")

    # Retrieve examples (if DB available)
    examples = ""
    try:
        if DB and body:
            t0 = time.time()
            docs = DB.similarity_search(body, k=3)
            dt = time.time() - t0
            examples = "\n\n".join(
                f"Email: {d.page_content}\nReply: {d.metadata.get('reply','')}" for d in docs
            )
            log.info(f"[{inv}] Retrieved {len(docs)} examples in {dt:.3f}s")
        else:
            if not DB:
                log.info(f"[{inv}] No DB available; skipping retrieval.")
    except Exception:
        log.exception(f"[{inv}] Retrieval failed (continuing without examples)")

    # Build prompt & call LLM
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

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        model   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not api_key:
            log.error(f"[{inv}] OPENAI_API_KEY missing")
            return _json_response({"error": "missing_openai_api_key"}, 500)

        t0 = time.time()
        if ChatOpenAI is None:
            log.error(f"[{inv}] langchain-openai not installed")
            return _json_response({"error": "missing_dependency", "detail": "langchain-openai not installed"}, 500)

        llm = ChatOpenAI(openai_api_key=api_key, model=model, temperature=0.3)
        reply_text = llm.invoke(prompt).content.strip()
        log.info(f"[{inv}] LLM call ok in {time.time()-t0:.2f}s")

        return _json_response({
            "reply_subject": f"Re: {subject}" if subject else "Re:",
            "reply_body": reply_text,
            "examples_used": examples
        }, 200)

    except Exception as e:
        log.exception(f"[{inv}] LLM call failed")
        return _json_response({"error": "llm_failed", "detail": str(e)}, 500)
