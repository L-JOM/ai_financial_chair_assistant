import logging, os, json, csv
import azure.functions as func
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "src" / "emails.csv"


def build_db(path="CSV_PATH"):
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    rows = []
    if not os.path.exists(path):
        logging.warning(f"emails.csv not found at {os.path.abspath(path)}")
        return None
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            email = (r.get("email") or "").strip()
            reply = (r.get("reply") or "").strip()
            if email:
                rows.append(Document(page_content=email, metadata={"reply": reply}))
    if not rows:
        logging.warning("emails.csv had no usable rows.")
        return None
    return FAISS.from_documents(rows, embeddings)

DB = build_db()

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("LLM email draft triggered")
    try:
        data = req.get_json()
        subject = (data.get('subject') or "").strip()
        body    = (data.get('body') or "").strip()
        sender  = (data.get('sender') or "").strip()

        # retrieve examples
        examples = ""
        if DB and body:
            docs = DB.similarity_search(body, k=3)
            examples = "\n\n".join(
                [f"Email: {d.page_content}\nReply: {d.metadata.get('reply','')}" for d in docs]
            )
        logging.info(f"Retrieved {0 if not examples else len(examples.split('Email: '))-1} examples")

        # prompt + LLM
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

        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
        )
        reply_text = llm.invoke(prompt).content.strip()

        result = {
            "reply_subject": f"Re: {subject}" if subject else "Re:",
            "reply_body": reply_text,
            "examples_used": examples  # <— INCLUDED
        }
        return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.exception("handler failed")
        return func.HttpResponse(json.dumps({"error": str(e)}), mimetype="application/json", status_code=500)
