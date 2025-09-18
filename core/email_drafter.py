import os, csv
from typing import Tuple, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "src" / "emails.csv"

def build_vectorstore(csv_path: Optional[Path] = None, api_key: Optional[str] = None) -> Optional[FAISS]:
    csv_path = Path(csv_path or CSV_PATH)
    if not csv_path.exists():
        print(f"[build_vectorstore] CSV not found at: {csv_path}")
        return None

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # expected headers: sender, subject, body, your_reply
        for r in reader:
            sender = (r.get("sender") or "").strip()
            subject = (r.get("subject") or "").strip()
            body_in = (r.get("body") or r.get("incoming") or "").strip()  # tolerate either header name
            reply_out = (r.get("your_reply") or r.get("outgoing") or "").strip()

            # skip if no body text
            if not body_in:
                continue

            # index on the incoming message (best for retrieval)
            # include sender/subject to help similarity search
            content = f"From: {sender}\nSubject: {subject}\nBody: {body_in}".strip()
            rows.append(Document(page_content=content, metadata={"reply": reply_out}))

    if not rows:
        print("[build_vectorstore] No usable rows in CSV (check headers and data).")
        return None

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vs = FAISS.from_documents(rows, embeddings)
    print(f"[build_vectorstore] Built FAISS with {len(rows)} rows from {csv_path}")
    return vs

def _render_prompt(sender: str, subject: str, body: str, examples: str) -> str:
    return f"""
You are drafting an email reply on behalf of Jurmain Mitchell (Finance Chair, Region 3 NSBE).
You are an assistant of Jurmain drafting professional emails for NSBE corporate relations.

The user received an email from {sender} with subject: "{subject}".
Here is the message body:

New Email:
{body}

Here are Jurmain's past replies to emails like these:
{examples}

Write a reply to the new email that is consistent with Jurmain’s style of writing.
Return ONLY the email text.
""".strip()


def draft_reply(
    sender: str,
    subject: str,
    body: str,
    vectorstore: Optional[FAISS],
    openai_api_key: str,
    model: str = "gpt-4o-mini",
    k: int = 3,
):
    examples = ""
    if vectorstore and body:
        docs = vectorstore.similarity_search(body, k=k)
        examples = "\n\n".join(
            [f"Email: {d.page_content}\nReply: {d.metadata.get('reply','')}" for d in docs]
        )
    prompt = _render_prompt(sender, subject, body, examples)
    llm = ChatOpenAI(openai_api_key=openai_api_key, model=model, temperature=0.3)
    reply_text = llm.invoke(prompt).content.strip()
    return (f"Re: {subject}" if subject else "Re:"), reply_text, examples