import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import csv

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parents[1]  # project root
CSV_PATH = BASE_DIR / "src" / "emails.csv"

app = FastAPI()

VS: FAISS | None = None  # will be built on startup


def build_vectorstore(csv_path: Path) -> FAISS | None:
    if not csv_path.exists():
        print(f"[build_vectorstore] CSV not found: {csv_path}")
        return None

    rows: list[Document] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # EXPECTED HEADERS:
        # 'sender', 'subject', 'body (incoming)', 'your_reply (outgoing)'
        for rec in r:
            sender = (rec.get("sender") or "").strip()
            subject = (rec.get("subject") or "").strip()
            body_in = (rec.get("body (incoming)") or "").strip()
            reply_out = (rec.get("your_reply (outgoing)") or "").strip()
            if not body_in:
                continue
            content = f"From: {sender}\nSubject: {subject}\nBody: {body_in}"
            rows.append(Document(page_content=content, metadata={"reply": reply_out}))

    if not rows:
        print("[build_vectorstore] No usable rows in CSV.")
        return None

    emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vs = FAISS.from_documents(rows, emb)
    print(f"[build_vectorstore] Built FAISS with {len(rows)} rows from {csv_path}")
    return vs


@app.on_event("startup")
def _startup():
    global VS
    VS = build_vectorstore(CSV_PATH)


class DraftReq(BaseModel):
    sender: str
    subject: str
    body: str


@app.post("/draft")
def draft(req: DraftReq):
    # 1) Retrieve examples (with scores so you can see matches)
    examples = ""
    retrieved_count = 0
    if VS and req.body:
        pairs = VS.similarity_search_with_score(req.body, k=3)
        retrieved_count = len(pairs)
        examples = "\n\n".join(
            f"[score={score:.3f}] {doc.page_content}\nReply: {doc.metadata.get('reply','')}"
            for doc, score in pairs
        )

    # 2) Call the LLM
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL, temperature=0.3)
    prompt = f"""
You are drafting an email reply on behalf of Jurmain Mitchell (Finance Chair, Region 3 NSBE).

New Email:
From: {req.sender}
Subject: {req.subject}
Body: {req.body}

Here are Jurmain's past replies to emails like these:
{examples}

Return ONLY the email text.
""".strip()
    reply_text = llm.invoke(prompt).content.strip()

    return {
        "reply_subject": f"Re: {req.subject}",
        "reply_body": reply_text,
        "retrieved_examples_count": retrieved_count,
        "examples_used": examples,
    }
