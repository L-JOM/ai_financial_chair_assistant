import logging
import azure.functions as func
import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-5-nano", 
    temperature=0.3
)

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
db = FAISS.load_local("emails.csv", embeddings)

prompt_template = ChatPromptTemplate.from_template("""
You are drafting an email reply on behalf of Jurmain Mitchell (Finance Chair, Region 3 NSBE).
You are an assistant of Jurmain drafting professional emails for NSBE corporate relations.
The user received an email from {sender} with subject: "{subject}".
Here is the message body: 

New Email: {body}
    
    
Here are Jurmain's past replies to emails like these
Similar Past Replies:
{retrieved_examples}

Write a reply to the new email that is consistent with Jurmain’s style of writing 
Return ONLY the email text.
""")

def main(request):
    logging.info("LLM email draft triggered")
    
    try:
        req_body = request.get_json()
        subject = req_body.get('subject')
        body = req_body.get('body')
        sender = req_body.get('sender')
        
        docs = db.similarity_search(body, k=3)
        examples = "\n\n".join([f"Email: {d.page_content}\nReply: {d.metadata['reply']}" for d in docs])
        
        
        prompt = messages = prompt_template.format_messages(
            sender=sender, subject=subject, body=body, retrieved_examples=examples
        )

        response = llm.predict(prompt)    
        
        result = {
            "reply_subject": f"Re: {subject}",
            "reply_body": response.content,
        }
        
        return func.HttpResponse(json.dumps(result), mimetype="application/json",status_code=200)
    except Exception as e:
        logging.error(str(e))
        return func.HttpResponse(json.dumps({"error": str(e)}), mimetype="application/json", status_code=500)



