import os
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import langchainhub as hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

app = FastAPI(title="CampConnect WhatsApp Bot")

# --- Configuration ---
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_ID", "979160471956454")
WEBHOOK_VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN", "webhook")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_FAISS_PATH="vectorstore/db_faiss"

# --- Load RAG Chain ---
try:
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    GROQ_MODEL_NAME = "llama-3.1-8b-instant"
    llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0.5, max_tokens=512, api_key=GROQ_API_KEY)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)
    print("RAG Pipeline Loaded Successfully.")
except Exception as e:
    print(f"Error loading RAG pipeline: {e}")
    rag_chain = None

def send_whatsapp_message(phone_number: str, text: str):
    """ Helper to send WhatsApp messages using Meta Graph API. """
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"body": text}
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Failed to send message: {response.text}")
    print(f"Sent message properly! Status: {response.status_code}")
    return response

@app.get("/webhook")
async def verify_webhook(request: Request):
    """ 
    Meta Webhook Verification endpoint.
    Meta explicitly expects the 'hub.challenge' to be returned as plain text. 
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == WEBHOOK_VERIFY_TOKEN:
            print("Webhook verified successfully!")
            return PlainTextResponse(content=challenge)
    
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def handle_whatsapp_message(request: Request):
    """
    Main endpoint to receive webhook payloads from WhatsApp.
    We extract the user message and run it through the RAG Chain.
    """
    body = await request.json()
    
    try:
        # Check if the payload matches WhatsApp messaging schema
        entry = body.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        
        # Check if it's a message
        if "messages" in value:
            message = value["messages"][0]
            phone_number = message["from"]  # User's phone number
            text = message.get("text", {}).get("body", "")  # Text content
            
            if text and rag_chain:
                print(f"Incoming WhatsApp message from {phone_number}: {text}")
                
                # Fetch RAG Response
                response = rag_chain.invoke({'input': text})
                result = response["answer"]
                
                # Dispatch answer to user
                send_whatsapp_message(phone_number, result)
                
        # Meta expects a 200 OK simply indicating receipt.
        return {"status": "success"}
        
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Optional execution block when script is ran directly.
    uvicorn.run(app, host="0.0.0.0", port=8000)
