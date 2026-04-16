"""
whatsapp_bot.py
---------------
CampConnect WhatsApp Bot — powered by the Smart 3-Phase RAG Navigator.

Webhook flow:
  1. Meta sends a POST to /webhook with the user's message.
  2. We run it through smart_rag.smart_query() (FAISS → Web → LLM).
  3. We send the formatted answer back via the WhatsApp Graph API.
"""

import os
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

# Load environment variables
load_dotenv()

app = FastAPI(title="CampConnect WhatsApp Bot - Smart Navigator")

# --- Configuration ---
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_ID", "1064997016695390")
WEBHOOK_VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN", "webhook")


# ---------------------------------------------------------------------------
# WhatsApp sender
# ---------------------------------------------------------------------------

def send_whatsapp_message(phone_number: str, text: str):
    """Send a WhatsApp text message via the Meta Graph API."""
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"body": text},
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"[WhatsApp] Failed to send message: {response.text}")
    else:
        print(f"[WhatsApp] Message sent ✓ (status {response.status_code})")
    return response


# ---------------------------------------------------------------------------
# Webhook endpoints
# ---------------------------------------------------------------------------

@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Meta Webhook Verification endpoint.
    Returns 'hub.challenge' as plain text so Meta confirms the callback URL.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == WEBHOOK_VERIFY_TOKEN:
        print("[Webhook] Verification successful!")
        return PlainTextResponse(content=challenge)

    raise HTTPException(status_code=403, detail="Webhook verification failed")


@app.post("/webhook")
async def handle_whatsapp_message(request: Request):
    """
    Receive incoming WhatsApp messages and respond using the Smart RAG pipeline.
    """
    body = await request.json()

    try:
        entry = body.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})

        if "messages" not in value:
            # Delivery receipts, status updates, etc. — acknowledge and move on
            return {"status": "ok", "detail": "non-message event"}

        message = value["messages"][0]
        phone_number = message["from"]
        text = message.get("text", {}).get("body", "").strip()

        if not text:
            return {"status": "ok", "detail": "empty message"}

        print(f"[WhatsApp] Incoming from {phone_number}: {text}")

        # ── Smart RAG pipeline ──────────────────────────────────────────────
        from smart_rag import smart_query
        result = smart_query(text)

        answer = result["answer"]
        source = result["source"]

        # Append a small source footer so the user knows where the info came from
        reply = f"{answer}\n\n_Source: {source}_"

        send_whatsapp_message(phone_number, reply)

    except IndexError:
        # Payload without expected structure — safe to ignore
        pass
    except Exception as e:
        print(f"[WhatsApp] Error processing message: {e}")
        return {"status": "error", "message": str(e)}

    # Meta requires a 200 OK to acknowledge receipt
    return {"status": "success"}


@app.get("/health")
async def health():
    """Simple health-check endpoint for Render."""
    return {"status": "healthy", "service": "CampConnect WhatsApp Bot"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
