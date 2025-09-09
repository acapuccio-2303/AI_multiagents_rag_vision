import chainlit as cl
import httpx
import os
from dotenv import load_dotenv
from utils.session import nuova_sessione_id
import requests

#Carico variabili da file .env
load_dotenv()

# URL backend FastAPI
BACKEND_PORT = os.getenv("BACKEND_PORT", "8000")
BACKEND_BASE = os.getenv("BACKEND_BASE", f"http://localhost:{BACKEND_PORT}")
BACKEND_URL = f"{BACKEND_BASE}/chat"
RESET_URL = f"{BACKEND_BASE}/reset"
UPLOAD_URL = f"{BACKEND_BASE}/upload"
INIT_URL = f"{BACKEND_BASE}/init_session"

# ---------------------------
# Inizio chat
# ---------------------------
@cl.on_chat_start
async def on_chat_start():
    global session_id
    try:
        res = requests.post(INIT_URL)
        session_id = res.json()["session_id"]
        if not session_id:
            await cl.Message(content="Errore: risposta senza session_id").send()
            return
        welcome_msg = await cl.Message(content=f"👋 **Ciao!** Sono pronto ad aiutarti.").send()
    except Exception as e:
        await cl.Message(content=f"Errore inizializzazione sessione: {e}").send()
        return

    await cl.Action(
        name="reset_chat",
        value="reset",
        label="🔄 Reset Chat",
        payload={"reason": "user_clicked_button"}
    ).send(for_id=welcome_msg.id)

# ---------------------------
# Reset chat
# ---------------------------
@cl.action_callback("reset_chat")
async def on_reset(action):
    global session_id
    try:
        if session_id is not None:
            async with httpx.AsyncClient() as client:
                await client.post(RESET_URL, params={"session_id": session_id})
        async with httpx.AsyncClient() as client:
            res = await client.post(INIT_URL)
            session_id = res.json().get("session_id")
        await cl.Message(content=f"🔄 **Chat resettata!**").send()
    except Exception as e:
        await cl.Message(content=f"Errore durante reset/init: {e}").send()

# -------------------------------------------
# Gestione messaggio + file upload
# -------------------------------------------
@cl.on_message
async def on_message(message: cl.Message):
    global session_id

    # File caricati
    if message.elements:
        for element in message.elements:
            if element.type in ["file", "image"]:
                try:
                    with open(element.path, "rb") as f:
                        file_bytes = f.read()
                    files = {"file": (element.name, file_bytes, "application/octet-stream")}
                    data = {"session_id": session_id}
                    async with httpx.AsyncClient() as client:
                        res = await client.post(UPLOAD_URL, files=files, data=data, timeout=120.0)
                    try:
                        res.raise_for_status()
                        json_data = res.json()
                        session_id = json_data.get("session_id", session_id)
                        msg = json_data.get("message", f"✅ File ricevuto ({res.status_code})")
                    except httpx.HTTPStatusError as exc:
                        try:
                            err_json = exc.response.json()
                            msg = err_json.get("detail", f"⚠️ Errore dal backend ({exc.response.status_code})")
                        except Exception:
                            msg = f"⚠️ Errore sconosciuto dal backend: {exc.response.text[:300]}"
                except Exception as e:
                    msg = f"⚠️ Errore durante l’upload: {type(e).__name__}"
                await cl.Message(content=msg).send()

    # Messaggio testo
    user_text = message.content
    await cl.Message(content="⏳ **Sto pensando...**").send()
    try:
        async with httpx.AsyncClient(timeout=80.0) as client:
            payload = {"session_id": session_id, "message": user_text}
            res = await client.post(BACKEND_URL, json=payload)
            data = res.json()
            session_id = data.get("session_id")
            risposta = data.get("risposta", "⚠️ Errore nella risposta.")
        await cl.Message(content=risposta, author="🤖 AI").send()
    except Exception as e:
        await cl.Message(content=f"⚠️ Errore di connessione al backend: {str(e)}").send()