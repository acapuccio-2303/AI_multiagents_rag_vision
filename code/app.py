from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import uvicorn
import asyncio
import os, time
from contextlib import asynccontextmanager

from .logger_utils import setup_logger
from .config import LOG_DIR, VECTORSTORE_DIR, UPLOAD_DIR, MAX_FILE_SIZE_BYTE, LOG_LEVEL
from .utils.session import nuova_sessione_id
from .utils.hashing import hash_file
from .rag.loader_doc import load_documents
from .rag.vectorstore import get_vectorstore_multidoc
from .rag.rag_chain import build_rag_chain
from .memory.chat_memory import get_memory, save_memory
from .loader.llm_loader import get_llm_API, get_vlm_API
from .agents.build_graph import build_graph
from .storage.s3_utils import sync_folder_to_s3, sync_s3_to_folder

# Lifespan: sync iniziale/finale con S3
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = setup_logger(session="startup", level=LOG_LEVEL)
    logger.info("⬇️ Sync iniziale da S3...")
    try:
        sync_s3_to_folder("logs", LOG_DIR, logger)
        sync_s3_to_folder("models_e_docs", UPLOAD_DIR, logger)
        logger.info("✅ Sync iniziale completata")
    except Exception:
        logger.warning("⚠️ Sync iniziale da S3 saltato.")

    yield

    logger.info("⬆️ Sync finale su S3 prima dello shutdown...")
    try:
        sync_folder_to_s3(LOG_DIR, "logs", logger)
        sync_folder_to_s3(UPLOAD_DIR, "models_e_docs", logger)
        logger.info("✅ Sync finale completata")
    except Exception:
        logger.warning("⚠️ Sync finale su S3 saltato.")

# App FastAPI
app = FastAPI(title="Conversational Multiagent API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache in RAM
memory_cache = {}
hash_cache = {}
loggers_cache = {}
grafi_cache = {}

# Inizializzazione globale
log_level = LOG_LEVEL
logger = setup_logger(session="Inizializzazione", level=log_level)
logger.info("🚀 Inizializzazione backend...")
llm = get_llm_API(model_name="gemini-1.5-flash", logger=logger)
vlm = get_vlm_API(model_name="gemini-1.5-flash", logger=logger)
grafo_default = build_graph(llm=llm, vision_model=vlm, logger=logger)
grafi_cache["default"] = {"grafo": grafo_default, "rag_chain": None, "image_paths": []}

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

# Endpoint: Init session
@app.post("/init_session")
def init_session():
    session_id = nuova_sessione_id()
    loggers_cache[session_id] = setup_logger(session=session_id, level=log_level)
    memory_cache[session_id] = get_memory(session_id)
    grafi_cache[session_id] = {"grafo": grafo_default, "rag_chain": None, "vectorstore": None, "image_paths": []}
    hash_cache[session_id] = None
    return {"session_id": session_id}

# Endpoint: Chat
@app.post("/chat")
async def chat_endpoint(data: ChatRequest):
    session_id = data.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id obbligatorio")

    if session_id not in loggers_cache:
        loggers_cache[session_id] = setup_logger(session=session_id, level=log_level)
    logger = loggers_cache[session_id]

    memory = memory_cache.get(session_id) or get_memory(session_id)
    memory_cache[session_id] = memory

    if (session_id, "initialized") not in loggers_cache:
        if not memory.messages:
            logger.info(f"🟢 Avvio sessione: '{session_id}'")
        else:
            logger.info(f"📚 Memoria iniziale: {len(memory.messages)} messaggi")
        loggers_cache[(session_id, "initialized")] = True

    message = data.message.strip()
    logger.info(f"👤 Domanda: {message}")
    start = time.perf_counter()

    try:
        grafo_entry = grafi_cache.get(session_id, grafi_cache["default"])
        grafo = grafo_entry["grafo"]
        rag_chain = grafo_entry["rag_chain"]
        image_paths = grafo_entry.get("image_paths", [])

        risposta = await asyncio.wait_for(
            asyncio.to_thread(
                grafo.invoke,
                {
                    "input": message,
                    "session_id": session_id,
                    "chat_memory": memory,
                    "rag_chain": rag_chain,
                    "image_paths": image_paths[-1:],
                },
            ),
            timeout=30,
        )

        elapsed = time.perf_counter() - start
        logger.info(f"🤖 Risposta: {risposta['output']}")
        logger.info(f"⏳ Tempo di risposta: {elapsed:.3f}s")

        memory_cache[session_id] = risposta["chat_memory"]
        save_memory(session_id, risposta["chat_memory"])

        return {
            "session_id": session_id,
            "agente": risposta["tipo"],
            "risposta": risposta["output"],
            "elapsed_time": round(elapsed, 3),
        }

    except asyncio.TimeoutError:
        msg = "⚠️ Timeout: il modello non ha risposto in tempo."
    except Exception as e:
        logger.error(f"❌ Errore durante chat: {e}", exc_info=True)
        msg = f"⚠️ Errore: {str(e)}"

    return {"session_id": session_id, "agente": "errore", "risposta": msg, "elapsed_time": 0.0}

# Endpoint: Reset session
@app.post("/reset")
async def reset_session(session_id: str):
    if session_id not in loggers_cache:
        loggers_cache[session_id] = setup_logger(session=session_id, level=log_level)
    logger = loggers_cache[session_id]

    loggers_cache.pop((session_id, "initialized"), None)
    memory_cache.pop(session_id, None)
    grafi_cache.pop(session_id, None)
    hash_cache.pop(session_id, None)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "memorie_utenti", f"{session_id}.json")
    
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"RESET: Sessione {session_id} cancellata")
    else:
        logger.info(f"RESET: Nessun file memoria per {session_id}")

    loggers_cache[session_id] = setup_logger(session=session_id, level=log_level)
    memory_cache[session_id] = get_memory(session_id)
    grafi_cache[session_id] = {"grafo": grafo_default, "rag_chain": None, "vectorstore": None, "image_paths": []}

    return {"status": "reset", "message": f"Sessione {session_id} cancellata"}

# Endpoint: Upload file
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id mancante")

    if session_id not in loggers_cache:
        loggers_cache[session_id] = setup_logger(session=session_id, level="INFO")
    logger = loggers_cache[session_id]

    if session_id not in memory_cache:
        memory_cache[session_id] = get_memory(session_id)

    if session_id not in grafi_cache:
        grafi_cache[session_id] = {"grafo": grafi_cache["default"]["grafo"], "rag_chain": None, "vectorstore": None, "image_paths": []}

    try:
        filename = file.filename
        FILE_DIR = os.path.join(UPLOAD_DIR, filename)
        ext = filename.lower().split('.')[-1]

        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_BYTE:
            msg = f"❌ File troppo grande (max {MAX_FILE_SIZE_BYTE / 1024 / 1024} MB)"
            logger.warning(msg)
            raise HTTPException(status_code=413, detail=msg)

        with open(FILE_DIR, "wb") as buffer:
            buffer.write(contents)
        file_size = os.path.getsize(FILE_DIR)
        logger.info(f"📂 File '{filename}' ({file_size} byte) caricato in sessione {session_id}")

        # Immagini
        if ext in ["png", "jpg", "jpeg", "bmp", "webp"]:
            grafi_cache[session_id].setdefault("image_paths", []).append(FILE_DIR)
            grafo = build_graph(llm=llm, vision_model=vlm, logger=logger)
            grafi_cache[session_id]["grafo"] = grafo
            hash_cache[session_id] = None
            return {"message": f"✅ Immagine '{filename}' caricata", "size": file_size, "session_id": session_id, "type": "image"}

        # Documenti
        if ext not in ["pdf", "txt", "docx", "csv"]:
            raise HTTPException(status_code=415, detail=f"Formato non supportato: .{ext}")

        pdf_hash = hash_file(FILE_DIR)
        vectorstore_path = f"{VECTORSTORE_DIR}/vectorstore_faiss_{session_id}"
        docs = load_documents(logger=logger, FILE_DIR=FILE_DIR)
        if not docs:
            raise HTTPException(status_code=400, detail=f"⚠️ Nessun contenuto valido in {filename}")

        if session_id in grafi_cache and "vectorstore" in grafi_cache[session_id]:
            vectorstore = get_vectorstore_multidoc(docs, vectorstore_path, logger, FILE_DIR, grafi_cache[session_id]["vectorstore"])
            logger.info("🔁 Vectorstore aggiornato.")
        else:
            vectorstore = get_vectorstore_multidoc(docs, vectorstore_path, logger, FILE_DIR)
            logger.info("🆕 Vectorstore creato.")

        rag_chain = build_rag_chain(llm, vectorstore)
        grafo = build_graph(llm=llm, vision_model=vlm, logger=logger)
        grafi_cache[session_id] = {"grafo": grafo, "rag_chain": rag_chain, "vectorstore": vectorstore}
        hash_cache[session_id] = pdf_hash

        sync_folder_to_s3(UPLOAD_DIR, "models_e_docs", logger)
        sync_folder_to_s3(LOG_DIR, "logs", logger)
        logger.info("✅ Cartelle sincronizzate su S3")

        return {"message": f"✅ File '{filename}' caricato", "size": file_size, "session_id": session_id, "type": "document"}

    except Exception as e:
        logger.error(f"❌ Errore upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Errore durante upload.")
