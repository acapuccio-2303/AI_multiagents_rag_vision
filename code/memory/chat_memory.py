
import os, json
from langchain_community.chat_message_histories import ChatMessageHistory
from ..config import MEM_DIR

# Carica la memoria conversazionale per una sessione

#funzioni per recuperare e salvare la memoria per ogni sessione utente

def get_memory(session_id: str) -> ChatMessageHistory:
    """
    Recupera la cronologia chat salvata per una data sessione.
    Se non esiste, crea una nuova cronologia vuota.
    """
    #Creo la cartella per le memorie se non esiste
    os.makedirs(MEM_DIR, exist_ok=True)
    path = os.path.join(MEM_DIR, f"{session_id}.json")
    memory = ChatMessageHistory()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:  
            for role, content in json.load(f):      
                if role == "human":
                    memory.add_user_message(content)
                else:
                    memory.add_ai_message(content)
    return memory


def save_memory(session_id: str, memory: ChatMessageHistory):
    """
    Salva la cronologia chat per la sessione in un file JSON.
    """
    os.makedirs(MEM_DIR, exist_ok=True)
    path = os.path.join(MEM_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(              
            [["human" if m.type == "human" else "ai", m.content] for m in memory.messages],
            f,
            ensure_ascii=False,  
            indent=2             
        )