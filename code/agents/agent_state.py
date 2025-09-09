from langchain_community.chat_message_histories import ChatMessageHistory
from typing import TypedDict, Any, List
from langchain_core.runnables import Runnable

# Definizione stato tipizzato per il grafo 
class AgentState(TypedDict, total=False):
    input: str
    tipo: str             # Tipo di agente: "tecnico" o "generale" o "vision"
    output: str
    chat_memory: ChatMessageHistory
    session_id: str       # Identificatore sessione per caricare/salvare memoria
    rag_chain: Runnable   # c'è solo se si carica aun file e si userà agente tecnico
                          # Meglio Runnable di Any...+ splicito: stai dicendo che lì ci deve stare una catena/costrutto LangChain compatibile con .invoke, .stream, ecc.
    image_paths: List[str]     