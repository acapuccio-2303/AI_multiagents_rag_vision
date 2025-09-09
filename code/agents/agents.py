
# Imports
from ..agents.agent_state import AgentState
from ..utils.tools import cerca_contenuti, vlm_qna
from ..memory.chat_memory import get_memory
import re

# === ROUTER PRINCIPALE ===
#Funzione per creare un router che seleziona l'agente corretto in base alla query
def router(state: AgentState, llm_router, logger=None) -> AgentState:
    """
    Router intelligente con approccio ibrido:
    - Pattern first (veloce e gratuito)
    - Fallback su LLM se il pattern non è chiaro
    """

    if "tipo" in state:   # serve a non far ricalcolare il routing se è già stato deciso in precedenza
        return state

    # Normalizzazione testo: minuscole + rimozione punteggiatura
    testo = state["input"].lower()
    testo = re.sub(r"[^\w\s]", "", testo)
    
    rag_disponibile = state.get("rag_chain") is not None
    has_image = bool(state.get("image_paths"))

    # -------------------
    # 1) PATTERN FIRST
    # -------------------

    # IMMAGINE + PATTERN TESTUALE --> VISION
    if has_image:

    # --- Parole chiave per Vision (radici) ---
        parole_vision = [
            r"vedi",
            r"img",    
            r"immagin",    # immagine, immagini, immaginare
            r"foto",       # foto, fotografare, fotografia
            r"raffigur",   # raffigura, raffigurazione
            r"disegn",     # disegno, disegnare
        ]
        # Regex OR: intercetta radici (non solo la parola intera)
        pattern_vision = r"(" + "|".join(parole_vision) + r")"
        
        # Se c'è un'imagine e almeno una parola relativa nel testo → agente vision
        if re.search(pattern_vision, testo):
            state["tipo"] = "vision"
            logger.info("🕵️ E' stato scelto l'agente vision")
            return state

    # DOCUMENTO + PATTENR TESTUALE --> TECNICO
    if rag_disponibile:

        # --- Parole chiave rag (radici) ---
        parole_per_rag = [
            r"pdf", r"csv", r"txt", r"docx", r"documento", r"manuale", r"contenuto",
            r"riassum", r"analizz", r"esamin",
            r"cerc", r"trova", r"ricerc", r"informazioni",
            r"spieg", r"riassunto", r"file", r"parla", r"tratta"
        ]
        # Regex OR: intercetta radici (non solo la parola intera)
        pattern_tecniche = r"(" + "|".join(parole_per_rag) + r")" 

        # --- Pattern domande sul contenuto ---
        pattern_domande_contenuto = [
            "che cosa contiene",
            "qual è il contenuto",
            "fammi un riassunto",
            "spiega il documento",
        ]

        match_tecniche = re.search(pattern_tecniche, testo)               
        match_domande = any(p in testo for p in pattern_domande_contenuto)

        # Se c'è una rag_chain e almeno una keyword tecnica nel testo → tecnico
        if match_tecniche or match_domande:
            state["tipo"] = "tecnico"
            logger.info("🕵️ E' stato scelto l'agente tecnico")
            return state
    
    # Generale → se nessuna immagine e documento caricati 
    if not has_image and not rag_disponibile:
        state["tipo"] = "generale"
        logger.info("🕵️ E' stato scelto l'agente generale")
        return state


    # -------------------
    # 2) FALLBACK LLM
    # -------------------
    if llm_router:
        router_prompt = f"""
        Sei un router che decide quale agente deve rispondere:
        - Se la domanda riguarda un'immagine caricata (descriverla, analizzarla, capire cosa contiene) → "vision"
        - Se riguarda un documento caricato (contenuti di PDF, CSV, ecc.) → "tecnico"
        - Se è una domanda generale, senza legame con immagini o documenti → "generale"

        Rispondi SOLO con una delle tre parole: vision, tecnico, generale.

        Contesto sessione:
        - Immagini caricate: {bool(state.get("image_paths"))}
        - Documenti caricati: {bool(state.get("rag_chain"))}

        Messaggio utente: "{state['input']}"
        """
        try:
            decision = llm_router.invoke(router_prompt).strip().lower()
            if decision in ["vision", "tecnico", "generale"]:
                logger.info(f"🕵️ Il router intelligente ha scelto l'agente {decision}")
                state["tipo"] = decision
            else:
                logger.info("🕵️ Il router intelligente non è riuscito a scegliere l'agente da usare: uso agente generale")
                state["tipo"] = "generale"  # fallback sicuro
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Router LLM failed, fallback a Generale: {e}")
            state["tipo"] = "generale"
    else:
        # Se non hai passato un llm_router → fallback standard
        state["tipo"] = "generale"

    return state

    


# Funzione per decidere il nodo successivo in base al tipo
def agente_switch(state: AgentState) -> str:
    """
    Switch routing per passare al nodo corretto.
    """
    tipo = state.get("tipo")
    if tipo == "tecnico":
        return "tecnico"
    elif tipo == "vision":
        return "vision"
    return "generale"


# === AGENTE GENERALE ===
def run_generale(state: AgentState, llm=None, logger=None) -> AgentState:
    """
    Nodo agente generale.
    Risponde a domande generiche, esegue calcoli e definizioni.
    Memorizza cronologia in file specifico  per sessione.
    """
    input_text = state["input"]
    session_id = state["session_id"]

    # Carica cronologia della sessione, o crea nuova
    chat_memory = state.get("chat_memory", get_memory(session_id)) #testo....Human: 1+1AI: 2
    chat_memory.add_user_message(input_text)                       #<langgraph.graph.state.CompiledStateGraph object at 0x7b6566c41a00>

    result = llm.invoke(input_text)
    chat_memory.add_ai_message(result)                             #Testo.. Human: 1+1 Human: 1+1 AI: 2
    
    state.update({
        "output": result,
        "chat_memory": chat_memory,
        "tipo": "generale",
    })
    return state


# === AGENTE TECNICO ===
def run_tecnico(state: AgentState, llm=None, logger=None) -> AgentState:
    """
    Nodo agente tecnico.
    Usa RAG per rispondere a domande sui documenti o richieste specifiche.
    Può anche eseguire calcoli o definizioni.
    Memorizza cronologia in file specifico per sessione.
    """
    input_text = state["input"]
    session_id = state["session_id"]
    rag_chain = state.get("rag_chain")  # preso dallo state

    # Carica cronologia della sessione, o crea nuova
    chat_memory = state.get("chat_memory", get_memory(session_id))  #testo....Human: 1+1AI: 2

    # Logica per decidere quale funzione/tool usare in base all'input

    #RAG:
    if rag_chain is not None:
        try:
            chat_memory.add_user_message(input_text) 
            logger.info("🛠️ E' stato scelto il tool 'cerca_contenuti'")
            result = cerca_contenuti(input_text, rag_chain=rag_chain, chat_memory=chat_memory)

            if not result.strip() or result.strip().lower() == "non lo so":
                # fallback to LLM (se RAG non trova nulla uso gemini)
                logger.info("📄 RAG non ha trovato risultati, uso LLM come fallback")
                result = llm.invoke(input_text)
                

        except Exception as e:
            if logger:
                logger.warning(f"❌ Errore su RAG {str(e)}")
            result = llm.invoke(input_text)

    #No tools, no RAG: fallback to LLM
    else:
        result = llm.invoke(input_text)

    chat_memory.add_ai_message(result)

    #elif "definizione" in input_text or "cosa significa" in input_lower:
    #    # Richiesta definizione dizionario
    #    result = dizionario(input_text)

    # Ritorna stato aggiornato con output, memoria e session_id
    state.update({
        "output": result,
        "chat_memory": chat_memory,
        "tipo": "tecnico",
    })

    return state


# === AGENTE VISION ===


def run_vision(state: AgentState, vision_model=None, logger=None) -> AgentState:
    """
    Agente Vision: usa il tool vlm_qna per rispondere a domande sulle immagini.
    """

    logger.info("🛠️ E' stato scelto il tool 'vlm_qna'")
    input_text = state.get("input", "")
    session_id = state.get("session_id")

    chat_memory = state.get("chat_memory", get_memory(session_id))
    
    result = vlm_qna(
        query=input_text,
        image_paths=state.get("image_paths", []),
        vision_model=vision_model,
        chat_memory=chat_memory
    )

    state.update({
        "output": result,
        "chat_memory": chat_memory,
        "tipo": "vision",
    })
    return state