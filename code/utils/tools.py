import re
from difflib import get_close_matches

# === TOOL: Dizionario con definizioni semplici e fuzzy matching ===
def dizionario(termine: str) -> str:
    """
    Cerca una definizione per un termine nel dizionario interno.
    Usa get_close_matches per trovare la chiave più simile.
    """
    definizioni = {
        "agente": "In AI, un agente è un'entità che percepisce il suo ambiente e agisce su di esso.",
        "intelligenza artificiale": "L'intelligenza artificiale è il ramo dell'informatica che si occupa di creare sistemi capaci di simulare l'intelligenza umana.",
    }
    termine = termine.lower().strip()
    chiavi_simili = get_close_matches(termine, definizioni.keys(), n=1, cutoff=0.5)
    if chiavi_simili:
        return definizioni[chiavi_simili[0]]
    return f"Nessuna definizione trovata per '{termine}'."


# TOOL come funzione RAG, per ricerca semplice nel knowledge base PDF
def cerca_contenuti(query: str, rag_chain=None, chat_memory=None) -> str:
 
    """
    Cerca contenuti nel database vettoriale RAG usando anche la memoria della chat se disponibile.
    Ritorna solo il testo della risposta (campo "answer") come stringa.
    """
    try:
        chat_memory = [] if chat_memory is None else chat_memory.messages

        res = rag_chain.invoke({
            "input": query,
            "chat_history": chat_memory
        })

        # Estraggo solo il testo finale
        if isinstance(res, dict):
            return res.get("answer", "⚠️ Nessuna risposta trovata.")
        return str(res)

    except Exception as e:
        return f"Errore RAG: {e}"
    



# === TOOL: Vision Q&A con VLM ===
def vlm_qna(query: str, image_paths: list[str], vision_model, chat_memory=None) -> str:
    """
    Esegue Q&A multimodale su una o più immagini con Gemini VLM.
    - query: testo della domanda
    - image_paths: lista percorsi immagini (usa solo l'ultima di default)
    - vision_model: modello VLM caricato
    - chat_memory: memoria conversazionale opzionale
    """
    try:
        # Prendo solo l’ultima immagine caricata
        paths = image_paths[-1:] if image_paths else []
        if not paths:
            return "⚠️ Nessuna immagine caricata."

        if chat_memory:
            chat_memory.add_user_message(query)

        # Costruisco input multimodale
        parts = [{"text": query}]
        for p in paths:
            with open(p, "rb") as f:
                parts.append({
                    "mime_type": "image/jpeg",  # TODO: rilevare MIME dinamicamente
                    "data": f.read()
                })

        # Chiamata al modello VLM
        response = vision_model.generate_content(parts)
        result = response.text if hasattr(response, "text") else str(response)

        if chat_memory:
            chat_memory.add_ai_message(result)

        return result

    except Exception as e:
        return f"❌ Errore VLM Q&A: {str(e)}"    