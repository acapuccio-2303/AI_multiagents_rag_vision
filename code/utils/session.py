import uuid

# === FUNZIONE PER GENERARE ID SESSIONE UNIVOCO ===
def nuova_sessione_id() -> str:
    """
    Genera un identificatore unico universale (UUID) come stringa.
    Serve per tracciare singole sessioni utente distinte.
    """
    session_id = str(uuid.uuid4())
    return session_id
