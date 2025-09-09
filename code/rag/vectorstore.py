import os
import humanize
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai

#CREAZIONE DATABASE VETTORIALE FAISS

class GoogleGenerativeEmbeddings(Embeddings):
    def __init__(self, model: str = "models/embedding-001", api_key: str = None): #Modello di embedding di Gemini scelto
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY non impostata.")
        genai.configure(api_key=self.api_key)

    def embed_documents(self, texts):
        """Restituisce gli embedding di una lista di testi"""
        embeddings = []
        for text in texts:
            response = genai.embed_content(model=self.model, content=text)
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, text):
        """Restituisce l'embedding di una singola query"""
        response = genai.embed_content(model=self.model, content=text)
        return response["embedding"]


#Versione con GoogleGenerativeEmbeddings (modello Gemini)
def get_vectorstore_multidoc(docs=None, vectors_path=None, logger=None, FILE_DIR=None, vectorstore_esistente=None):
    """
    Crea o aggiorna un vectorstore FAISS.
    Se vectorstore_esistente è passato, aggiunge i nuovi documenti invece di ricrearlo da zero.
    """
    embeddings = GoogleGenerativeEmbeddings()

    index_file_path = os.path.join(vectors_path, "index.faiss")

    # Caso 1: aggiorno un vectorstore già in memoria (più file cumulativi)
    if vectorstore_esistente is not None and docs:
        vectorstore_esistente.add_documents(docs)
        if logger:
            logger.info("--------")
            logger.info(f"➕ Aggiunti {len(docs)} documenti/chunk al vectorstore esistente.")
            logger.info(f"📊 Ora contiene circa {len(vectorstore_esistente.index_to_docstore_id)} documenti / chunk")
            logger.info("--------")
        vectorstore_esistente.save_local(vectors_path)
        return vectorstore_esistente

    # Caso 2: carico da disco se già esiste
    if os.path.exists(vectors_path) and os.path.exists(index_file_path):
        vectorstore = FAISS.load_local(vectors_path, embeddings, allow_dangerous_deserialization=True)
        file_size = os.path.getsize(index_file_path)
        num_docs = len(vectorstore.index_to_docstore_id)
        if logger:
            logger.info("--------")
            logger.info(f"📦 Vectorstore trovato e caricato da: {vectors_path}")
            logger.info(f"📊 Contiene circa {num_docs} documenti / chunk")
            logger.info(f"💾 Dimensione file FAISS: {humanize.naturalsize(file_size)}")
            logger.info("--------")
        return vectorstore

    # Caso 3: non esiste → lo creo da zero
    if docs is None:
        logger.info("--------")
        raise ValueError("Documenti non forniti per creare un nuovo vectorstore.")

    vectorstore = FAISS.from_documents(docs, embeddings) 
    vectorstore.save_local(vectors_path)

    index_file_path = os.path.join(vectors_path, "index.faiss")
    file_size = os.path.getsize(index_file_path)
    num_docs = len(vectorstore.index_to_docstore_id)

    if logger:
        logger.info("--------")
        logger.info(f"🆕 Vectorstore creato e salvato in: {vectors_path}")
        logger.info(f"📊 Contiene circa {num_docs} documenti / chunk")
        logger.info(f"💾 Dimensione file FAISS: {humanize.naturalsize(file_size)}")
        logger.info("--------")
    return vectorstore


