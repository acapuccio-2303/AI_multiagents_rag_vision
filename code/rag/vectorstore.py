

import os
import humanize
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai




#MODELLO DI EMBEDING GOOLGE - OGGI NON PIù GRATUITO

class GoogleGenerativeEmbeddings(Embeddings):
    def __init__(self, model: str = "models/text-embedding-004", api_key: str = None): #Modello di embedding di Gemini scelto
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

def get_vectorstore_multidoc(docs=None, vectors_path=None, logger=None, vectorstore_esistente=None):
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


## === CREAZIONE DATABASE VETTORIALE FAISS (VERSIONE HUGGING FACE) ===

'''
#import os
#import humanize
#from langchain_community.vectorstores import FAISS
#from langchain_huggingface import HuggingFaceEmbeddings
#from ..config import EMBEDDING_MODEL_DIR


#Crea la cartella se non esiste
os.makedirs(EMBEDDING_MODEL_DIR, exist_ok=True)

# === CREAZIONE DATABASE VETTORIALE FAISS (VERSIONE HUGGING FACE) ===


class LocalHuggingFaceEmbeddings:
    """
    Wrapper per Hugging Face Embeddings basato su 'intfloat/multilingual-e5-small'
    """
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", model_dir: str = EMBEDDING_MODEL_DIR):
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(
        model_name=self.model_name,
        model_kwargs={"cache_folder": model_dir}  # forza cartella locale
        )

    def embed_documents(self, texts):
        """Restituisce gli embedding di una lista di testi"""
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        """Restituisce l'embedding di una singola query"""
        return self.model.embed_query(text)

def get_vectorstore_multidoc_local(
    docs=None,
    vectors_path=None,
    logger=None,
    vectorstore_esistente=None
):
    """
    Crea o aggiorna un vectorstore FAISS.
    Se vectorstore_esistente è passato, aggiunge i nuovi documenti invece di ricrearlo da zero.
    """
    embeddings = LocalHuggingFaceEmbeddings()

    index_file_path = os.path.join(vectors_path, "index.faiss")

    # Caso 1: aggiorno un vectorstore già in memoria
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
        vectorstore = FAISS.load_local(vectors_path, embeddings.model, allow_dangerous_deserialization=True)
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
        raise ValueError("Documenti non forniti per creare un nuovo vectorstore.")

    vectorstore = FAISS.from_documents(docs, embeddings.model)
    vectorstore.save_local(vectors_path)

    file_size = os.path.getsize(index_file_path)
    num_docs = len(vectorstore.index_to_docstore_id)

    if logger:
        logger.info("--------")
        logger.info(f"🆕 Vectorstore creato e salvato in: {vectors_path}")
        logger.info(f"📊 Contiene circa {num_docs} documenti / chunk")
        logger.info(f"💾 Dimensione file FAISS: {humanize.naturalsize(file_size)}")
        logger.info("--------")

    return vectorstore

'''