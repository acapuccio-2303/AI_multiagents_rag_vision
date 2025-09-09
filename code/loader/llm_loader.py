import os, sys
from langchain_community.llms import LlamaCpp
import google.generativeai as genai
from ..config import GEMINI_API_KEY
import sys
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import ChatPromptValue



# Carica il modello LLM da file locale
def get_llm(model_path, logger=None):

    if not os.path.exists(model_path):
        logger.error(f"❌ Modello non trovato in: {model_path}")
        sys.exit(1)

    return LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        max_tokens=512,
        temperature=0.3,
        top_k=50,
        n_threads=8,
        n_gpu_layers=0,
        verbose=False
    )



# Modello API-based: Gemini (Flash o Pro)
def get_llm_API(model_name=None, logger=None):
    api_key = GEMINI_API_KEY
    
    if not api_key:
        if logger:
            logger.error("❌ Nessuna chiave API Gemini trovata. Assicurati che sia impostata in .env o config.py.")
        sys.exit(1)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        class GeminiLLM:
            def __init__(self, model):
                self.model = model

            def invoke(self, input):
                prompt = ""

                # 🔹 Caso 1: input = ChatPromptValue
                if isinstance(input, ChatPromptValue):
                    messages = input.messages
                    prompt = "\n".join(m.content for m in messages if hasattr(m, "content"))

                # 🔹 Caso 2: input = ( "messages", [...] )
                elif isinstance(input, tuple) and input[0] == "messages":
                    messages = input[1]
                    prompt = "\n".join(
                        m.content if isinstance(m, BaseMessage) else str(m)
                        for m in messages
                    )

                # 🔹 Caso 3: input = dict
                elif isinstance(input, dict):
                    prompt = input.get("input", "")

                # 🔹 Caso 4: input = stringa
                elif isinstance(input, str):
                    prompt = input

                else:
                    raise TypeError(f"Tipo di input non supportato: {type(input)}")

                # Chiamata a Gemini
                response = self.model.generate_content(prompt)
                return response.text if hasattr(response, "text") else str(response)

            def __call__(self, input):   
                return self.invoke(input)
            
        return GeminiLLM(model)

    except Exception as e:
        if logger:
            logger.exception("❌ Errore durante la configurazione del modello Gemini.")
        sys.exit(1)


#Qui non usiamo il wrapper GeminiLLM, ma direttamente GenerativeModel dell’SDK genai.
#Supporta input testuali e immagini.
def get_vlm_API(model_name=None, logger=None):
    """
    Restituisce un modello Gemini multimodale pronto per Vision Q&A.
    Usa direttamente l'SDK genai per generative multimodal.
    """
    if not GEMINI_API_KEY:
        if logger:
            logger.error("❌ Nessuna chiave API trovata per Gemini.")
        raise ValueError("API key Gemini mancante")
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name)
        if logger:
            logger.info(f"✅ Modello VLM '{model_name}' inizializzato")
        return model
    except Exception as e:
        if logger:
            logger.exception("❌ Errore inizializzazione VLM Gemini")
        raise e