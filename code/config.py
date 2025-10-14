import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   
load_dotenv()   
                             
UPLOAD_DIR = BASE_DIR/"models_e_docs"
MAX_FILE_SIZE_BYTE = 20 * 1024 * 1024   # Limite massimo in byte
VECTORSTORE_DIR = BASE_DIR/"models_e_docs"/"vectorstore"
EMBEDDING_MODEL_DIR = BASE_DIR/"embedding_model"
MEM_DIR = BASE_DIR/"memorie_utenti"
LOG_DIR = BASE_DIR/"logs"
LOG_LEVEL = "INFO"                       # Choose: DEBUG, INFO, WARNING, ERROR, CRITICAL

for d in [UPLOAD_DIR, VECTORSTORE_DIR, MEM_DIR, LOG_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)


# === Gemini ===
LLM_MODEL_NAME="gemini-2.5-flash" #"gemini-flash-latest"                          #run check_models_available_gemini.py
VLM_MODEL_NAME="gemini-2.5-flash"                        
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === AWS S3 ===
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")          # segreta -> .env / Railway vars
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")  # segreta -> .env / Railway vars
S3_BUCKET_NAME="buckets3-ninni" 
AWS_REGION="eu-north-1"

