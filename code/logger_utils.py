import os, sys, logging
from .config import LOG_DIR

#LOGGING per tracciare ciò che voglio
#Qui configuriamo un logger che scrive sia su file rag_cli.log che su terminale; Ogni evento (domanda, errore...) verrà registrato.

def setup_logger(session="default", name="multiagents_logger", level="INFO"):
    log_file = f"{session}_multiagent.log"
    level = getattr(logging, level.upper(), logging.INFO) 

    os.makedirs(LOG_DIR, exist_ok=True)  

    full_log_path = os.path.join(LOG_DIR, log_file) if not os.path.isabs(log_file) else log_file

    # Usa un nome logger univoco per sessione per evitare conflitti e duplicazioni
    unique_logger_name = f"{name}_{session}"
    logger = logging.getLogger(unique_logger_name)  
    logger.setLevel(level)  

    if not logger.handlers:
        # Handler su file
        file_handler = logging.FileHandler(full_log_path, encoding='utf-8') 
        file_handler.setFormatter(logging.Formatter(  
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)  

        # Handler su terminale
        console_handler = logging.StreamHandler(sys.stdout) 
        console_handler.setFormatter(logging.Formatter(
            "%(levelname)s - %(message)s"
        ))
        logger.addHandler(console_handler)

    return logger