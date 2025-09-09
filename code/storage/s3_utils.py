#Gestisce la connessione a S3 e upload/download di singoli file.

import os
import boto3
from botocore.exceptions import NoCredentialsError
from ..config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME
from pathlib import Path
from botocore.exceptions import NoCredentialsError, ClientError

# --- Inizializza il client S3 ---
s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def upload_file_to_s3(local_path: Path | str, s3_key: str, logger=None):
    """Carica un singolo file locale su S3"""
    local_path = Path(local_path)
    try:
        s3_client.upload_file(str(local_path), S3_BUCKET_NAME, s3_key)
        if logger:  # log singolo solo se richiesto esplicitamente
            logger.info("✅ Upload %s → s3://%s/%s", local_path, S3_BUCKET_NAME, s3_key)

    except KeyboardInterrupt:
        if logger:
            logger.warning("⚠️ Upload interrotto dall'utente: %s", local_path)
        raise  # rilancia se vuoi che l'interruzione fermi anche il loop superiore

def sync_folder_to_s3(local_folder: Path | str, s3_prefix: str, logger):
    """Sincronizza tutti i file locali di una cartella verso S3 (upload).
       Silenzia i dettagli tecnici degli errori di credenziali.
    """
    local_folder = Path(local_folder)
    if not local_folder.exists():
        logger.warning("⚠️ Cartella locale %s non trovata, skip sync", local_folder)
        return

    count_uploaded = 0

    try:
        for root, _, files in os.walk(local_folder):
            for fname in files:
                local_path = Path(root) / fname
                rel = local_path.relative_to(local_folder).as_posix()
                s3_key = f"{s3_prefix.rstrip('/')}/{rel}"
                try:
                    upload_file_to_s3(local_path, s3_key)  # log interno opzionale
                    count_uploaded += 1
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    if error_code in ("InvalidAccessKeyId", "SignatureDoesNotMatch", "AuthorizationHeaderMalformed"):
                        logger.warning("⚠️ Sync S3 disabilitato (credenziali non valide).")
                        return
                    else:
                        logger.error("❌ Errore generico durante upload S3.")
                except KeyboardInterrupt:
                    logger.warning("⚠️ Upload interrotto dall'utente: %s", local_path)
                    raise
    except NoCredentialsError:
        logger.warning("⚠️ Sync S3 disabilitato (nessuna credenziale trovata).")
        return

    if count_uploaded > 0:
        logger.info("✅ Upload completato: %d file caricati su s3://%s/%s",
                    count_uploaded, S3_BUCKET_NAME, s3_prefix.rstrip('/'))
        



def sync_s3_to_folder(s3_prefix: str, local_folder: Path | str, logger):
    """Scarica tutti i file da S3 verso la cartella locale (download).
       Silenzia i dettagli tecnici degli errori di credenziali.
    """
    local_folder = Path(local_folder)
    local_folder.mkdir(parents=True, exist_ok=True)

    count_downloaded = 0

    try:
        paginator = s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(s3_prefix):].lstrip("/")
                if not rel:
                    continue
                target = local_folder / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    s3_client.download_file(S3_BUCKET_NAME, key, str(target))
                    count_downloaded += 1
                except ClientError:
                    logger.error("❌ Errore generico durante download da S3.")

        if count_downloaded > 0:
            logger.info("⬇️ Download completato: %d file scaricati da s3://%s/%s",
                        count_downloaded, S3_BUCKET_NAME, s3_prefix.rstrip('/'))

    except NoCredentialsError:
        logger.warning("⚠️ Sync S3 disabilitato (nessuna credenziale trovata).")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("InvalidAccessKeyId", "SignatureDoesNotMatch", "AuthorizationHeaderMalformed"):
            logger.warning("⚠️ Sync S3 disabilitato (credenziali non valide).")
        else:
            logger.error("❌ Errore generico durante sync da S3.")




