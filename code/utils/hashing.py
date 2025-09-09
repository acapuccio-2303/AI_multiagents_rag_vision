import hashlib

# Calcolo l'hash del file PDF per invalidare il vectorstore se il PDF cambia
def hash_file(filepath):
    hasher = hashlib.sha256()         
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()