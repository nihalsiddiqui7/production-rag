from pathlib import Path


# 1. Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent


# 2. Define the directory for storing data

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

#3. CHUNK SETTINGS
CHUNK_SIZE = 800 # Number of characters per chunk
CHUNK_OVERLAP = 170  # Number of characters to overlap between chunks
