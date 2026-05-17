import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Scraping ---
BASE_URL = "https://www.w3schools.com"
SITEMAP_URL = "https://www.w3schools.com/sitemap/sitemap_www.xml"
REQUEST_DELAY_MIN = 1.5
REQUEST_DELAY_MAX = 3.0
MAX_CONCURRENT = 5
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# --- Chunking ---
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# --- Embedding ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_BATCH_SIZE = 64

# --- Storage ---
CHROMA_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "w3schools")

# --- Servers ---
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# --- Anthropic ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set. "
        "Export it or add it to a .env file in the project root."
    )
