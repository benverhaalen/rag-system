import os
from pathlib import Path
from dotenv import load_dotenv

# loads env variables
load_dotenv()

# project root
BASE_DIR = Path(__file__).parent.parent

# API config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# vector store config
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma"))

# creating directories if they dont exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_PERSIST_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)