import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = Path(
    os.getenv("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "knowledge_base"))
)
KNOWLEDGE_RAW_DIR = KNOWLEDGE_BASE_DIR / "raw"
KNOWLEDGE_INDEX_PATH = KNOWLEDGE_BASE_DIR / "index.json"
RAG_SOURCE_DIR = Path(
    os.getenv("RAG_SOURCE_DIR", str(Path.home() / "Desktop" / "rag_literature"))
)

PGVECTOR_DATABASE_URL = os.getenv(
    "PGVECTOR_DATABASE_URL",
    "postgresql://herg:herg@127.0.0.1:5432/herg_rag",
)
RAG_VECTOR_DIMENSION = int(os.getenv("RAG_VECTOR_DIMENSION", "1024"))
BGE_M3_MODEL = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
RAG_HYBRID_DENSE_WEIGHT = float(os.getenv("RAG_HYBRID_DENSE_WEIGHT", "0.72"))
RAG_HYBRID_SPARSE_WEIGHT = float(os.getenv("RAG_HYBRID_SPARSE_WEIGHT", "0.28"))

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.6-sol")
