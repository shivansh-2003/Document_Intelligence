# core/config.py
import os

import ctranslate2
from dotenv import load_dotenv

load_dotenv()  # reads backend/.env if present -- real secrets never get hardcoded here

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")

_HAS_CUDA = ctranslate2.get_cuda_device_count() > 0

WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE", "cuda" if _HAS_CUDA else "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16" if _HAS_CUDA else "int8")
WHISPER_BATCH   = int(os.getenv("WHISPER_BATCH", "16"))

# dev default matches docker-compose.yml -- override via env in every real deployment
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://doc_intel:doc_intel@localhost:5432/doc_intel"
)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me-32-bytes-min")  # ponytail: dev-only default
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # None for local dev without auth

# retrieval — see context/retrieval.md
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "8"))     # chunks kept after rerank
FANOUT_TOP_K = int(os.getenv("FANOUT_TOP_K", "40"))    # per-variant hybrid_search limit, pre-merge

RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "180"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

VALKEY_URL = os.getenv("VALKEY_URL", "redis://localhost:6379/0")  # Valkey speaks RESP; redis:// scheme works unchanged
CACHE_TTL_EMBEDDING = int(os.getenv("CACHE_TTL_EMBEDDING", str(7 * 24 * 3600)))
CACHE_TTL_RETRIEVAL = int(os.getenv("CACHE_TTL_RETRIEVAL", str(3600)))
CACHE_TTL_RESPONSE = int(os.getenv("CACHE_TTL_RESPONSE", str(3600)))
CACHE_TTL_CONVERSATION = int(os.getenv("CACHE_TTL_CONVERSATION", str(3600)))
RESPONSE_CACHE_SIM_THRESHOLD = float(os.getenv("RESPONSE_CACHE_SIM_THRESHOLD", "0.95"))
