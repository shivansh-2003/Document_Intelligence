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
