# config.py
import os

import ctranslate2

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")

_HAS_CUDA = ctranslate2.get_cuda_device_count() > 0

WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE", "cuda" if _HAS_CUDA else "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16" if _HAS_CUDA else "int8")
WHISPER_BATCH   = int(os.getenv("WHISPER_BATCH", "16"))