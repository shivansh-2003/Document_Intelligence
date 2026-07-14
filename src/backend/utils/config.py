# config.py
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")


WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "large-v3")      
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")    
WHISPER_BATCH   = int(os.getenv("WHISPER_BATCH", "16"))  