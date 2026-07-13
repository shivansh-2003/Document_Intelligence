# llm_service.py — single Ollama entrypoint, text or text+image
import ollama

from utils.config import OLLAMA_HOST, OLLAMA_MODEL

_client = ollama.Client(host=OLLAMA_HOST)


def generate(prompt: str, image_b64: str | None = None) -> str:
    """One prompt in, one text out. images=[b64] triggers vision path in qwen2.5vl."""
    resp = _client.chat(
        model=OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            **({"images": [image_b64]} if image_b64 else {}),
        }],
    )
    return resp["message"]["content"].strip()