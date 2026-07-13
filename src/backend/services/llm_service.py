# llm_service.py — single Ollama entrypoint, text or text+image
import logging

import ollama
from pydantic import BaseModel

from utils.config import OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)

_client = ollama.Client(host=OLLAMA_HOST)


def generate(prompt: str, image_b64: str | None = None, schema: type[BaseModel] | None = None) -> str:
    """One prompt in, one text out. images=[b64] triggers vision path in qwen2.5vl.
    schema=<BaseModel> constrains decoding to that JSON shape (Ollama structured outputs)."""
    logger.info(
        "Calling %s (prompt_len=%d, with_image=%s, schema=%s)",
        OLLAMA_MODEL, len(prompt), bool(image_b64), schema.__name__ if schema else None,
    )
    try:
        resp = _client.chat(
            model=OLLAMA_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                **({"images": [image_b64]} if image_b64 else {}),
            }],
            **({"format": schema.model_json_schema()} if schema else {}),
        )
    except Exception:
        logger.error("Ollama call to %s failed", OLLAMA_MODEL, exc_info=True)
        raise
    text = resp["message"]["content"].strip()
    logger.info("Ollama response received (len=%d)", len(text))
    return text