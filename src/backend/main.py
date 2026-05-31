"""
main.py
───────
FastAPI application factory.

Start the server:
    uvicorn src.backend.main:app --reload --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.ingest import ingest_router


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown hooks
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-warm anything expensive here (DB pool, model load, etc.)
    print("[startup] Document Intelligence API is ready.")
    yield
    # Shutdown: close connections, flush buffers, etc.
    print("[shutdown] Cleaning up.")


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Document Intelligence API",
    description = (
        "Multi-format document ingestion pipeline.  "
        "Parses, chunks, and metadata-tags any supported resource "
        "(PDF · DOCX · PPTX · TXT · URL) and returns retrieval-ready chunks."
    ),
    version     = "0.1.0",
    lifespan    = lifespan,
)

# ── CORS (adjust origins for production) ────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])


# ─────────────────────────────────────────────────────────────────────────────
# Root health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "Document Intelligence API"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}