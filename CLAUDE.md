# Document Intelligence — Claude Agent Instructions

> **Start here.** This file provides context and conventions for working on the Document Intelligence system.

---

## 🗺️ Essential Context Files

Before making any changes, **read these files** in the `context/` folder:

1. **`context/architecture.md`** — System design, component responsibilities, data flow, and key decisions
2. **`context/folder.md`** — Complete folder structure and what lives where

These files contain the architectural blueprint and organizational map for the entire codebase.

---

## 🎯 Project Overview

**Document Intelligence** is an LLM-powered document processing pipeline that:
- Ingests multi-format documents (PDF, DOCX, HTML, images)
- Extracts text, tables, and figures using specialized extractors
- Analyzes content using Claude API (classification, summarization, NER, Q&A)
- Exposes results via a FastAPI REST API
- Processes jobs asynchronously using Celery + Redis

**Tech Stack:** FastAPI, SQLAlchemy 2.x (async), Celery, Redis, PostgreSQL, Anthropic Claude API

---

## 📐 Architecture Quick Reference

```
Client → FastAPI → Celery Workers → Pipeline (Ingestion → Extraction → Analysis) → PostgreSQL
```

**Core Components:**
- `src/ingestion/` — File intake, MIME detection, chunking
- `src/extraction/` — Format-specific extraction (PDF/DOCX/HTML/images)
- `src/analysis/` — LLM-powered intelligence via Claude API
- `src/api/` — FastAPI routes and dependency injection
- `src/models/` — Single source of truth for schemas (Pydantic + SQLAlchemy)

See `context/architecture.md` for complete details.

---

## 🛠️ Working with This Codebase

### Before You Code

1. **Read context files first** — `context/architecture.md` and `context/folder.md`
2. **Check layer-specific CLAUDE.md** — Each major folder (`src/ingestion/`, `src/extraction/`, etc.) may have its own `CLAUDE.md` with layer-specific conventions
3. **Understand the data flow** — Reference the "Data Flow (Happy Path)" section in `context/architecture.md`

### Key Conventions

- **Config:** Always use `src/config.py` (Pydantic settings), never raw `os.environ`
- **LLM calls:** Use `src/analysis/llm_client.py` exclusively — single Anthropic client with retry/caching
- **Schemas:** Define in `src/models/` only to prevent drift between layers
- **Async DB:** All DB operations use SQLAlchemy 2.x async sessions
- **Prompt templates:** Store in `src/analysis/prompts/` as `.txt` files
- **Testing:** Mirror `src/` structure in `tests/`, use fixtures from `tests/conftest.py`

### When Adding Features

1. **Identify the layer** — Ingestion, Extraction, Analysis, or API?
2. **Follow existing patterns** — Check similar modules in that layer
3. **Update schemas** — Add/modify models in `src/models/` if needed
4. **Add tests** — Write tests in the corresponding `tests/` subfolder
5. **Update context docs** — Propose updates to `context/architecture.md` or `context/folder.md` if you introduce new patterns

### When Debugging

1. **Check job status** — `Document.status` tracks pipeline progress (`PENDING` → `PROCESSING` → `COMPLETE` / `FAILED`)
2. **Review Celery logs** — Workers log extraction and analysis steps
3. **Inspect DB state** — Query `documents`, `chunks`, and `analysis_results` tables
4. **Trace LLM calls** — `llm_client.py` logs token usage and Claude API responses

---

## 🚀 Common Tasks

### Run the API locally
```bash
uvicorn src.api.main:app --reload
```

### Start Celery worker
```bash
celery -A src.worker worker --loglevel=info
```

### Run tests
```bash
pytest tests/
```

### Run tests for specific layer
```bash
pytest tests/ingestion/
pytest tests/api/
```

### Database migrations
```bash
alembic upgrade head
alembic revision --autogenerate -m "description"
```

---

## 🔐 Environment Setup

Required environment variables (see `context/architecture.md` for full list):

- `ANTHROPIC_API_KEY` — Claude API access
- `DATABASE_URL` — PostgreSQL connection string
- `REDIS_URL` — Celery broker/backend
- `API_KEY_SECRET` — API authentication
- `MAX_CHUNK_SIZE` — Chunk token limit (default 1000)
- `ENVIRONMENT` — `development` or `production`

---

## 🧠 Agent Guidelines

### When I Ask You To...

**"Add a new document format":**
1. Add loader to `src/ingestion/loaders/`
2. Update MIME routing in `src/ingestion/router.py`
3. Add extractor to `src/extraction/`
4. Write tests in `tests/ingestion/` and `tests/extraction/`

**"Add a new analysis feature":**
1. Create module in `src/analysis/` (e.g., `sentiment.py`)
2. Add prompt template to `src/analysis/prompts/`
3. Use `llm_client.py` for all Claude API calls
4. Update `src/models/schemas.py` for response structure
5. Write tests with mocked LLM responses

**"Fix a bug":**
1. Identify the layer (check `context/folder.md`)
2. Read layer-specific `CLAUDE.md` if it exists
3. Write a failing test first
4. Fix the bug
5. Verify test passes

**"Refactor":**
1. Understand current architecture from `context/architecture.md`
2. Maintain single source of truth for schemas (`src/models/`)
3. Preserve async DB patterns
4. Update tests to match new structure
5. Document significant changes

### What to Avoid

- Creating duplicate schema definitions outside `src/models/`
- Direct `os.environ` access instead of `src/config.py`
- Creating new Anthropic clients instead of using `llm_client.py`
- Blocking DB calls (always use async SQLAlchemy)
- Modifying `.claude/` hooks without explicit user request

---

## 📚 Project Structure Highlights

```
Document_Intelligence/
├── CLAUDE.md                    ← You are here
├── context/                     ← Read these first
│   ├── architecture.md
│   └── folder.md
├── src/                         ← All application code
│   ├── config.py                ← Config loader (use this!)
│   ├── ingestion/               ← Document intake
│   ├── extraction/              ← Content extraction
│   ├── analysis/                ← LLM intelligence
│   │   └── llm_client.py        ← Single Claude client
│   ├── api/                     ← FastAPI app
│   └── models/                  ← Single source of truth
├── tests/                       ← pytest suite
└── .claude/                     ← Agent config & skills
```

Full structure in `context/folder.md`.

---

## 🎓 Learning the Codebase

**Quickest path:**
1. Read `context/architecture.md` (10 min)
2. Skim `context/folder.md` (5 min)
3. Read `src/api/main.py` (entry point)
4. Trace one request: `POST /upload` → `src/api/routers/documents.py` → `src/ingestion/tasks.py`

**Want to understand a specific layer?**
- Check if that layer has a `CLAUDE.md` in its folder
- Read the main module (e.g., `src/ingestion/router.py`)
- Look at the corresponding tests (e.g., `tests/ingestion/`)

---

## 🤝 Collaboration Principles

- **Context awareness:** Always reference `context/architecture.md` before proposing changes
- **Consistency:** Follow existing patterns in the layer you're modifying
- **Testing:** Write tests for any new functionality
- **Documentation:** Update context docs when introducing new patterns
- **Clarity:** Ask questions if the architecture is unclear

---

## 📌 Key Decisions Reference

| Decision | Choice | Why |
|---|---|---|
| Async DB | SQLAlchemy 2.x async | Non-blocking API |
| Job queue | Celery + Redis | Reliable, retryable |
| LLM client | Single `llm_client.py` | Centralized retry/caching/tracking |
| Schema source | `src/models/` only | Prevents drift |
| Config | `src/config.py` | Type-safe, `.env` support |

Full table in `context/architecture.md`.

---

**Questions?** Check `context/architecture.md` for design details or `context/folder.md` for file locations.
