# Document Intelligence — Current Status

> What's actually built, as of 2026-07-21. Companion to `architecture.md` (the target
> blueprint), `folder.md` (the target folder layout), and `backend.md` (the phased build
> plan for everything past parsing) — this file is the reality check against all three.
> Everything in this doc lives under `src/backend/`.

## One-line summary

Two things are built and verified end to end: the **multimodal ingestion/parsing
stage** (six document formats + audio → normalized `Chunk`s, tables/images summarized
by a local vision LLM), and **backend.md Phases 1-3** — real Postgres-backed auth/RBAC,
sitting in front of `/parse`, on a live Neon database. Register → login → create
company/department → add member → upload through `/parse` has been driven through
Swagger against Neon and works, including the 403-on-no-membership path. **Still
nothing past that**: no embeddings, no vector store, no graph, no retrieval, no
generation, no async job queue.

## What's built

**API** — one FastAPI router, `/parse` (`api/parsing_router.py`):
- `POST /parse` — sync, whole-file-in / `ParsedDocument`-out, for pdf/docx/pptx/txt/md/html
- `POST /parse/audio/stream` — SSE, streams `Chunk`s as Whisper transcribes

**Parsing** (`parsing/`):
- `document_parser.py` — `unstructured` (not Docling) partitions pdf/docx/pptx/txt/md/html
  into `RawElement`s, reading order preserved, nothing dropped
- `audio_parser.py` — faster-whisper, streamed via a background thread
  (`utils/async_utils.iter_in_thread`) so transcription doesn't block the event loop;
  device/compute type now auto-detect CUDA vs CPU instead of hardcoding `cuda`

**Pipelines** (`pipelines/`):
- `text_pipeline.py` — filters boilerplate (Header/Footer/UncategorizedText), attaches
  figure captions to their nearest table/image, groups text under Titles, splits
  oversized chunks (2000 char / 200 overlap) via LangChain's
  `RecursiveCharacterTextSplitter`
- `table_pipeline.py` — table HTML → dense LLM summary (values named explicitly, no
  invented numbers)
- `image_pipeline.py` — image bytes → structured JSON (`type`, `description`,
  `text_in_image`) via Ollama vision, with a fallback wrap if the model returns
  malformed JSON

**LLM** (`services/llm_service.py`) — single `generate()` entrypoint over Ollama
(`qwen2.5vl:7b` by default), text-only or text+image, optional Pydantic-schema-
constrained decoding. This is the *parsing-time* LLM (table/image description) — there
is no separate answer-generation LLM call yet.

**Identity, RBAC & multi-tenancy — backend.md Phases 1-3, built and DB-verified:**
- `core/` — `config.py` (env vars incl. `DATABASE_URL`, loaded from a gitignored `.env`
  via `python-dotenv`), `database.py` (async SQLAlchemy engine/session against Neon,
  with `statement_cache_size=0` + `ssl=True` for Neon's pooled/PgBouncer endpoint),
  `security.py` (JWT issue/verify, bcrypt hashing)
- `models/` — all 7 tables from `backend.md` §3 (`Company`, `User`, `Department`,
  `DepartmentMembership`, `Document`, `IngestionJob`, `AuditLog`), no `relationship()`
  attributes yet (nothing needs ORM traversal, plain FKs are enough so far)
- `identity/` — `auth_service.py` (register/login), `rbac_service.py`
  (`require_dept_access` for path-param routes, `require_dept_access_form` for
  multipart upload routes — both share one `_check_access` helper), `membership_service.py`
- `api/` — `auth_router.py`, `company_router.py`, `department_router.py`,
  `membership_router.py`, all wired into `main.py`; `parsing_router.py`'s `/parse` and
  `/parse/audio/stream` now require a bearer token + editor-or-above department
  membership instead of trusting a client-supplied `dept_id`
- Every chunk is still tagged with `chunk_id`/`doc_id`/`dept_id` at
  `parsing_service._tag_chunk`, but `dept_id` on the way in is now the caller's *real*,
  membership-verified department, not an unchecked string
- `/parse` now writes a row to Postgres `documents` on success (`id` = the same
  `doc_"+hex` string already stamped on every chunk, not a disconnected UUID — so a
  Postgres document row and its chunks share one id); `/parse/audio/stream` writes
  `status=processing` up front and `ready`/`failed` when the stream ends, via its own
  session opened in the SSE generator (a request-scoped `Depends(get_db)` session isn't
  guaranteed to outlive the endpoint function returning on a `StreamingResponse`)
- Enforcement stops at Postgres — there's still no Qdrant/Neo4j for a `BoundClient` to
  guard, so `dept_id` isolation doesn't extend past this layer yet

### Content-type coverage

| Type | Status |
|---|---|
| PDF / DOCX / PPTX | ✅ via `unstructured`, hi-res strategy, table structure inferred |
| TXT / MD / HTML | ✅ via `unstructured` |
| Tables (any of the above) | ✅ LLM-summarized from `text_as_html` |
| Images (in PDF) | ✅ LLM-described (vision) |
| Audio (mp3/wav/m4a/flac/ogg/webm) | ✅ streamed transcription, real timestamps per chunk |
| URL / web pages | ❌ `web_parser.py` was removed (git status shows it deleted, not yet replaced) |
| Video | ❌ not started |
| CSV/XLS structured data | ❌ not started |

## What's not built (backend.md Phases 4-5, and everything past parsing)

PostgreSQL/RBAC is now real (see above). Still not built, per `architecture.md` and
`backend.md`: Qdrant (dense+sparse storage, `HybridResolver`/`BoundClient` isolation —
`department_router.py` hardcodes `isolation_mode=LOGICAL` as a placeholder, the real
`classify()` decision is Phase 4), Neo4j/LightRAG (entity graph), embeddings (BGE-M3,
SPLADE, CLIP), `indexing/` (embedding + routing per `vector-graph-routing.md`),
Celery/Redis (async job queue — parsing today runs in-request), reranking, MMR, hybrid
search, query classification/routing, citation tracking, chat/SSE generation endpoint,
LangFuse/Prometheus/Grafana observability, RAGAS/DeepEval.

`architecture.md`/`backend.md` are aspirational/target-state; this status file is the
only accurate record of what's actually running.

## Known divergences from the blueprint

- **Parser**: `unstructured`, not Docling — works for the same format set, no
  TableFormer-specific tuning
- **Vision/LLM**: local Ollama (`qwen2.5vl:7b`), not GPT-4o/LLaVA
- **No async job queue**: ingestion is a direct FastAPI call, not Celery-queued —
  fine at current scale, revisit if a `/parse` call needs to outlive one request
- **Web parsing removed**: was present, now deleted from the tree with nothing to
  replace it yet
- **Database is Neon**, not self-hosted Postgres — `backend.md` assumed a local/Docker
  instance; connecting to Neon's pooled endpoint needed `ssl=True` +
  `statement_cache_size=0` in `connect_args` that a non-pooled setup wouldn't
- **No Alembic** — scaffolded once, then deleted at the user's request. Schema exists
  on Neon via a one-shot `Base.metadata.create_all()`, not versioned migrations. Any
  future model change (e.g. the `Document.id` type fix already needed once) means
  hand-dropping/recreating the affected table — fine solo, revisit before this has
  real data worth preserving across a schema change
- **`company_router.py`/`department_router.py`/`membership_router.py` are
  unauthenticated** — provisioning a company, department, or the *first* member of a
  department can't go through `require_dept_access` (there's no membership yet to
  check). No superuser/bootstrap concept exists to close this; fine for solo dev,
  a real gap before this is multi-user
