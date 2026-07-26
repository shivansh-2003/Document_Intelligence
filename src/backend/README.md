# Document Intelligence — Backend

Multi-tenant document ingestion for RAG. Parses PDF / DOCX / PPTX / TXT / MD / HTML and audio (mp3/wav/m4a/flac/ogg/webm) into embedding-ready chunks — text, tables, and images normalized into one ordered stream — behind JWT auth with department-level RBAC backed by Postgres (Neon), then embeds and indexes every chunk into Qdrant (dense + sparse, hybrid search, `dept_id`-filtered) automatically.

Built on [unstructured](https://docs.unstructured.io) for partitioning, [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for transcription, [Ollama](https://ollama.com) (`qwen2.5vl:7b`) for table/image enrichment, and [fastembed](https://github.com/qdrant/fastembed) + [Qdrant](https://qdrant.tech) for embedding/retrieval.

## Components


| Layer            | Modules                                                                                                                                    | Role                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| **API**          | `api/auth_router.py` · `company_router.py` · `department_router.py` · `membership_router.py` · `parsing_router.py` · `documents_router.py` | REST surface: auth, tenant provisioning, membership, parsing (single/stream/batch), document status |
| **Identity**     | `identity/auth_service.py` · `rbac_service.py` · `membership_service.py`                                                                   | register/login, JWT verification, `require_dept_access` RBAC gate                                   |
| **Core**         | `core/config.py` · `database.py` · `security.py`                                                                                           | env config (`.env`), async SQLAlchemy engine (Neon), JWT + bcrypt primitives                        |
| **Models**       | `models/` — Company, User, Department, DepartmentMembership, Document, IngestionJob, AuditLog                                              | the 7 Postgres tables                                                                               |
| **Parsing**      | `parsing/document_parser.py` · `audio_parser.py`                                                                                           | file → `RawElement`s / streamed Whisper transcription                                               |
| **Pipelines**    | `pipelines/text_pipeline.py` · `table_pipeline.py` · `image_pipeline.py`                                                                   | normalization + LLM enrichment → `Chunk`s                                                           |
| **Services**     | `services/parsing_service.py` · `llm_service.py`                                                                                           | orchestration, identity tagging, single Ollama entrypoint                                           |
| **Indexing**     | `indexing/embedding_service.py` · `indexing_pipeline.py`                                                                                   | chunk → dense + sparse vectors, thin orchestrator                                                   |
| **Vector store** | `vectorstore/vector_store.py`                                                                                                              | Qdrant collection mgmt, hybrid upsert/search, `dept_id` pre-filter                                  |


## Architecture

```mermaid
flowchart TD
    subgraph AUTH["Auth & tenancy (Postgres / Neon)"]
        AR["/auth/register · /auth/login<br/>api/auth_router.py"] --> JWTs[JWT: user_id only]
        CR["/companies · /departments · /departments/{id}/members<br/>provisioning routers"]
        RBAC["identity/rbac_service.py<br/>require_dept_access(min_role)<br/>fresh membership lookup per request"]
    end

    U[Upload + bearer token] --> R["api/parsing_router.py<br/>POST /parse · /parse/audio/stream · /parse/batch"]
    JWTs -.-> RBAC
    RBAC -->|403 if no membership / role too low| R

    R -->|single doc, sync| S["services/parsing_service.py<br/>parse_document()"]
    R -->|single audio, SSE| AS["parse_audio_stream()<br/>chunks stream as Whisper transcribes"]
    R -->|batch: N files, mixed formats| BQ["202 immediately +<br/>asyncio.create_task per file<br/>doc files → thread pool, audio → same async path"]
    BQ -.->|poll| DR["api/documents_router.py<br/>GET /departments/{id}/documents/{doc_id}"]

    S --> P["parsing/document_parser.py<br/>pdf/docx/pptx/txt/md/html → RawElement"]
    AS --> W["parsing/audio_parser.py<br/>faster-whisper, segments → Chunks<br/>real start_sec/end_sec"]
    BQ --> P
    BQ --> W

    P --> TB["pipelines/table_pipeline.py<br/>Table.text ← LLM summary"]
    P --> IM["pipelines/image_pipeline.py<br/>Image.text ← LLM JSON (schema-constrained)"]
    TB & IM --> LLM["services/llm_service.py<br/>Ollama · qwen2.5vl:7b"]

    LLM --> NORM["pipelines/text_pipeline.py<br/>filter → captions → group_by_title → split_oversized"]
    NORM --> TAG["_tag_chunk: chunk_id · doc_id · dept_id<br/>dept_id = membership-verified, not client-supplied"]
    W --> TAG
    TAG --> IDX["indexing/indexing_pipeline.py<br/>index_chunks() -- NOT called from parsing/pipelines"]
    IDX --> EMB["embedding_service.py<br/>dense (bge-large-en-v1.5) + sparse (SPLADE++)"]
    EMB --> VS["vectorstore/vector_store.py<br/>Qdrant 'chunks' collection<br/>point id = uuid5(chunk_id), dept_id indexed"]
    IDX --> OUT["ParsedDocument / SSE events / documents row<br/>documents.id = doc_id on every chunk"]
```



## Core Ideology

One document → **one ordered stream**, not three separate buckets.

The moment you split elements into `tables = [...]`, `images = [...]`, `texts = [...]`, a table loses its heading, its lead-in sentence, and its page neighbours. Retrieval then returns a naked HTML grid with no idea what it's a table *of*. Reading order (`idx`) is the context — it is preserved end to end.

```
PDF page 3
 ├─ Title          "Q3 Revenue"           ← context for the table
 ├─ NarrativeText  "Revenue grew 12%..."  ← context for the table
 ├─ Table          <html>…</html>         ← stays downstream of the two above
 ├─ NarrativeText  "The decline in APAC…"
 └─ Image          [b64] + caption        ← belongs where it sits
```

Audio deliberately bypasses this (no Titles to group by): Whisper segments pack into ~1500-char chunks carrying their own real `start_sec`/`end_sec`, streamed to the caller as they transcribe rather than after the whole file finishes.

## Auth & Multi-Tenancy

- **Tokens carry `user_id` only** — never `dept_id`/role. Role is looked up fresh from `department_memberships` per request, so a role change takes effect without re-login.
- `**require_dept_access(min_role)`** — the single RBAC choke point. Path-param variant for `/dept/{dept_id}/...` routes, `require_dept_access_form` for multipart upload routes. No membership row or insufficient role → `403`.
- **Role ranks**: `admin > editor > viewer`, compared with `>=` — an admin passes an editor check. Both `/parse` endpoints require **editor**.
- `**dept_id` on chunks is real** — taken from the caller's verified `DepartmentMembership`, not trusted from the request.
- Client-supplied IDs are typed `uuid.UUID` in Pydantic/path params — malformed input fails at the boundary with `422`, never reaches handler code.

## Pipeline Stages

### 1. Partition (`parsing/document_parser.py`)

`DocumentPartitioner` — one class, dispatched by file suffix. Raw partition only: nothing collapsed, nothing dropped, order preserved.


| Format              | Function                            | Notes                                                           |
| ------------------- | ----------------------------------- | --------------------------------------------------------------- |
| `.pdf`              | `partition_pdf`                     | `strategy="hi_res"`, table structure + image payload extraction |
| `.docx` / `.pptx`   | `partition_docx` / `partition_pptx` | `infer_table_structure=True`                                    |
| `.txt`              | `partition_text`                    | explicit utf-8 (chardet misfires on short files)                |
| `.md` / `.markdown` | `partition_md`                      | YAML frontmatter stripped; pipe tables → Table + `text_as_html` |
| `.html` / `.htm`    | `partition_html`                    |                                                                 |


Paginationless sources (txt/md/html) emit `page_number=None` throughout; downstream handles it.

### 2. Table Enrichment (`pipelines/table_pipeline.py`)

For every `Table`: `metadata.text_as_html` → LLM → dense prose summary written into `.text`. The HTML stays untouched as source of truth. The prompt forces: subject, structure with per-column units, **named standout values with labels attached**, multi-level header resolution, a no-fabrication rule, and an 8-sentence cap (table chunks skip `split_oversized`, so an unbounded summary would become one giant unsplit chunk).

### 3. Image Enrichment (`pipelines/image_pipeline.py`)

For every `Image`/`Figure` with an `image_base64` payload: vision LLM → JSON written into `.text`, via Ollama **structured outputs** (decoding constrained to the Pydantic schema):

```python
class ImageDescription(BaseModel):
    type: str            # priority-ordered decision list in the prompt
    description: str     # leads with the type noun for keyword matchability
    text_in_image: str   # verbatim transcription of visible text
```

A `ValidationError` falls back to `{"type": "unknown", "description": raw}` instead of crashing the batch.

### 4. Text Normalization (`pipelines/text_pipeline.py`)

Four passes: `filter_elements` (drop Header/Footer/UncategorizedText) → `attach_captions` (FigureCaption prepended to nearest Table/Image by `idx`) → `group_by_title` (Title anchors a section; NarrativeText/Formula/ListItem join until the next Title; Table/Image pass through standalone) → `split_oversized` (2000 chars / 200 overlap via `RecursiveCharacterTextSplitter` — the safety net for Title-less txt/md/html; tables and images are never split).

### 5. Audio (`parsing/audio_parser.py`)

faster-whisper (`large-v3` on CUDA, auto-falls back to `small`-class CPU settings via `ctranslate2.get_cuda_device_count()`), bridged from its blocking generator to async via `utils/async_utils.iter_in_thread` with bounded-queue backpressure. Segments pack to ~1500 chars; every chunk carries real `start_sec`/`end_sec` — audio never goes through `split_oversized`, which would clone one timestamp pair across pieces.

### 6. Identity Tagging & Persistence

`parsing_service._tag_chunk` stamps every chunk with `chunk_id` (`chnk_` + hex), `doc_id` (`doc_` + hex), `dept_id`, `doc_type` — after `split_oversized`, so sub-chunks never share a `chunk_id`. `/parse` writes a `documents` row on success; `/parse/audio/stream` writes `status=processing` up front and flips to `ready`/`failed` when the stream ends. `documents.id` **is** the same `doc_`-prefixed string on every chunk — one ID across Postgres and chunk metadata.

### 7. Embedding & Indexing (`indexing/`, `vectorstore/`)

Deliberately a separate, later step — `parsing/` and `pipelines/` never call embedding or vector-store code directly. All three ingestion routes call `index_chunks()` after parsing and *before* marking a document `ready`: if indexing fails, the document is marked `failed` (or the sync `/parse` request itself fails) rather than reporting success on a document that isn't actually searchable.

- **What gets embedded** — never a raw structural payload. Text chunks embed as-is; table chunks embed the LLM summary (not `text_as_html`); image chunks embed just the `description` field pulled out of the JSON in `chunk.text` (not the raw `{"type": ...}` string — braces and key names are noise in vector space).
- **Dense model is `BAAI/bge-large-en-v1.5`, not BGE-M3** — the original plan called for BGE-M3, but it isn't shipped as an ONNX model in `fastembed` (checked directly against `TextEmbedding.list_supported_models()`, confirmed still missing after upgrading to the latest version). Same 1024-dim/cosine, so no schema impact; the tradeoff is English-only and a ~512-token window instead of BGE-M3's multilingual 8192-token context. Escape hatch if that measurably hurts retrieval: `FlagEmbedding.BGEM3FlagModel(...).encode(texts, return_dense=True, return_sparse=True)` for true BGE-M3 dense+sparse in one pass — heavier (real torch model, ~2.3GB+), so don't reach for it until SPLADE++ hybrid is actually insufficient.
- **Sparse model is `prithivida/Splade_PP_en_v1`** (SPLADE++) — BGE-M3's native sparse head isn't exposed in the Python `fastembed` package either; this is the supported pairing.
- **One Qdrant collection (`chunks`)**, not split by content type — `kind` is a payload field. Splitting by type would refragment exactly what the parsing stage deliberately keeps as one stream.
- **Point ID is `uuid.uuid5(NAMESPACE_URL, chunk_id)`, not `chunk_id` directly** — Qdrant only accepts unsigned integers or UUIDs as point IDs; an arbitrary prefixed string is rejected outright. The UUID5 derivation is deterministic, so re-processing a document still overwrites the same points instead of duplicating them (the idempotency the original `chunk_id`-as-ID plan wanted) — `chunk_id` itself still lives in the payload for cross-system correlation.
- `**dept_id` is an indexed payload field and a mandatory pre-filter** — `create_payload_index(..., "dept_id", KEYWORD)` at collection creation, and `hybrid_search()` raises if `dept_id` is empty/`None` rather than silently building `Filter(must=[])` (a filter that filters nothing — the difference between a loud failure and a cross-tenant data leak).
- **Fusion is Qdrant-native RRF** via `Prefetch` (dense + sparse legs) + `FusionQuery` — one round trip, no app-level merge step.
- Heavy fields (`image_base64`, `text_as_html`, full `element_ids`) stay in Postgres, not mirrored into the payload — they bloat every point for data rarely needed at search time.

```python
from vectorstore.vector_store import hybrid_search

points = await hybrid_search("Q3 revenue by region", dept_id=membership.dept_id, limit=10)
# each point.payload carries: dept_id, doc_id, doc_type, chunk_id, kind, title,
# filename, page_number, text -- enough to cite a source with zero extra DB round trip
```

No `/search` route calls `hybrid_search()` yet — that's the next piece, not built here.

## Batch / Multi-File Ingestion (`POST /parse/batch`)

Upload any mix of doc and audio formats in one request; each is processed independently. The route validates every file up front (fails fast, naming the offending file, before writing anything), writes a `documents` row per file (`status=pending`), and returns `202` immediately with `{"jobs": [{"doc_id", "filename"}, ...]}` — poll each one via `GET /departments/{dept_id}/documents/{doc_id}`.

Files are handed to `**asyncio.create_task()**`, not FastAPI's `BackgroundTasks` — that distinction is deliberate: `BackgroundTasks` awaits its queued tasks one at a time, so N files would still run serially in the background. Plain tasks actually overlap: doc-type files run their blocking `parse_document()` call in the default thread pool (`run_in_executor`, same reasoning as `iter_in_thread` for audio), audio files reuse `parse_audio_stream()` directly. A module-level set holds a strong reference to each task until it finishes (`task.add_done_callback(set.discard)`) — an unreferenced `asyncio.Task` can otherwise be garbage-collected mid-run.

**Ceiling, by design:** in-process, no broker. An in-flight batch is lost if the server restarts, and there's no retry. That's the gap Phase 5 (Celery + Redis, separate `embed`/`graph_extract` queue lanes per `context/backend.md`) exists to close — fine at current scale, revisit before this needs to survive a restart or scale across machines.

## Output Schema

```python
class Chunk(BaseModel):
    kind: str                  # "text" | "table" | "image"
    idx: int
    title: str | None
    text: str                  # prose / LLM table summary / LLM image JSON
    metadata: ChunkMetadata

class ChunkMetadata(BaseModel):
    chunk_id: str | None       # identity — stamped by parsing_service
    doc_id: str | None
    dept_id: str | None
    doc_type: str | None
    filename: str | None       # provenance
    page_number: int | None
    pages: list[int]
    element_ids: list[str]     # audit trail back to raw partition
    text_as_html: str | None   # table only
    image_base64: str | None   # image only
    image_mime_type: str | None
    start_sec: float | None    # audio only
    end_sec: float | None      # audio only
```

## API


| Method & path                                   | Auth            | Purpose                                                                                                                |
| ----------------------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `POST /auth/register`                           | —               | create user (needs a `company_id`)                                                                                     |
| `POST /auth/login`                              | —               | → `{access_token}`                                                                                                     |
| `POST /companies`                               | — *             | provision a company                                                                                                    |
| `POST /departments`                             | — *             | provision a department (isolation_mode hardcoded `logical` until Phase 4)                                              |
| `POST /departments/{dept_id}/members`           | — *             | add member with role                                                                                                   |
| `POST /parse`                                   | bearer + editor | file + `dept_id` form field → `ParsedDocument`, ≤ 50 MB                                                                |
| `POST /parse/audio/stream`                      | bearer + editor | audio + `dept_id` form field → SSE `chunk`/`done`/`error` events, ≤ 500 MB                                             |
| `POST /parse/batch`                             | bearer + editor | `files: list[...]` (mixed formats) + `dept_id` form field → `202` + job list, processed concurrently in the background |
| `GET /departments/{dept_id}/documents/{doc_id}` | bearer + viewer | poll a document's `status`/`chunk_count` (pending → processing → ready/failed)                                         |


 provisioning routes are deliberately unauthenticated for now — the first member of a new department can't pass an RBAC check that requires a membership. Close before multi-user.

```bash
uvicorn main:app --reload
# Swagger UI: http://localhost:8000/docs — Authorize with the login token, then:
# POST /companies → POST /auth/register → /auth/login → POST /departments
# → POST /departments/{id}/members → POST /parse (or /parse/batch, then poll
# GET /departments/{id}/documents/{doc_id})
```

## Setup

```bash
pip install -r requirements.txt

# macOS
brew install poppler tesseract
# Linux / Docker
apt-get install -y poppler-utils tesseract-ocr libgl1

ollama pull qwen2.5vl:7b
```

Create `backend/.env` (gitignored):

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@host/db   # Neon: no sslmode param — handled in connect_args
JWT_SECRET_KEY=<32+ byte secret>                       # dev default exists, never ship it
QDRANT_URL=https://<cluster-id>.<region>.aws.cloud.qdrant.io   # or http://localhost:6333 for local Docker
QDRANT_API_KEY=<qdrant cloud api key>                  # unset/blank for a local instance with no auth
```

Neon's pooled endpoint (PgBouncer, transaction mode) is handled in `core/database.py`: `ssl=True` + `statement_cache_size=0` in `connect_args`. Tables are created via a one-shot `Base.metadata.create_all()` — no Alembic (deliberate; revisit when there's data worth preserving across schema changes).

First embed call downloads ~1.2 GB (`bge-large-en-v1.5`) + SPLADE++ weights, cached after — worth warming once (`python3 -c "from indexing.embedding_service import embed_batch; embed_batch(['warm'])"`) rather than letting the first real upload eat that latency.

## Project Layout

```
backend/
├── main.py                          # FastAPI app, all routers wired
├── requirements.txt
├── api/
│   ├── auth_router.py               # /auth/register, /auth/login
│   ├── company_router.py            # /companies
│   ├── department_router.py         # /departments
│   ├── documents_router.py          # /departments/{id}/documents/{doc_id} status
│   ├── membership_router.py         # /departments/{id}/members
│   └── parsing_router.py            # /parse, /parse/audio/stream, /parse/batch
├── core/
│   ├── config.py                    # OLLAMA_*, WHISPER_*, DATABASE_URL, JWT_* (.env via dotenv)
│   ├── database.py                  # async engine + sessions (Neon-aware connect_args)
│   └── security.py                  # JWT encode/decode, bcrypt
├── identity/
│   ├── auth_service.py              # register, login
│   ├── rbac_service.py              # role_rank, require_dept_access(_form)
│   └── membership_service.py        # add/remove/change-role/list
├── indexing/
│   ├── embedding_service.py         # chunk_embed_text, embed_batch (dense + sparse)
│   └── indexing_pipeline.py         # index_chunks() orchestrator
├── vectorstore/
│   └── vector_store.py              # ensure_collection, upsert_chunks, hybrid_search
├── models/                          # Company, User, Department, DepartmentMembership,
│   └── ...                          # Document, IngestionJob, AuditLog
├── parsing/
│   ├── document_parser.py           # DocumentPartitioner (6 formats)
│   └── audio_parser.py              # faster-whisper streaming
├── pipelines/
│   ├── text_pipeline.py             # filter → captions → group → split → Chunk
│   ├── table_pipeline.py            # Table.text ← LLM summary
│   └── image_pipeline.py            # Image.text ← LLM JSON
├── services/
│   ├── llm_service.py               # generate(prompt, image_b64, schema) → Ollama
│   └── parsing_service.py           # orchestrators + _tag_chunk
└── utils/
    ├── async_utils.py               # iter_in_thread (blocking gen → async iter)
    └── parsing_utils.py             # RawElement, to_raw(), insight()
```

## Known Limitations / Open Items

- `**/parse` is still sync + CPU-bound** — a `hi_res` PDF blocks the request; each table/image is a blocking LLM round-trip. `/parse/batch` gets you `202 + job_id` and concurrent processing without a task queue (see Batch section above), but it's in-process — no broker, no retry, lost on restart. Production wants Phase 5 / Celery per `context/backend.md`.
- **Caption attachment runs after LLM enrichment** — `describe_tables()`/`describe_images()` run before `attach_captions`, so a caption gets prepended onto the LLM output; for images that breaks JSON-parseability of `Chunk.text` on captioned images.
- **Image chunk `.text` is still raw JSON in the `documents`/parsing layer** — `chunk_embed_text()` pulls just the `description` out before embedding, so the vector itself is clean, but the stored `Chunk.text` (and the payload's `"text"` field) is still the full `{"type":...,"description":...}` string.
- **Dense model is `bge-large-en-v1.5`, not BGE-M3** (see Embedding & Indexing section) — English-only, ~512-token window.
- **No `HybridResolver`/`BoundClient`** — `department_router.py` still hardcodes `isolation_mode=LOGICAL`; Qdrant tenant isolation today is the `dept_id` payload filter alone, not a structurally-enforced bound client the way Postgres RBAC is.
- **No `/search` or `/chat` route** — `hybrid_search()` exists and works, nothing in `api/` calls it yet.
- **Provisioning routes unauthenticated** (see API table) — no superuser/bootstrap concept yet.
- **No Alembic** — schema changes mean hand-dropping/recreating tables.
- Paragraphs split across page breaks are not stitched; consecutive Titles produce a title-only chunk; base64 images ride in the JSON response (fat payloads).
- No graph (Neo4j/LightRAG), no reranking/MMR, no Celery — that's Phase 5 and beyond in `context/backend.md`, routing decided in `context/vector-graph-routing.md`.

