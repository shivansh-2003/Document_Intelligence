# Document Intelligence — Backend

Multi-tenant document ingestion for RAG. Parses PDF / DOCX / PPTX / TXT / MD / HTML and audio (mp3/wav/m4a/flac/ogg/webm) into embedding-ready chunks — text, tables, and images normalized into one ordered stream — behind JWT auth with department-level RBAC backed by Postgres (Neon), then embeds and indexes every chunk into Qdrant (dense + sparse, hybrid search, `dept_id`-filtered) automatically.

On top of that index, `POST /query` answers natural-language questions across every department a user belongs to: query understanding (decompose + resolve, with a skip-router for the common case), cross-department fanout with quota-guaranteed RRF merge, reranking, recency weighting, a confidence gate, and a single cited-answer generation call — backed by a 3-tier cache and per-department corpus versioning so a re-ingest invalidates stale answers automatically.

Built on [unstructured](https://docs.unstructured.io) for partitioning, [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for transcription, [Ollama](https://ollama.com) (`qwen2.5vl:7b`) for table/image enrichment and generation, [fastembed](https://github.com/qdrant/fastembed) + [Qdrant](https://qdrant.tech) for embedding/retrieval/reranking, and [Valkey](https://valkey.io) for caching.

## Components


| Layer            | Modules                                                                                                                                    | Role                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| **API**          | `api/auth_router.py` · `company_router.py` · `department_router.py` · `membership_router.py` · `parsing_router.py` · `documents_router.py` · `retrieval_router.py` | REST surface: auth, tenant provisioning, membership, parsing (single/stream/batch), document status + supersession, `POST /query` |
| **Identity**     | `identity/auth_service.py` · `rbac_service.py` · `membership_service.py`                                                                   | register/login, JWT verification, `require_dept_access` RBAC gate                                   |
| **Core**         | `core/config.py` · `database.py` · `security.py` · `valkey_client.py`                                                                      | env config (`.env`), async SQLAlchemy engine (Neon), JWT + bcrypt primitives, lazy Valkey singleton |
| **Models**       | `models/` — Company, User, Department (+`corpus_version`), DepartmentMembership, Document (+`is_current`/`superseded_by`), IngestionJob, AuditLog | the 7 Postgres tables                                                                    |
| **Parsing**      | `parsing/document_parser.py` · `audio_parser.py`                                                                                           | file → `RawElement`s / streamed Whisper transcription                                               |
| **Pipelines**    | `pipelines/text_pipeline.py` · `table_pipeline.py` · `image_pipeline.py`                                                                   | normalization + LLM enrichment → `Chunk`s                                                           |
| **Services**     | `services/parsing_service.py` · `llm_service.py` · `retrieval_service.py`                                                                  | orchestration, identity tagging, single Ollama entrypoint, query → cited-answer orchestrator        |
| **Indexing**     | `indexing/embedding_service.py` · `indexing_pipeline.py`                                                                                   | chunk → dense + sparse vectors (diverging per-kind for sparse, §7.3), thin orchestrator + corpus-version bump |
| **Vector store** | `vectorstore/vector_store.py`                                                                                                              | Qdrant collection mgmt, hybrid upsert/search, dense-only search, `dept_id` pre-filter, supersession payload flip |
| **Retrieval**    | `retrieval/` — `query_transform.py` · `scope_resolver.py` · `conversation_slots.py` · `fanout.py` · `merge.py` · `rerank.py` · `recency.py` · `confidence_gate.py` · `generation.py` | query understanding → cross-dept fanout → RRF merge → rerank → recency → confidence gate → cited generation |
| **Caching**      | `caching/` — `cache_keys.py` · `embedding_cache.py` · `retrieval_cache.py` · `response_cache.py`                                           | 3-tier Valkey cache, keys bake in `corpus_version` for free invalidation                             |
| **Versioning**   | `ingestion_versioning/` — `corpus_version.py` · `supersession.py`                                                                          | atomic per-dept corpus counter, near-duplicate detection + explicit confirm-supersession workflow    |


## Architecture

### Ingestion

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

### Retrieval — `POST /query`

```mermaid
flowchart TD
    Q["POST /query<br/>bearer + {query, conversation_id?, dept_hint?}"] --> AUTH2["identity.rbac_service.get_current_user<br/>bearer only -- no path dept_id, a query can span<br/>every department the user belongs to"]
    AUTH2 --> SVC["services/retrieval_service.py<br/>answer_query()"]

    SVC --> SLOTS["retrieval/conversation_slots.py<br/>load_context(conversation_id) — Valkey,<br/>dept NAMES only, never dept_id/role"]
    SLOTS --> QT["retrieval/query_transform.py<br/>skip-router (short/single-intent) OR<br/>1 structured llm_service.generate() call<br/>→ QueryPlan{resolved_query, sub_queries[]}"]
    QT --> SCOPE["retrieval/scope_resolver.py<br/>dept_hint ∩ live memberships (fresh DB lookup)<br/>403 on zero memberships, unmatched hint dropped not rejected"]

    SCOPE --> RC1{"tier-2 retrieval_cache hit?<br/>key bakes in corpus_version per dept"}
    RC1 -->|hit| SCORED
    RC1 -->|miss| FAN["retrieval/fanout.py<br/>asyncio.gather: sub_query × dept_id<br/>→ vector_store.hybrid_search()<br/>(tier-1 embedding_cache feeds vector=)"]
    FAN --> MERGE["retrieval/merge.py<br/>level 2 RRF across variants, dedupe by chunk_id,<br/>per-department quota (level 1 RRF is inside hybrid_search)"]
    MERGE --> RERANK["retrieval/rerank.py<br/>fastembed TextCrossEncoder<br/>(BAAI/bge-reranker-base), run_in_executor"]
    RERANK --> RECENCY["retrieval/recency.py<br/>score × exp(-ln2 · age_days / RECENCY_HALF_LIFE_DAYS)"]
    RECENCY --> SCORED[scored, deduped chunks]

    SCORED --> GATE{"retrieval/confidence_gate.py<br/>top score ≥ CONFIDENCE_THRESHOLD?"}
    GATE -->|no| NOANS["not-confident response + raw citations<br/>-- no cache write, no slot update"]
    GATE -->|yes| WRITE2["retrieval_cache.set_cached()"]
    WRITE2 --> RC2{"tier-3 response_cache hit?<br/>cosine match vs stored query embeddings, same scope"}
    RC2 -->|hit| ANS[answer + citations]
    RC2 -->|miss| GEN["retrieval/generation.py<br/>llm_service.generate() — 1 cited-answer call"]
    GEN --> WRITE3["response_cache.set_cached() + conversation_slots.save_slots()"]
    WRITE3 --> ANS
```

The flowchart above shows the branching logic; this is the same request as an actual call sequence — who calls whom, and in what order:

```mermaid
sequenceDiagram
    participant C as Client
    participant R as retrieval_router.py
    participant Svc as retrieval_service.answer_query()
    participant Slots as conversation_slots.py
    participant QT as query_transform.py
    participant Scope as scope_resolver.py
    participant RC as retrieval_cache.py
    participant Fan as fanout.py
    participant VS as vector_store.hybrid_search()
    participant Merge as merge.py
    participant Rerank as rerank.py
    participant Recency as recency.py
    participant Gate as confidence_gate.py
    participant RespC as response_cache.py
    participant Gen as generation.py

    C->>R: POST /query {query, conversation_id, dept_hint}
    R->>Svc: answer_query(db, user, query, ...)
    Svc->>Slots: load_context(conversation_id)
    Slots-->>Svc: prior topic/period/dept names (or "")
    Svc->>QT: plan_query(query, context)
    QT-->>Svc: QueryPlan{resolved_query, sub_queries[]}
    Svc->>Scope: resolve_scope(db, user, dept_hint)
    Scope-->>Svc: dept_ids[] (403 raised here if empty)

    Svc->>RC: get_cached(resolved_query, dept_ids, versions)
    alt tier-2 hit
        RC-->>Svc: scored chunks
    else miss
        Svc->>Fan: fanout(sub_queries, dept_ids)
        Fan->>VS: hybrid_search() per (sub_query, dept_id)
        VS-->>Fan: ScoredPoints
        Fan-->>Svc: variant_results
        Svc->>Merge: merge(variant_results)
        Merge-->>Svc: deduped, quota'd points
        Svc->>Rerank: rerank(resolved_query, points)
        Rerank-->>Svc: cross-encoder scored
        Svc->>Recency: apply_recency(scored)
        Recency-->>Svc: age-decayed scores
    end

    Svc->>Gate: passes(scored)
    alt below CONFIDENCE_THRESHOLD
        Svc-->>R: not-confident + raw citations
    else above threshold
        Svc->>RC: set_cached(...)
        Svc->>RespC: get_cached(resolved_query, dept_ids, versions)
        alt tier-3 hit
            RespC-->>Svc: cached answer + citations
        else miss
            Svc->>Gen: generate_answer(resolved_query, scored)
            Gen-->>Svc: answer + citations
            Svc->>RespC: set_cached(...)
        end
        Svc->>Slots: save_slots(conversation_id, plan)
    end
    Svc-->>R: RetrievalResponse
    R-->>C: 200 {answer, citations, confident}
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

## Retrieval Pipeline (`retrieval/`, `caching/`, `ingestion_versioning/`)

Phase 6 per `context/backend.md`/`context/retrieval.md` — the first thing built on top of Phase 4's hybrid search. Full spec (numbered `§` sections referenced below) lives in `context/retrieval.md`. Nothing here needs Celery/workers: a query is synchronous request/response, it doesn't need a queue.

### 1. Query understanding (`retrieval/query_transform.py`, `scope_resolver.py`, `conversation_slots.py`)

`query_transform.plan_query()` produces a `QueryPlan{resolved_query, sub_queries[], doc_type_hint, period_hint}`. A **skip-router** (`SKIP_ROUTER_MAX_WORDS = 12`, no `and`/`vs`/`versus`) bypasses the LLM call entirely for short, single-intent queries with no prior conversation context — the plan is just the raw query, unresolved and undecomposed. Everything else goes through one structured call (`services.llm_service.generate(prompt, schema=QueryPlan)` — the same schema-constrained-decoding path `image_pipeline.py` already uses) that classifies, decomposes, and resolves coreference together, not three separate round trips.

`scope_resolver.resolve_scope()` intersects `dept_hint` (department **names**, an LLM-derived suggestion) against the caller's live `DepartmentMembership` rows, looked up fresh every call — same anti-staleness principle as `identity.rbac_service._check_access`, nothing cached. Any membership row already clears the viewer floor (`Role.VIEWER` is rank 0). Zero memberships → `403`. A hint that matches no live membership is **dropped, not rejected** — it's a suggestion, never an access grant, so it can only narrow within the real boundary, never expand past it.

`conversation_slots.py` persists only `period`/`topic`/`doc_type` + department **names** (never `dept_id`, never `role`) in Valkey, keyed by `conversation_id`, TTL'd (`CACHE_TTL_CONVERSATION`). Read back as a coreference hint on the next turn ("what about last quarter") — but `scope_resolver` re-validates any department name against live memberships on every single call regardless. The slot store is never itself a source of truth for access, and it's never written on the low-confidence path (§4 below) — an unreliable plan shouldn't seed the next turn.

### 2. Fanout & merge (`retrieval/fanout.py`, `merge.py`)

`fanout()` runs `asyncio.gather` over every `(sub_query, dept_id)` pair, each a direct call into the existing `vectorstore.vector_store.hybrid_search()` — no new search logic, this is orchestration only. Each *unique* sub-query text is embedded once (tier-1 cached, see §5) and reused across every department it fans out to, rather than re-embedding per pair.

`merge()` is **two-level RRF**: level 1 (dense+sparse fusion) already happens inside `hybrid_search()` via Qdrant's own `FusionQuery`. Level 2 fuses across sub-query/department variants (`score = Σ 1/(60 + rank)`), dedupes by `chunk_id`, then applies a **per-department quota** — every department with at least one match keeps a slot in the merged output, so one department's score distribution can't fully starve another's genuinely relevant chunk in a cross-department query.

```mermaid
flowchart TB
    subgraph L1["Level 1 — inside hybrid_search(), once per (sub_query, dept_id) variant"]
        direction LR
        D["dense prefetch<br/>limit × 4"] --> FQ["Qdrant FusionQuery(RRF)"]
        Sp["sparse prefetch<br/>limit × 4"] --> FQ
        FQ --> V["ranked ScoredPoints<br/>for this one variant"]
    end

    V1["variant: (sub_query A, dept_id X)"] --> L2
    V2["variant: (sub_query A, dept_id Y)"] --> L2
    V3["variant: (sub_query B, dept_id X)"] --> L2

    subgraph L2["Level 2 — merge.py, across every variant"]
        direction TB
        Sum["Σ 1/(60 + rank) per chunk_id,<br/>summed across every variant it appears in"]
        Sum --> Dedupe["dedupe by chunk_id"]
        Dedupe --> Quota["per-department quota:<br/>each dept with ≥1 match keeps ≥1 slot,<br/>even if outscored by another dept"]
        Quota --> Out["merged, quota'd top-K → rerank.py"]
    end
```

### 3. Rerank & recency (`retrieval/rerank.py`, `recency.py`)

Single rerank pass against `resolved_query` (not one pass per sub-query) using `fastembed`'s `TextCrossEncoder` (`BAAI/bge-reranker-base` — already a project dependency via `fastembed`, no new package). Same lazy-singleton + `run_in_executor` pattern as `embedding_service._get_dense()`/`_get_sparse()` and the CPU-bound-blocks-the-loop reasoning already documented for `vector_store.upsert_chunks`.

`recency.py` then decays each reranked score by chunk age: `adjusted = rerank_score × exp(-ln2 × age_days / RECENCY_HALF_LIFE_DAYS)`, reading the new `indexed_at` Qdrant payload field (ISO timestamp, stamped at upsert time — see §6 below).

### 4. Confidence gate & generation (`retrieval/confidence_gate.py`, `generation.py`)

`confidence_gate.passes()` checks the top adjusted score against `CONFIDENCE_THRESHOLD`. This single check gates **two** independent things downstream: whether `generation.py` runs at all, and whether either cache tier gets written — a low-confidence result is never cached and reused as if it were trustworthy. Below threshold, the caller still gets the raw citations back (near-matches are worth showing) with a fixed "not confident" message, no LLM call spent.

`generation.generate_answer()` reuses `services.llm_service.generate()` (no new LLM plumbing) with a prompt-template convention copied from `table_pipeline.py` — module-level template + `build_prompt()` — requiring inline `[chunk_id]` citations for every claim.

### 5. Caching (`caching/`) — 3 tiers on Valkey

No Redis/Celery infra existed in this repo before this phase (everything else is Neon or Qdrant Cloud, both managed) — this introduces **Valkey**, a Redis-protocol-compatible fork, via one lazy client (`core/valkey_client.py`, same module-level-singleton shape as `vector_store.py`'s `_client`). `redis-server` works as the local dev backend since Valkey speaks the same RESP protocol — no separate binary required.

- **Tier 1 `embedding_cache.py`** — get-or-compute around a query's dense+sparse vectors, feeding `hybrid_search()`'s new optional `vector=` param so a hit skips the ONNX embed call entirely. `hybrid_search(..., vector=None)` is fully backward compatible — omitted, behavior is unchanged from before this param existed.
- **Tier 2 `retrieval_cache.py`** — get-or-compute around the fanout→merge→rerank→recency result list. Serializes to a minimal `CachedPoint` (just `.payload`) rather than a real Qdrant `ScoredPoint`, since that's all downstream code reads.
- **Tier 3 `response_cache.py`** — the one genuinely new mechanism: **cosine match against stored query embeddings**, not an exact key match. Valkey has no vector index, but a scope's candidate set is small (a cache, not a corpus) — `SCAN` the scope's key prefix and brute-force cosine-compare in Python against `RESPONSE_CACHE_SIM_THRESHOLD`. Reusing Qdrant for this would be reaching for a search index to solve a cache problem.

`cache_keys.py` bakes each scoped department's `corpus_version` into every tier-2/3 key — **that alone is the entire cache-invalidation mechanism**. A re-ingest bumps the counter (§6) and every old cache entry for that department naturally misses; there's no separate cache-bust step anywhere in this codebase. Shown over time, not just asserted:

```mermaid
sequenceDiagram
    participant U as User
    participant Svc as retrieval_service
    participant PG as Postgres (departments.corpus_version)
    participant RC as retrieval_cache (Valkey)

    U->>Svc: POST /query "refund policy?"
    Svc->>PG: get_versions([dept_A])
    PG-->>Svc: {dept_A: 3}
    Svc->>RC: get_cached(key = "ret:dept_A:3:<hash>")
    RC-->>Svc: miss
    Note over Svc: fanout → merge → rerank → recency (real work happens)
    Svc->>RC: set_cached(key = "ret:dept_A:3:<hash>", result)
    Svc-->>U: fresh answer

    U->>Svc: POST /parse — new document into dept_A
    Svc->>PG: bump_corpus_version(dept_A)
    PG-->>Svc: corpus_version now 4

    U->>Svc: POST /query "refund policy?" — same question again
    Svc->>PG: get_versions([dept_A])
    PG-->>Svc: {dept_A: 4}
    Svc->>RC: get_cached(key = "ret:dept_A:4:<hash>")
    RC-->>Svc: miss — "...:3:..." ≠ "...:4:...", no explicit cache-bust call anywhere
    Note over Svc: recomputes, now sees the newly ingested document
    Svc-->>U: fresh answer, reflects the re-ingest
```

### 6. Ingestion versioning (`ingestion_versioning/`) + the `§7.3` sparse-encode fix

`corpus_version.py`'s `bump_corpus_version()` is Postgres's own atomic increment (`corpus_version = corpus_version + 1 ... RETURNING`), not hand-rolled locking — called from `indexing.indexing_pipeline.index_chunks()` right after every successful `upsert_chunks()`, the same choke point every ingestion route already flows through.

`supersession.py`'s `detect_duplicates()` runs automatically right after ingest — dense-only cosine search (`vector_store.dense_search()`, a new raw-dense query distinct from `hybrid_search()`, since RRF-fused rank positions aren't meaningfully thresholdable the way a real cosine score is) over a sample of the new document's chunks (`SAMPLE_SIZE=5`, best-effort not exhaustive) against the same department's existing chunks. **Detection is informational only** — surfaced as `possible_duplicates` on `/parse`'s response and the `/parse/audio/stream` `done` event (logged for `/parse/batch`, which has no synchronous response channel back to the caller). Nothing here flips anything automatically: silently hiding a document from retrieval is a hard-to-reverse action, so confirming supersession is a **separate, explicit, editor-gated route**:

```
POST /departments/{dept_id}/documents/{doc_id}/supersede/{old_doc_id}
```

which sets `Document.is_current=False`/`superseded_by` in Postgres *and* flips the `is_current` Qdrant payload field via `vector_store.mark_superseded()` (a `set_payload` call filtered by `doc_id`, not a full re-upsert).

```mermaid
stateDiagram-v2
    [*] --> pending: POST /parse(/batch) — Document row written
    pending --> processing: _mark_document(status="processing")
    processing --> ready: index_chunks() succeeds
    processing --> failed: parse or index error
    ready --> superseded: POST .../supersede/{old_doc_id}\n(editor-gated, explicit — never automatic)
    superseded --> [*]

    note right of ready
        detect_duplicates() runs automatically on every
        ingest here — informational only (possible_duplicates
        in the response). It never changes state by itself.
    end note
    note right of superseded
        Postgres: is_current=False, superseded_by=doc_id
        Qdrant: mark_superseded() flips the payload field,
        no re-upsert. Excluded from hybrid_search()/dense_search()
        via must_not(is_current==False) — never must(is_current==True),
        so chunks indexed before this field existed still show up.
    end note
```

**`§7.3`**: `chunk_embed_text()` (dense) and the new `chunk_sparse_text()` (sparse) now diverge for table chunks. SPLADE is a term-matching model — the LLM table summary that dense uses is worse for it than the actual cell values would be ("$4.2M", "Q3"). `chunk_sparse_text()` strips tags from `ChunkMetadata.text_as_html` and sparse-encodes that instead, while dense keeps embedding the summary. `embedding_service.embed_batch()` grew a second optional `sparse_texts` param to carry this (defaults to the dense list, so every other caller — `hybrid_search`'s query embed, `supersession`'s detection embed — is unaffected).

**A backward-compatibility gotcha worth knowing if you're touching this code:** `hybrid_search()`/`dense_search()` filter out superseded chunks with `must_not: is_current == False`, deliberately **not** `must: is_current == True`. Every chunk indexed before this phase has no `is_current` key in its payload at all — a bare `must` on `True` would have silently erased all of them from every future search instead of just the ones actually superseded. Same reasoning applies to the `is_current` payload index: it's only created inside `ensure_collection()`'s "collection doesn't exist yet" branch, so an existing `chunks` collection (this project already has one) needs it added once by hand — `await _client.create_payload_index("chunks", "is_current", models.PayloadSchemaType.BOOL)` — filtering still works without it, just as an unindexed (slower) scan.

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
| `GET /departments/{dept_id}/documents/{doc_id}` | bearer + viewer | poll a document's `status`/`chunk_count` (pending → processing → ready/failed) + `is_current`/`superseded_by`         |
| `POST /departments/{dept_id}/documents/{doc_id}/supersede/{old_doc_id}` | bearer + editor | confirm `doc_id` supersedes `old_doc_id` — flips Postgres + Qdrant, never automatic (§7.2)      |
| `POST /query`                                   | bearer          | `{query, conversation_id?, dept_hint?}` → cited answer. No path `dept_id` — spans every department the caller belongs to, `scope_resolver.py` narrows it |


 provisioning routes are deliberately unauthenticated for now — the first member of a new department can't pass an RBAC check that requires a membership. Close before multi-user.

```bash
uvicorn main:app --reload
# Swagger UI: http://localhost:8000/docs — Authorize with the login token, then:
# POST /companies → POST /auth/register → /auth/login → POST /departments
# → POST /departments/{id}/members → POST /parse (or /parse/batch, then poll
# GET /departments/{id}/documents/{doc_id}) → POST /query {"query": "..."}
```

## Setup

```bash
pip install -r requirements.txt

# macOS
brew install poppler tesseract
# Linux / Docker
apt-get install -y poppler-utils tesseract-ocr libgl1

ollama pull qwen2.5vl:7b

# Valkey (or redis-server — protocol-compatible, works identically for dev)
brew install redis && redis-server --daemonize yes   # macOS, ad hoc
# brew services start redis                            # macOS, persists across reboots
# apt-get install -y valkey-server                      # Linux / Docker
```

Create `backend/.env` (gitignored):

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@host/db   # Neon: no sslmode param — handled in connect_args
JWT_SECRET_KEY=<32+ byte secret>                       # dev default exists, never ship it
QDRANT_URL=https://<cluster-id>.<region>.aws.cloud.qdrant.io   # or http://localhost:6333 for local Docker
QDRANT_API_KEY=<qdrant cloud api key>                  # unset/blank for a local instance with no auth
VALKEY_URL=redis://localhost:6379/0                    # optional — this is already the default
```

Neon's pooled endpoint (PgBouncer, transaction mode) is handled in `core/database.py`: `ssl=True` + `statement_cache_size=0` in `connect_args`. Tables are created via a one-shot `Base.metadata.create_all()` — no Alembic (deliberate; revisit when there's data worth preserving across schema changes). The retrieval phase added three columns by hand the same way: `departments.corpus_version`, `documents.is_current`, `documents.superseded_by` — `Base.metadata.create_all()` only creates missing *tables*, not missing *columns* on an existing one, so an already-provisioned DB needs the matching `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` run once.

First embed call downloads ~1.2 GB (`bge-large-en-v1.5`) + SPLADE++ weights, cached after — worth warming once (`python3 -c "from indexing.embedding_service import embed_batch; embed_batch(['warm'])"`) rather than letting the first real upload eat that latency. The reranker (`BAAI/bge-reranker-base`, ~1 GB) downloads on first `/query` call the same way — warm it too if you'd rather not eat that latency on a real request: `python3 -c "from retrieval.rerank import _get_reranker; _get_reranker()"`.

## Project Layout

```
backend/
├── main.py                          # FastAPI app, all routers wired
├── requirements.txt
├── api/
│   ├── auth_router.py               # /auth/register, /auth/login
│   ├── company_router.py            # /companies
│   ├── department_router.py         # /departments
│   ├── documents_router.py          # /departments/{id}/documents/{doc_id} status + supersede
│   ├── membership_router.py         # /departments/{id}/members
│   ├── parsing_router.py            # /parse, /parse/audio/stream, /parse/batch
│   └── retrieval_router.py          # POST /query — bearer only, no path dept_id
├── core/
│   ├── config.py                    # OLLAMA_*, WHISPER_*, DATABASE_URL, JWT_*, RERANK_*, CACHE_*,
│   │                                 #   RECENCY_HALF_LIFE_DAYS, CONFIDENCE_THRESHOLD, VALKEY_URL
│   ├── database.py                  # async engine + sessions (Neon-aware connect_args)
│   ├── security.py                  # JWT encode/decode, bcrypt
│   └── valkey_client.py             # lazy Valkey singleton, backs caching/ + conversation_slots
├── identity/
│   ├── auth_service.py              # register, login
│   ├── rbac_service.py              # role_rank, require_dept_access(_form), get_current_user
│   └── membership_service.py        # add/remove/change-role/list
├── indexing/
│   ├── embedding_service.py         # chunk_embed_text/chunk_sparse_text, embed_batch (dense + sparse)
│   └── indexing_pipeline.py         # index_chunks() — upsert → corpus_version bump → duplicate detect
├── vectorstore/
│   └── vector_store.py              # ensure_collection, upsert_chunks, hybrid_search, dense_search,
│                                     #   mark_superseded
├── retrieval/
│   ├── query_transform.py           # QueryPlan/SubQuery, skip-router, 1 structured LLM call
│   ├── scope_resolver.py            # dept_hint ∩ live memberships
│   ├── conversation_slots.py        # Valkey-backed semantic slot store (names only, never dept_id/role)
│   ├── fanout.py                    # asyncio.gather over sub_query × dept_id → hybrid_search
│   ├── merge.py                     # two-level RRF, dedupe, per-dept quota
│   ├── rerank.py                    # fastembed TextCrossEncoder rerank
│   ├── recency.py                   # post-rerank score decay by indexed_at
│   ├── confidence_gate.py           # threshold gate — generation + cache writes
│   └── generation.py                # cited-answer LLM call
├── caching/
│   ├── cache_keys.py                # tier 1/2/3 key builders, bakes in corpus_version
│   ├── embedding_cache.py           # tier 1 — query embedding get-or-compute
│   ├── retrieval_cache.py           # tier 2 — fanout→merge→rerank→recency get-or-compute
│   └── response_cache.py            # tier 3 — cosine match against stored query embeddings
├── ingestion_versioning/
│   ├── corpus_version.py            # atomic per-dept_id counter, bump-on-ingest
│   └── supersession.py              # near-duplicate detection (informational only)
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
│   ├── parsing_service.py           # orchestrators + _tag_chunk
│   └── retrieval_service.py         # answer_query() — wires the full query → answer path
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
- **Provisioning routes unauthenticated** (see API table) — no superuser/bootstrap concept yet.
- **No Alembic** — schema changes mean hand-dropping/recreating tables or, for the retrieval-phase columns, hand-running `ALTER TABLE` on an existing DB (see Setup).
- Paragraphs split across page breaks are not stitched; consecutive Titles produce a title-only chunk; base64 images ride in the JSON response (fat payloads).
- No graph (Neo4j/LightRAG), no Celery/workers — that's Phase 5 and beyond in `context/backend.md`, routing decided in `context/vector-graph-routing.md`.
- **No RAGAS/DeepEval/LangFuse observability** on the retrieval path — confidence-gate pass rate, rerank quality, and cache hit rate aren't measured anywhere yet, only logged ad hoc.
- **Tier-3 `response_cache` is a Python-side `SCAN` + brute-force cosine compare** over a scope's key prefix, deliberately — fine while a department's cached-query set is small, would need a real ANN index (or a TTL tight enough to bound the scan) if a single scope accumulates thousands of distinct cached queries.
- **Supersession detection is sampled, not exhaustive** (`SAMPLE_SIZE=5` chunks per document) and only ever suggests — see the `§7.2` writeup above for why confirming it is a separate, explicit, editor-gated call rather than automatic.
- **Valkey is a new, unauthenticated local dependency** — `VALKEY_URL` defaults to `redis://localhost:6379/0` with no password; fine for solo dev, needs real auth/TLS before this cache holds anything sensitive across a network.
- Reranking and the confidence gate haven't been load-tested against a large corpus — `RERANK_TOP_K`/`FANOUT_TOP_K`/`CONFIDENCE_THRESHOLD` in `core/config.py` are reasonable starting defaults, not tuned numbers.

