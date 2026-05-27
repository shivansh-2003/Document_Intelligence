# Document Intelligence System — Architecture Design Reference

> Companion to `Document_Intelligence_System_Architecture.md`
> Contains: complete folder structure · all third-party configurations · component architecture diagrams

---

## Table of Contents

1. [Complete Folder Structure](#1-complete-folder-structure)
2. [Environment Variables & Third-Party Configurations](#2-environment-variables--third-party-configurations)
   - 2.1 Core Application `.env`
   - 2.2 PostgreSQL
   - 2.3 Qdrant
   - 2.4 Neo4j
   - 2.5 Redis & Celery
   - 2.6 Object Storage (S3 / MinIO)
   - 2.7 Embedding Models (BGE-M3, BGE-small, SPLADE)
   - 2.8 Docling
   - 2.9 Crawl4AI
   - 2.10 LightRAG
   - 2.11 RAG-Anything
   - 2.12 LangFuse
   - 2.13 Prometheus & Grafana
   - 2.14 LangChain (chunking / reranking)
   - 2.15 Docker Compose — Full Production Stack
3. [Architecture Diagrams](#3-architecture-diagrams)
   - 3.1 System overview
   - 3.2 Multi-tenant data isolation
   - 3.3 Qdrant isolation decision flow (HybridResolver)
   - 3.4 Ingestion pipeline — doc-type-specific queues
   - 3.5 Video RAG pipeline — late-fusion embedding
   - 3.6 Hybrid search pipeline
   - 3.7 Adaptive latency routing — three tiers
   - 3.8 Citation tracking flow
   - 3.9 SSE streaming protocol — event sequence
   - 3.10 Neo4j graph schema
   - 3.11 Observability stack
   - 3.12 Evaluation framework pipeline

---

## 1. Complete Folder Structure

```
document-intelligence/
│
├── backend/                              # FastAPI Python application
│   ├── main.py                           # App factory, lifespan, middleware
│   ├── Dockerfile
│   ├── Dockerfile.gpu                    # Extends base; adds CUDA + torch
│   ├── requirements.txt
│   ├── requirements-gpu.txt
│   │
│   ├── core/
│   │   ├── config.py                     # Pydantic Settings; reads .env
│   │   ├── security.py                   # JWT encode/decode, bcrypt hashing
│   │   ├── database.py                   # Async SQLAlchemy engine + session
│   │   └── logging.py                    # Structured JSON logging (structlog)
│   │
│   ├── models/                           # SQLAlchemy ORM models
│   │   ├── company.py
│   │   ├── user.py
│   │   ├── department.py                 # isolation_mode, dept_type, regulatory_flags
│   │   ├── document.py
│   │   ├── ingestion_job.py
│   │   ├── audit_log.py                  # Append-only; never UPDATE
│   │   ├── department_quota.py
│   │   ├── department_usage.py
│   │   └── eval_run.py                   # RAGAS / DeepEval scores per dept
│   │
│   ├── api/                              # FastAPI routers
│   │   ├── auth.py                       # POST /auth/login, /auth/register
│   │   ├── companies.py                  # POST/GET /companies/
│   │   ├── departments.py                # POST/GET/PATCH /depts/ (RBAC-gated)
│   │   ├── ingest.py                     # POST /dept/{id}/ingest — quota → queue
│   │   ├── search.py                     # POST /dept/{id}/search — hybrid search
│   │   ├── chat.py                       # POST /dept/{id}/chat/stream — SSE
│   │   ├── admin.py                      # GET /admin/eval, /admin/usage
│   │   └── deps.py                       # Shared FastAPI dependencies (RBAC, DB)
│   │
│   ├── services/
│   │   │
│   │   ├── vector_store/                 # Multi-tenant Qdrant isolation framework
│   │   │   ├── resolver.py               # HybridResolver, DeptSensitivity enum
│   │   │   ├── bound_client.py           # BoundClient (logical), SimpleBoundClient (physical)
│   │   │   ├── logical_isolation.py      # LogicalIsolation — company collections
│   │   │   ├── physical_isolation.py     # PhysicalIsolation — lifecycle, rebalance alert
│   │   │   └── migration.py              # migrate_logical↔physical (zero-downtime)
│   │   │
│   │   ├── parsing/
│   │   │   ├── docling_parser.py         # PDF, PPTX, DOCX — layout-aware + TableFormer
│   │   │   ├── crawl4ai_parser.py        # URL — JS rendering, pruning content filter
│   │   │   ├── audio_parser.py           # MP3/WAV/M4A — Whisper v3 → timed chunks
│   │   │   ├── video_parser.py           # MP4/MOV — scene detect + CLIP + LLaVA
│   │   │   ├── image_parser.py           # JPG/PNG — RAG-Anything caption + OCR
│   │   │   └── table_parser.py           # CSV/XLS — pandas → markdown table chunks
│   │   │
│   │   ├── chunking/
│   │   │   ├── strategies.py             # RecursiveChar, Semantic, MarkdownHeader, SentenceWindow
│   │   │   ├── table_chunker.py          # Dual-representation: full_table + row_group
│   │   │   └── router.py                 # select_chunker(doc_type) → strategy
│   │   │
│   │   ├── embedding/
│   │   │   ├── dense_embedder.py         # BGE-M3 1024-dim (standard / deep tier)
│   │   │   ├── small_embedder.py         # BGE-small-en 384-dim (fast tier)
│   │   │   ├── sparse_embedder.py        # SPLADE — sparse dict {token_id: weight}
│   │   │   ├── video_embedder.py         # CLIP ViT-L/14 → proj 1024-dim; mean pool
│   │   │   └── cache.py                  # EmbeddingCache — Redis SHA-256 keyed, 7d TTL
│   │   │
│   │   ├── metadata/
│   │   │   ├── extractor.py              # extract_metadata() — 3-step pipeline
│   │   │   ├── schema.py                 # ChunkMetadata dataclass
│   │   │   └── recency.py                # exp(-days/180) recency score
│   │   │
│   │   ├── indexing/
│   │   │   ├── qdrant_service.py         # upsert_batch, delete_by_doc_id, scroll
│   │   │   └── neo4j_service.py          # Neo4jService — label-sanitised Cypher
│   │   │
│   │   ├── retrieval/
│   │   │   ├── hybrid_retriever.py       # asyncio.gather(dense_leg, sparse_leg) → RRF
│   │   │   ├── graph_retriever.py        # LightRAG aquery(mode='hybrid')
│   │   │   ├── reranker.py               # CrossEncoder ms-marco-MiniLM-L-12-v2
│   │   │   └── mmr.py                    # MMR diversity filter (λ=0.5)
│   │   │
│   │   ├── citations/
│   │   │   ├── tracker.py                # CitationTracker — register(), build_context_block()
│   │   │   ├── renderer.py               # CitationCard → SSE citation event
│   │   │   └── deep_links.py             # PDF bbox, video timestamp, slide thumbnail
│   │   │
│   │   ├── routing/
│   │   │   ├── classifier.py             # QueryClassifier — FAST/STANDARD/DEEP
│   │   │   └── pipeline_executor.py      # route_and_execute() → async generator
│   │   │
│   │   ├── generation/
│   │   │   ├── query_decomposer.py       # LLM → 3–5 sub-questions JSON
│   │   │   ├── response_planner.py       # LLM → [{"type":"table"},{"type":"text"},...]
│   │   │   └── structured_streamer.py    # SSE block-by-block streaming
│   │   │
│   │   ├── quotas/
│   │   │   ├── checker.py                # check_ingestion_quota()
│   │   │   └── rate_limiter.py           # DeptRateLimiter — Redis sliding window
│   │   │
│   │   ├── evaluation/
│   │   │   ├── ragas_runner.py           # run_ragas_eval() — faithfulness, recall, precision
│   │   │   ├── deepeval_runner.py        # HallucinationMetric, FaithfulnessMetric
│   │   │   └── golden_dataset.py         # build_golden_dataset() — auto-generate Q&A
│   │   │
│   │   └── observability/
│   │       ├── langfuse_tracer.py        # @observe decorators, score logging
│   │       ├── metrics.py                # Prometheus Histogram/Counter/Gauge definitions
│   │       └── audit.py                  # AuditLogService — write() middleware helper
│   │
│   ├── workers/
│   │   ├── ingestion_worker.py           # Celery app + all task definitions
│   │   │                                 #   parse_document_task  → queue=pdf
│   │   │                                 #   parse_video_task     → queue=video
│   │   │                                 #   parse_audio_task     → queue=video
│   │   │                                 #   parse_url_task       → queue=web
│   │   │                                 #   embed_and_index_task → queue=embed
│   │   └── beat.py                       # Celery Beat — scheduled eval runs
│   │
│   ├── alembic/                          # Database migrations
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/
│   │       ├── 001_initial_schema.py
│   │       ├── 002_add_isolation_mode.py
│   │       └── 003_add_quotas.py
│   │
│   └── tests/
│       ├── conftest.py                   # Fixtures: test DB, mock Qdrant, mock Neo4j
│       ├── test_auth.py
│       ├── test_ingest.py
│       ├── test_hybrid_search.py
│       ├── test_isolation.py             # BoundClient filter enforcement
│       ├── test_citations.py
│       └── test_eval_gates.py            # RAGAS thresholds as pytest
│
├── infra/
│   ├── docker-compose.yml                # Full production stack (see §2.15)
│   ├── docker-compose.dev.yml            # Local dev overrides (Localstack, mock LLM)
│   ├── .env.example                      # All env vars documented with defaults
│   │
│   ├── qdrant/
│   │   └── config.yaml                   # Qdrant server config (see §2.3)
│   │
│   ├── neo4j/
│   │   ├── neo4j.conf                    # Neo4j server config (see §2.4)
│   │   └── plugins/                      # APOC jar (Enterprise)
│   │
│   ├── prometheus/
│   │   ├── prometheus.yml                # Scrape config (see §2.13)
│   │   └── alerts.yml                    # Latency + error rate alert rules
│   │
│   ├── grafana/
│   │   └── dashboards/
│   │       ├── rag_health.json
│   │       ├── ingestion_pipeline.json
│   │       ├── qdrant.json
│   │       └── neo4j.json
│   │
│   └── langfuse/
│       └── docker-compose.langfuse.yml   # Self-hosted LangFuse stack
│
└── scripts/
    ├── provision_dept.py                 # CLI: create dept, resolve isolation, seed Qdrant
    ├── migrate_isolation.py              # CLI: run logical→physical or reverse
    ├── seed_golden_dataset.py            # CLI: auto-generate golden Q&A for a dept
    └── run_eval.py                       # CLI: run RAGAS + DeepEval, write to Postgres
```

---

## 2. Environment Variables & Third-Party Configurations

### 2.1 Core Application `.env`

```bash
# ── Application ──────────────────────────────────────────────────────
APP_ENV=production                        # development | staging | production
APP_SECRET_KEY=<32-byte-random-hex>       # JWT signing secret
APP_DEBUG=false
APP_CORS_ORIGINS=["https://app.yourdomain.com"]

# ── JWT ──────────────────────────────────────────────────────────────
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# ── LLM (main generation) ────────────────────────────────────────────
LLM_PROVIDER=openai                       # openai | anthropic | local
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-...
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096

# ── LLM (fast — classifier, metadata extraction) ─────────────────────
LLM_FAST_MODEL=gpt-4o-mini               # or claude-haiku-3
LLM_FAST_MAX_TOKENS=512

# ── Vision model (video frame captioning) ────────────────────────────
VISION_MODEL=gpt-4o                       # or local llava endpoint
VISION_API_KEY=sk-...
```

### 2.2 PostgreSQL

```bash
# ── Connection ───────────────────────────────────────────────────────
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=doc_intel
POSTGRES_USER=doc_intel_user
POSTGRES_PASSWORD=<strong-password>
DATABASE_URL=postgresql+asyncpg://doc_intel_user:<pw>@postgres:5432/doc_intel

# ── Pool ─────────────────────────────────────────────────────────────
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=10
POSTGRES_POOL_TIMEOUT=30
```

```ini
# postgresql.conf tuning for this workload
max_connections = 200
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 64MB
maintenance_work_mem = 512MB
wal_level = replica
max_wal_senders = 3
```

### 2.3 Qdrant

```bash
# ── Connection ───────────────────────────────────────────────────────
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY=<qdrant-api-key>           # required when auth enabled
QDRANT_USE_GRPC=true                      # gRPC for bulk upsert throughput
```

```yaml
# infra/qdrant/config.yaml
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
  api_key: "${QDRANT_API_KEY}"            # enable auth

storage:
  storage_path: /qdrant/storage
  snapshots_path: /qdrant/snapshots
  on_disk_payload: false                  # keep payload in RAM for filter speed
  performance:
    max_search_threads: 8

optimizers:
  deleted_threshold: 0.2
  vacuum_min_vector_number: 1000
  default_segment_number: 4              # per-collection default; heavy depts override
  max_segment_size: 200000
  memmap_threshold: 50000                # vectors above this → memory-mapped
  indexing_threshold: 20000

hnsw_index:
  m: 16
  ef_construct: 100
  full_scan_threshold: 10000

wal:
  wal_capacity_mb: 512
  wal_segments_ahead: 0

telemetry_disabled: false                # enables Prometheus /metrics endpoint
```

### 2.4 Neo4j

```bash
# ── Connection ───────────────────────────────────────────────────────
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<strong-password>
NEO4J_DATABASE=neo4j                      # single DB for Community; company_X for Enterprise
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_CONNECTION_TIMEOUT=30
```

```ini
# infra/neo4j/neo4j.conf
server.default_listen_address=0.0.0.0
server.bolt.listen_address=:7687
server.http.listen_address=:7474

# Memory
server.memory.heap.initial_size=2g
server.memory.heap.max_size=4g
server.memory.pagecache.size=4g

# Plugins (Community)
server.directories.plugins=/var/lib/neo4j/plugins
dbms.security.procedures.unrestricted=apoc.*   # APOC full access

# Logging
server.logs.debug.enabled=false
db.logs.query.enabled=info
db.logs.query.threshold=1000ms             # log slow queries

# Multi-database (Enterprise only)
# dbms.max_databases=500
```

```bash
# APOC plugin: download and place in plugins/
# https://github.com/neo4j/apoc/releases
# filename: apoc-<version>-core.jar
NEO4J_PLUGINS='["apoc"]'                  # Docker auto-install syntax
```

### 2.5 Redis & Celery

```bash
# ── Redis ────────────────────────────────────────────────────────────
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB_BROKER=0                         # Celery broker
REDIS_DB_BACKEND=1                        # Celery result backend
REDIS_DB_RATELIMIT=2                      # Rate limiter sliding windows
REDIS_DB_EMBEDCACHE=3                     # Embedding cache (SHA-256 keyed)
REDIS_PASSWORD=<redis-password>
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379

# ── Celery ───────────────────────────────────────────────────────────
CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis:6379/1
CELERY_TASK_SERIALIZER=json
CELERY_RESULT_SERIALIZER=json
CELERY_ACCEPT_CONTENT=["json"]
CELERY_TIMEZONE=UTC
CELERY_TASK_TRACK_STARTED=true
CELERY_TASK_SOFT_TIME_LIMIT=1800          # 30 min (video tasks)
CELERY_TASK_TIME_LIMIT=2100              # 35 min hard kill
CELERY_WORKER_PREFETCH_MULTIPLIER=1       # fair scheduling (one task at a time per worker)
CELERY_TASK_ACKS_LATE=true               # ack only after task completes (crash safety)
```

```ini
# Redis server tuning
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
appendonly yes
appendfsync everysec
```

### 2.6 Object Storage (S3 / MinIO)

```bash
# ── Shared (boto3 client reads these) ────────────────────────────────
S3_ENDPOINT_URL=                          # blank = AWS S3; "http://minio:9000" = MinIO
S3_ACCESS_KEY=<access-key>
S3_SECRET_KEY=<secret-key>
S3_REGION=eu-west-1
S3_BUCKET_RAW=raw-documents               # original uploads
S3_BUCKET_KEYFRAMES=video-keyframes       # JPEG strips for video segments
S3_BUCKET_AUDIT=audit-exports             # immutable audit log exports
S3_ADDRESSING_STYLE=path                  # required for MinIO; "auto" for AWS

# ── MinIO-specific (on-prem only) ────────────────────────────────────
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=<strong-password>
```

### 2.7 Embedding Models

```bash
# ── BGE-M3 (dense, standard/deep tier) ──────────────────────────────
BGE_M3_MODEL=BAAI/bge-m3
BGE_M3_USE_FP16=true
BGE_M3_MAX_LENGTH=8192
BGE_M3_BATCH_SIZE=12
BGE_M3_DEVICE=cuda                        # cpu | cuda | mps

# ── BGE-small (dense, fast tier) ─────────────────────────────────────
BGE_SMALL_MODEL=BAAI/bge-small-en-v1.5
BGE_SMALL_DEVICE=cpu                      # fast tier workers are CPU-only

# ── SPLADE (sparse) ──────────────────────────────────────────────────
SPLADE_MODEL=naver/splade-cocondenser-ensembledistil
SPLADE_DEVICE=cpu                         # sparse inference is CPU-bound

# ── CLIP (video visual embedding) ────────────────────────────────────
CLIP_MODEL=ViT-L/14                       # 768-dim, projected to 1024 in code
CLIP_DEVICE=cuda

# ── Cross-encoder (reranking) ────────────────────────────────────────
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
RERANKER_DEVICE=cpu
RERANKER_MAX_LENGTH=512

# ── Model cache directory ────────────────────────────────────────────
HF_HOME=/models/huggingface               # mounted volume in Docker
TORCH_HOME=/models/torch
```

### 2.8 Docling

```bash
# ── Docling pipeline settings ────────────────────────────────────────
DOCLING_DO_OCR=true                       # enable for scanned PDFs
DOCLING_OCR_ENGINE=tesseract              # tesseract | easyocr
DOCLING_DO_TABLE_STRUCTURE=true           # enable TableFormer
DOCLING_TABLE_MODE=ACCURATE               # ACCURATE | FAST
DOCLING_DO_CELL_MATCHING=true
DOCLING_PDF_BACKEND=pypdfium2             # pypdfium2 | dlparse
DOCLING_MAX_NUM_PAGES=500                 # reject oversized docs
DOCLING_NUM_THREADS=4
```

```python
# services/parsing/docling_parser.py — runtime config
from docling.datamodel.pipeline_options import (
    PipelineOptions, TableFormerMode, OcrOptions
)

def build_pipeline_options(doc_type: str) -> PipelineOptions:
    opts = PipelineOptions()
    opts.do_ocr = settings.DOCLING_DO_OCR
    opts.ocr_options = OcrOptions(engine=settings.DOCLING_OCR_ENGINE)
    opts.do_table_structure = settings.DOCLING_DO_TABLE_STRUCTURE
    opts.table_structure_options.mode = TableFormerMode[settings.DOCLING_TABLE_MODE]
    opts.do_cell_matching = settings.DOCLING_DO_CELL_MATCHING
    return opts
```

### 2.9 Crawl4AI

```bash
# ── Browser / crawler settings ───────────────────────────────────────
CRAWL4AI_HEADLESS=true
CRAWL4AI_JS_ENABLED=true
CRAWL4AI_TIMEOUT=30                       # seconds
CRAWL4AI_PRUNING_THRESHOLD=0.48           # content density filter
CRAWL4AI_WAIT_FOR=css:.main-content       # CSS selector to wait for dynamic render
CRAWL4AI_EXCLUDE_SOCIAL_MEDIA=true
CRAWL4AI_REMOVE_FORMS=true
CRAWL4AI_MAX_PAGES=50                     # crawl depth limit per URL job
CRAWL4AI_USER_AGENT=DocIntelBot/1.0
```

```python
# services/parsing/crawl4ai_parser.py — runtime config
from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

def build_crawl_config() -> tuple[BrowserConfig, CrawlerRunConfig]:
    browser = BrowserConfig(
        headless=settings.CRAWL4AI_HEADLESS,
        java_script_enabled=settings.CRAWL4AI_JS_ENABLED,
        user_agent=settings.CRAWL4AI_USER_AGENT,
    )
    run = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=settings.CRAWL4AI_PRUNING_THRESHOLD,
                threshold_type="fixed",
            )
        ),
        wait_for=settings.CRAWL4AI_WAIT_FOR,
        remove_forms=settings.CRAWL4AI_REMOVE_FORMS,
        exclude_social_media_links=settings.CRAWL4AI_EXCLUDE_SOCIAL_MEDIA,
        page_timeout=settings.CRAWL4AI_TIMEOUT * 1000,
    )
    return browser, run
```

### 2.10 LightRAG

```bash
# ── LightRAG settings ────────────────────────────────────────────────
LIGHTRAG_WORKING_DIR=/app/lightrag_store  # mounted volume; one subdir per dept
LIGHTRAG_LLM_MODEL=gpt-4o-mini           # entity extraction LLM
LIGHTRAG_EMBEDDING_DIM=1024              # must match BGE-M3
LIGHTRAG_MAX_ASYNC=4                     # concurrent async insertions
LIGHTRAG_GRAPH_STORAGE=Neo4JStorage
LIGHTRAG_VECTOR_STORAGE=QdrantVectorDBStorage
LIGHTRAG_KV_STORAGE=JsonKVStorage
```

```python
# services/indexing/lightrag_factory.py
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

def build_lightrag(dept_id: str) -> LightRAG:
    return LightRAG(
        working_dir=f"{settings.LIGHTRAG_WORKING_DIR}/{dept_id}",
        llm_model_func=llm_fast_func,           # gpt-4o-mini
        embedding_func=EmbeddingFunc(
            embedding_dim=settings.LIGHTRAG_EMBEDDING_DIM,
            max_token_size=8192,
            func=bge_m3_embed_func,
        ),
        graph_storage=settings.LIGHTRAG_GRAPH_STORAGE,
        vector_storage=settings.LIGHTRAG_VECTOR_STORAGE,
        kv_storage=settings.LIGHTRAG_KV_STORAGE,
        addon_params={
            "insert_batch_size": 20,
            "entity_extract_max_gleaning": 1,
        },
    )
```

### 2.11 RAG-Anything

```bash
# ── RAG-Anything settings (image captioning only) ────────────────────
RAGANYTHING_VISION_MODEL=gpt-4o
RAGANYTHING_VISION_PROVIDER=openai        # openai | local
RAGANYTHING_VISION_API_KEY=sk-...
RAGANYTHING_ENABLE_IMAGE_PROCESSING=true
RAGANYTHING_ENABLE_AUDIO_PROCESSING=false # handled by Whisper directly
RAGANYTHING_WORKING_DIR=/app/raganything_store
```

```python
# services/parsing/image_parser.py
from raganything import RAGAnything, RAGAnythingConfig

def build_raganything(dept_id: str) -> RAGAnything:
    config = RAGAnythingConfig(
        working_dir=f"{settings.RAGANYTHING_WORKING_DIR}/{dept_id}",
        enable_image_processing=True,
        enable_audio_processing=False,
        vision_model_name=settings.RAGANYTHING_VISION_MODEL,
        vision_model_provider=settings.RAGANYTHING_VISION_PROVIDER,
        embedding_func=bge_m3_embed,
    )
    return RAGAnything(config=config, llm_model_func=llm_fast_func)
```

### 2.12 LangFuse

```bash
# ── LangFuse (self-hosted) ───────────────────────────────────────────
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://langfuse:3000        # self-hosted; or https://cloud.langfuse.com
LANGFUSE_DEBUG=false
LANGFUSE_FLUSH_AT=15                      # batch size before flush to server
LANGFUSE_FLUSH_INTERVAL=0.5              # seconds
```

```yaml
# infra/langfuse/docker-compose.langfuse.yml
services:
  langfuse-server:
    image: langfuse/langfuse:latest
    ports: ["3000:3000"]
    environment:
      DATABASE_URL: postgresql://langfuse:${LANGFUSE_DB_PASSWORD}@postgres-langfuse:5432/langfuse
      NEXTAUTH_SECRET: ${LANGFUSE_NEXTAUTH_SECRET}
      NEXTAUTH_URL: http://langfuse:3000
      SALT: ${LANGFUSE_SALT}
      ENCRYPTION_KEY: ${LANGFUSE_ENCRYPTION_KEY}
      LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: false
    depends_on: [postgres-langfuse, clickhouse]

  postgres-langfuse:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: ${LANGFUSE_DB_PASSWORD}

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    environment:
      CLICKHOUSE_DB: langfuse
      CLICKHOUSE_USER: langfuse
      CLICKHOUSE_PASSWORD: ${LANGFUSE_CLICKHOUSE_PASSWORD}
```

### 2.13 Prometheus & Grafana

```yaml
# infra/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: fastapi
    static_configs:
      - targets: ["api:8000"]
    metrics_path: /metrics

  - job_name: qdrant
    static_configs:
      - targets: ["qdrant:6333"]
    metrics_path: /metrics                 # Qdrant native Prometheus endpoint

  - job_name: neo4j
    static_configs:
      - targets: ["neo4j:2004"]            # Neo4j Prometheus plugin port
    metrics_path: /metrics

  - job_name: celery
    static_configs:
      - targets: ["flower:5555"]           # Celery Flower exporter
    metrics_path: /metrics

  - job_name: redis
    static_configs:
      - targets: ["redis-exporter:9121"]

rule_files:
  - "/etc/prometheus/alerts.yml"
```

```yaml
# infra/prometheus/alerts.yml
groups:
  - name: rag_health
    rules:
      - alert: HighRetrievalLatency
        expr: histogram_quantile(0.95, rag_retrieval_latency_seconds_bucket) > 4
        for: 5m
        labels: { severity: warning }

      - alert: IngestionFailureSpike
        expr: rate(rag_ingestion_failures_total[5m]) > 0.1
        for: 2m
        labels: { severity: critical }

      - alert: QdrantMemoryHigh
        expr: qdrant_collections_total_vectors_count > 50000000
        for: 10m
        labels: { severity: warning }
```

### 2.14 LangChain (chunking / reranking)

```bash
# ── Chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE_DEFAULT=512                    # tokens for RecursiveCharacterTextSplitter
CHUNK_OVERLAP_DEFAULT=64
CHUNK_SIZE_WEB=400
CHUNK_OVERLAP_WEB=50
CHUNK_SIZE_CODE=256
CHUNK_OVERLAP_CODE=32
SEMANTIC_CHUNK_BREAKPOINT_TYPE=percentile
SEMANTIC_CHUNK_BREAKPOINT_AMOUNT=95      # top 5% semantic jumps = breakpoints
SENTENCE_WINDOW_SIZE=3                   # ±3 sentences context around target sentence
```

```python
# services/chunking/strategies.py — LangChain configuration
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import MarkdownHeaderTextSplitter

RECURSIVE_DEFAULT = RecursiveCharacterTextSplitter(
    chunk_size=int(settings.CHUNK_SIZE_DEFAULT),
    chunk_overlap=int(settings.CHUNK_OVERLAP_DEFAULT),
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=count_tokens,
)

MARKDOWN_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",  "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"),
    ],
    strip_headers=False,
)

def build_semantic_chunker(embedder) -> SemanticChunker:
    return SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type=settings.SEMANTIC_CHUNK_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=int(settings.SEMANTIC_CHUNK_BREAKPOINT_AMOUNT),
    )
```

### 2.15 Docker Compose — Full Production Stack

```yaml
# infra/docker-compose.yml
version: "3.9"

x-api-common: &api-common
  image: doc-intel-api:${VERSION:-latest}
  env_file: .env
  restart: unless-stopped
  depends_on:
    postgres:  { condition: service_healthy }
    qdrant:    { condition: service_healthy }
    neo4j:     { condition: service_healthy }
    redis:     { condition: service_healthy }

services:

  # ── FastAPI ─────────────────────────────────────────────────────────
  api:
    <<: *api-common
    ports: ["8000:8000"]
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s

  # ── Celery workers ──────────────────────────────────────────────────
  worker-pdf:
    <<: *api-common
    command: >
      celery -A workers.ingestion_worker worker
      --queues pdf,embed --concurrency 8 --loglevel info
    deploy:
      replicas: 2

  worker-video:
    image: doc-intel-api-gpu:${VERSION:-latest}
    env_file: .env
    restart: unless-stopped
    command: >
      celery -A workers.ingestion_worker worker
      --queues video --concurrency 2 --loglevel info
    deploy:
      replicas: 1
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      redis: { condition: service_healthy }

  worker-web:
    <<: *api-common
    command: >
      celery -A workers.ingestion_worker worker
      --queues web --concurrency 16 --loglevel info
    deploy:
      replicas: 2

  celery-beat:
    <<: *api-common
    command: celery -A workers.beat beat --loglevel info

  flower:
    <<: *api-common
    command: celery -A workers.ingestion_worker flower --port=5555
    ports: ["5555:5555"]

  # ── Data stores ─────────────────────────────────────────────────────
  postgres:
    image: postgres:16-alpine
    env_file: .env
    environment:
      POSTGRES_DB: doc_intel
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./infra/postgres/postgresql.conf:/etc/postgresql/postgresql.conf
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s

  qdrant:
    image: qdrant/qdrant:v1.10.1
    volumes:
      - qdrant_data:/qdrant/storage
      - qdrant_snapshots:/qdrant/snapshots
      - ./infra/qdrant/config.yaml:/qdrant/config/production.yaml
    command: ./qdrant --config-path /qdrant/config/production.yaml
    ports: ["6333:6333", "6334:6334"]
    environment:
      QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/readyz"]
      interval: 15s

  neo4j:
    image: neo4j:5-community             # swap to neo4j:5-enterprise for APOC + multi-db
    environment:
      NEO4J_AUTH: ${NEO4J_USER}/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: apoc.*
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - ./infra/neo4j/neo4j.conf:/conf/neo4j.conf
    ports: ["7474:7474", "7687:7687"]
    healthcheck:
      test: ["CMD", "neo4j", "status"]
      interval: 20s

  redis:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
      - ./infra/redis/redis.conf:/usr/local/etc/redis/redis.conf
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - minio_data:/data
    ports: ["9000:9000", "9001:9001"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s

  minio-init:
    image: minio/mc:latest
    depends_on: [minio]
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD};
      mc mb --ignore-existing local/raw-documents;
      mc mb --ignore-existing local/video-keyframes;
      mc mb --ignore-existing local/audit-exports;
      mc anonymous set none local/raw-documents;
      exit 0;"

  # ── Observability ───────────────────────────────────────────────────
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./infra/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./infra/prometheus/alerts.yml:/etc/prometheus/alerts.yml
      - prometheus_data:/prometheus
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./infra/grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports: ["3001:3000"]

  redis-exporter:
    image: oliver006/redis_exporter:latest
    environment:
      REDIS_ADDR: redis://redis:6379
      REDIS_PASSWORD: ${REDIS_PASSWORD}
    ports: ["9121:9121"]

volumes:
  pgdata:
  qdrant_data:
  qdrant_snapshots:
  neo4j_data:
  neo4j_logs:
  redis_data:
  minio_data:
  prometheus_data:
  grafana_data:
```

---

## 3. Architecture Diagrams

### 3.1 System Overview

```mermaid
flowchart TD
    subgraph CLIENT["Client (Browser / Mobile)"]
        UI[Web UI]
        SSE[SSE Consumer]
    end

    subgraph GATEWAY["Enterprise Gateway"]
        GW[NGINX / API Gateway]
        AUTH[JWT + RBAC Middleware]
        RL[Rate Limiter\nRedis sliding window]
        AUDIT[Audit Logger\nPostgres append-only]
    end

    subgraph FASTAPI["FastAPI Backend"]
        INGEST[POST /ingest]
        SEARCH[POST /search]
        CHAT[POST /chat/stream]
    end

    subgraph WORKERS["Celery Worker Pools"]
        WPDF[worker-pdf\nCPU × 8]
        WVID[worker-video\nGPU × 2]
        WWEB[worker-web\nI/O × 16]
        WEMB[worker-embed\nCPU + Network]
    end

    subgraph STORES["Data Stores"]
        PG[(PostgreSQL\nRBAC · Metadata · Quotas)]
        QD[(Qdrant\nVectors per isolation mode)]
        N4J[(Neo4j\nEntity-relation graph)]
        RD[(Redis\nBroker · Cache · Rate)]
        S3[(Object Storage\nS3 / MinIO)]
    end

    subgraph OBS["Observability"]
        LF[LangFuse\nRAG traces]
        PROM[Prometheus\n+ Grafana]
    end

    UI --> GW
    GW --> AUTH --> RL --> FASTAPI
    FASTAPI --> AUDIT
    INGEST --> RD
    RD --> WPDF & WVID & WWEB --> WEMB
    WEMB --> QD & N4J
    CHAT --> QD & N4J
    CHAT --> SSE
    FASTAPI --> PG
    WPDF & WVID & WWEB --> S3
    FASTAPI --> LF
    FASTAPI --> PROM
```

### 3.2 Multi-Tenant Data Isolation

```mermaid
flowchart LR
    subgraph COMPANY["Company ACME"]
        subgraph QDRANT_LOGICAL["Qdrant — company_acme collection\nPayload-filtered (dept_id)"]
            D1[Sales dept\ndept_id filter]
            D2[Engineering dept\ndept_id filter]
            D3[Marketing dept\ndept_id filter]
        end

        subgraph QDRANT_PHYSICAL["Qdrant — dedicated collections"]
            DL[sensitive_dept_legal]
            DH[sensitive_dept_hr]
            DF[sensitive_dept_finance_audit]
        end

        subgraph NEO4J["Neo4j — single database\nLabel-scoped per dept"]
            NL1["Dept_sales:Entity"]
            NL2["Dept_legal:Entity"]
            NL3["Dept_hr:Entity"]
        end
    end

    API[FastAPI\nHybridResolver] -->|isolation_mode=logical\nBoundClient injects dept_id| QDRANT_LOGICAL
    API -->|isolation_mode=physical\nSimpleBoundClient\ncollection = boundary| QDRANT_PHYSICAL
    API -->|all modes\nNeo4jService sanitises\nlabel in every query| NEO4J
```

### 3.3 Qdrant Isolation Decision Flow (HybridResolver)

```mermaid
flowchart TD
    START([Department provisioning request])
    REG{regulatory_flags\ncontains fedramp\nor itar or hipaa?}
    SENS{dept_type in\nlegal / hr / m_and_a\nexecutive / compliance\nfinance_audit?}
    DEFAULT[Standard dept]

    REG -->|YES| ISO[DeptSensitivity.ISOLATED\ncollection: isolated_dept_id\nDedicated cluster recommended]
    REG -->|NO| SENS
    SENS -->|YES| PHY[DeptSensitivity.SENSITIVE\ncollection: sensitive_dept_id\nDedicated collection, shared cluster]
    SENS -->|NO| DEFAULT
    DEFAULT --> LOG[DeptSensitivity.STANDARD\ncollection: company_company_id\nPayload filter: dept_id]

    ISO & PHY --> PHYSPROV[PhysicalIsolation.provision_department\nCreate named collection\nHNSW config by tier]
    LOG --> LOGPROV[LogicalIsolation.create_company_collection\nCreate if not exists\ncreate_payload_index dept_id KEYWORD]

    PHYSPROV & LOGPROV --> PG[(Store qdrant_collection\n+ isolation_mode\nin PostgreSQL)]
    PG --> QUERY[Query time:\nlook up dept.qdrant_collection\nfrom Postgres\nnever re-derive]
```

### 3.4 Ingestion Pipeline — Doc-Type-Specific Queues

```mermaid
flowchart TD
    UPLOAD[POST /dept/id/ingest\nFile upload]
    QUOTA{Quota check\nPostgres}
    ROUTE{doc_type?}

    QUOTA -->|exceeded| REJECT[HTTP 402]
    QUOTA -->|ok| ROUTE

    ROUTE -->|pdf pptx docx txt csv| QPDF[queue=pdf\nCPU workers × 8\nDocling + TableFormer]
    ROUTE -->|mp4 mov avi| QVID[queue=video\nGPU workers × 2\nWhisper + CLIP + LLaVA]
    ROUTE -->|mp3 wav m4a| QAUDIO[queue=video\nGPU workers × 2\nWhisper only]
    ROUTE -->|url| QWEB[queue=web\nI/O workers × 16\nCrawl4AI]
    ROUTE -->|jpg png| QIMG[queue=pdf\nCPU workers\nRAG-Anything caption]

    QPDF & QVID & QAUDIO & QWEB & QIMG --> SHARED

    subgraph SHARED["queue=embed — shared final stage"]
        EMB[Dense + Sparse embedding\nBGE-M3 + SPLADE]
        META[Metadata extraction\nStructural free + LLM semantic]
        QDR[Qdrant upsert\nBoundClient / SimpleBoundClient]
        GRAPH[LightRAG ainsert\nNeo4j entity graph]
        DONE[Mark document status=ready\nPostgres]
    end

    EMB --> META --> QDR --> GRAPH --> DONE

    ROLLBACK[On failure:\nDelete Qdrant points by doc_id\nMark status=failed] -.->|exception| QDR
```

### 3.5 Video RAG Pipeline — Late-Fusion Embedding

```mermaid
flowchart TD
    VIDEO[Video file\nmp4 / mov / avi]

    VIDEO --> AUDIO[Whisper v3 Large\nWord-level transcript\nwith timestamps]
    VIDEO --> FRAMES[PySceneDetect\nScene boundary keyframes\n+ uniform 1fps/5s]

    FRAMES --> CAPTION[LLaVA / GPT-4o Vision\nFrame captions]
    FRAMES --> OCR[Docling OCR\nOn-screen text overlays\nslide content in video]
    FRAMES --> CLIP[CLIP ViT-L/14\n768-dim visual vectors]

    AUDIO & CAPTION & OCR --> FUSION[Temporal fusion\n30-second VideoSegments\ntranscript + captions + OCR aligned]
    CLIP --> PROJ[Linear projection\n768 → 1024-dim]

    FUSION --> TEXTVEC[BGE-M3\ntext vector 1024-dim]
    PROJ --> MEANPOOL[Mean pool\nacross keyframes in window]
    MEANPOOL --> VISVEC[Visual vector 1024-dim]

    TEXTVEC --> QDRANT_T["Qdrant named vector: text"]
    VISVEC --> QDRANT_V["Qdrant named vector: visual"]
    FUSION --> NEO4J_N["Neo4j VideoSegment nodes\nNEXT_SEGMENT temporal chain\nMENTIONS entity edges"]
```

### 3.6 Hybrid Search Pipeline

```mermaid
flowchart TD
    QUERY[User query]

    QUERY --> CLASSIFY[QueryClassifier\nFAST / STANDARD / DEEP]

    subgraph PARALLEL["asyncio.gather — parallel retrieval legs"]
        DENSE[BGE-M3 embed\n→ Qdrant dense search\ntop-20]
        SPARSE[SPLADE embed\n→ Qdrant sparse search\ntop-20]
    end

    CLASSIFY -->|STANDARD or DEEP| DECOMP[Query decomposer\n3-5 sub-questions\nper sub-Q retrieval]
    CLASSIFY -->|FAST| DENSE

    DECOMP --> PARALLEL
    PARALLEL --> RRF[Qdrant Query API\nServer-side RRF fusion\ntop-20 fused]
    RRF --> RERANK[Cross-encoder rerank\nms-marco-MiniLM-L-12\ntop-8]
    RERANK --> MMR[MMR diversity filter\nλ=0.5\ntop-6]
    MMR -->|DEEP only| GRAPH[LightRAG graph expansion\naquery mode=hybrid\n+4 graph-discovered chunks]
    MMR -->|STANDARD| CONTEXT
    GRAPH --> CONTEXT[Final context\n10-14 diverse chunks\nCitationTracker assigns N]

    CONTEXT --> LLM[LLM generation\ncitation-grounded prompt]
    LLM --> STREAM[Structured SSE stream\nblock_start → block_chunk → block_end\ncitation events]
```

### 3.7 Adaptive Latency Routing — Three Tiers

```mermaid
flowchart LR
    Q[Query] --> CLS[QueryClassifier]

    CLS -->|Simple factual\nmatches FAST_PATTERNS\nor complexity < 0.30| FAST
    CLS -->|Moderate\n0.30 ≤ complexity < 0.65| STD
    CLS -->|compare / all / across\ncomprehensive / analysis\nor complexity ≥ 0.65| DEEP

    subgraph FAST["FAST tier — ~350ms"]
        F1[BGE-small embed\n384-dim fast]
        F2[Dense search only\nk=5]
        F3[Direct LLM stream\nno rerank no graph]
    end

    subgraph STD["STANDARD tier — ~860ms"]
        S1[BGE-M3 + SPLADE\nasyncio.gather]
        S2[Qdrant RRF\nk=10]
        S3[Cross-encoder rerank\ntop-6]
        S4[MMR filter\ntop-5]
        S5[Structured stream]
    end

    subgraph DEEP["DEEP tier — ~1.8s first token"]
        D0[Planning SSE event\n< 100ms visible to user]
        D1[Query decomposition\n3-5 sub-questions]
        D2[Per sub-Q hybrid retrieval\nasyncio.gather]
        D3[Deduplicate + cross-encoder\nrerank top-15]
        D4[MMR top-10]
        D5[LightRAG graph expansion\n+4 chunks]
        D6[Response planner]
        D7[Structured block streaming]
    end
```

### 3.8 Citation Tracking Flow

```mermaid
sequenceDiagram
    participant R as Retrieval
    participant CT as CitationTracker
    participant LLM as LLM
    participant SSE as SSE Stream

    R->>CT: register(chunk_1) → [1]
    R->>CT: register(chunk_2) → [2]
    R->>CT: register(chunk_3) → [3]
    CT->>LLM: build_context_block()\n"[1] text... (Q4_Report.pdf p.12)\n[2] text... (Policy.docx §4.2)"
    LLM->>SSE: block_start {type: table}
    LLM->>SSE: block_chunk {row: ...} [1][2]
    LLM->>SSE: block_end
    LLM->>SSE: block_start {type: text}
    LLM->>SSE: block_chunk {delta: "...growth driven by..."} [1]
    LLM->>SSE: block_end
    CT->>SSE: citation {number:1, filename:"Q4_Report.pdf", pages:[12], bbox:[88,234,520,410]}
    CT->>SSE: citation {number:2, filename:"Policy.docx", section:"§4.2"}
    SSE->>SSE: stream_end {latency_ms: {...}}
```

### 3.9 SSE Streaming Protocol — Event Sequence

```mermaid
stateDiagram-v2
    [*] --> stream_start : connection opened

    stream_start --> block_start : plan emitted\n(block types + order)

    state block_loop {
        block_start --> block_chunk : streaming content
        block_chunk --> block_chunk : more tokens / rows
        block_chunk --> block_end : block complete
        block_end --> block_start : next block
    }

    block_start --> block_loop
    block_loop --> citation : all blocks done
    citation --> citation : one per cited source
    citation --> stream_end : all citations sent
    stream_end --> [*] : connection closed

    stream_start --> error : retrieval failure
    block_loop --> error : LLM failure
    error --> [*] : connection closed
```

### 3.10 Neo4j Graph Schema

```mermaid
graph LR
    subgraph DEPT["Department subgraph — label Dept_{dept_id}"]
        DOC[Document\nid · title · doc_type\nupload_date]
        CHUNK[Chunk\nid · text · page\nbbox · content_type]
        ENT[Entity\nname · type\nRegulation/Person/Org]
        VSEG[VideoSegment\nid · start_sec · end_sec\nthumb_path]
    end

    CHUNK -->|PART_OF| DOC
    ENT -->|APPEARS_IN| CHUNK
    ENT -->|RELATED_TO\nrelation label| ENT
    ENT -->|COOCCURS_WITH\nfreq: N| ENT
    VSEG -->|PART_OF| DOC
    VSEG -->|NEXT_SEGMENT\ngap_sec: 0| VSEG
    VSEG -->|MENTIONS\nconfidence: 0.9| ENT
```

### 3.11 Observability Stack

```mermaid
flowchart LR
    subgraph APP["Application"]
        API[FastAPI]
        WKR[Celery Workers]
    end

    subgraph TRACES["RAG Traces — LangFuse"]
        SPANS[Span: hybrid_retrieval\nSpan: cross_encoder_rerank\nSpan: llm_generation]
        SCORES[Context relevance score\nUser thumbs up/down]
        PROMPT[Prompt versions\nGolden dataset]
    end

    subgraph METRICS["Infrastructure Metrics — Prometheus + Grafana"]
        P1[rag_retrieval_latency_seconds\np50/p95/p99 by tier + dept]
        P2[rag_ingestion_failures_total\nby stage + doc_type]
        P3[qdrant native /metrics\ncollection sizes · RPS]
        P4[neo4j query time\nnode counts]
        P5[redis-exporter\nmemory · hit rate]
    end

    subgraph AUDIT["Audit Logs — Postgres"]
        AL[audit_logs table\nappend-only\nuser · dept · chunks · session]
    end

    API -->|@observe decorator| TRACES
    API -->|Prometheus middleware| METRICS
    API -->|middleware write| AUDIT
    WKR -->|task events| METRICS
```

### 3.12 Evaluation Framework Pipeline

```mermaid
flowchart TD
    GEN[Golden dataset generation\nbuild_golden_dataset per dept\n100 auto-generated Q-A pairs\n20% human-reviewed]

    GEN --> RAGAS

    subgraph RAGAS["RAGAS — retrieval quality"]
        R1[Context recall @ 10\ntarget > 0.85]
        R2[Context precision\ntarget > 0.75]
        R3[Answer faithfulness\ntarget > 0.90]
        R4[Answer relevancy\ntarget > 0.85]
    end

    subgraph DEEPEVAL["DeepEval — answer quality"]
        D1[HallucinationMetric\nthreshold < 0.10]
        D2[FaithfulnessMetric\nthreshold > 0.85]
    end

    RAGAS & DEEPEVAL --> GATE{Deployment gate\nall thresholds met?}

    GATE -->|YES| DEPLOY[Deploy to production]
    GATE -->|NO| BLOCK[Block deploy\nAlert on-call\nCreate regression issue]

    DEPLOY --> PERSIST[Persist scores\neval_runs table\nPostgres]
    PERSIST --> TREND[Grafana eval dashboard\nscore trends over time\nper department]
```

---

*This document is auto-generated from the Document Intelligence System Architecture. Update both files together when the architecture changes.*