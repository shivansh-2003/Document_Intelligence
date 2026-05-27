# Document Intelligence System — Enterprise Architecture Blueprint

> **Stack**: FastAPI · PostgreSQL · Qdrant · Neo4j · Docling · RAG-Anything · LightRAG · Crawl4AI · LangChain · Python

---

## 0. Is This the Best RAG Demonstration?

**Short answer: Yes — if built correctly, this is among the most comprehensive RAG architectures possible.** Here's why it stands at the frontier:

| Dimension | Basic RAG | Advanced RAG | This System |
|---|---|---|---|
| Parsing | PyPDF / plain text | Unstructured.io | Docling + RAG-Anything + Crawl4AI |
| Embeddings | Single dense | Dense only | Dual: dense (BGE-M3) + sparse (SPLADE) |
| Search | Vector only | Vector + keyword | Vector + sparse + graph + keyword (fused) |
| Re-ranking | None | Cross-encoder | Cross-encoder + MMR diversity |
| Graph | None | None | LightRAG → Neo4j (entity/relation graph) |
| Isolation | None | Per-collection (physical) | Hybrid: scored framework (logical / physical / isolated) driven by regulatory pressure + scale |
| Streaming | Plain text | Token stream | Structured: tables, citations, equations live |
| Auth | None | Basic JWT | RBAC with department-level data isolation |

---

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENTERPRISE GATEWAY                           │
│              Auth (JWT + RBAC) · Rate Limiting · Audit Log          │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │       FASTAPI BACKEND       │
        │  /ingest  /search  /chat   │
        └──┬──────────┬──────────────┘
           │          │
    ┌──────▼──┐  ┌────▼────────────────────────────────────────┐
    │POSTGRES │  │           DEPARTMENT ISOLATION LAYER         │
    │ - users │  │  dept_id → Qdrant collection + Neo4j subgraph│
    │ - depts │  └────┬──────────────┬──────────────────────────┘
    │ - roles │       │              │
    │ - docs  │  ┌────▼────┐   ┌────▼──────┐
    │ - jobs  │  │ QDRANT  │   │  NEO4J    │
    └─────────┘  │ dense+  │   │ entities  │
                 │ sparse  │   │ relations │
                 │ per dept│   │ per dept  │
                 └─────────┘   └───────────┘
```

---

## 2. RBAC & Multi-Tenant Data Architecture

### 2.1 PostgreSQL Schema

```sql
-- Companies (top-level tenants)
CREATE TABLE companies (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    slug        TEXT UNIQUE NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Employees
CREATE TABLE users (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id   UUID REFERENCES companies(id),
    email        TEXT UNIQUE NOT NULL,
    hashed_pw    TEXT NOT NULL,
    is_active    BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Departments (each is an isolated data silo)
-- isolation_mode and dept_type are set at provisioning time by HybridResolver (see §5.4)
-- and drive ALL collection-naming decisions for both Qdrant and Neo4j.
CREATE TYPE isolation_mode_enum AS ENUM ('logical', 'physical', 'isolated');

CREATE TABLE departments (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id         UUID REFERENCES companies(id),
    name               TEXT NOT NULL,
    dept_type          TEXT NOT NULL DEFAULT 'standard',
                       -- standard | legal | hr | finance_audit | m_and_a |
                       --   executive | compliance — drives isolation decision
    isolation_mode     isolation_mode_enum NOT NULL DEFAULT 'logical',
                       -- set by HybridResolver at provisioning; never changed without migration
    regulatory_flags   TEXT[] NOT NULL DEFAULT '{}',
                       -- e.g. {'fedramp','hipaa'} — drives ISOLATED tier
    qdrant_collection  TEXT UNIQUE NOT NULL,
                       -- logical  → "company_{company_id}"
                       -- physical → "sensitive_dept_{dept_id}"
                       -- isolated → "isolated_dept_{dept_id}"
    neo4j_label        TEXT UNIQUE NOT NULL,
                       -- always "Dept_{dept_id}" regardless of isolation mode
    created_at         TIMESTAMPTZ DEFAULT NOW()
);

-- Roles: ADMIN | EDITOR | VIEWER
CREATE TYPE role_enum AS ENUM ('admin', 'editor', 'viewer');
CREATE TABLE department_memberships (
    user_id      UUID REFERENCES users(id),
    dept_id      UUID REFERENCES departments(id),
    role         role_enum NOT NULL,
    PRIMARY KEY (user_id, dept_id)
);

-- Document metadata
CREATE TABLE documents (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dept_id      UUID REFERENCES departments(id),
    filename     TEXT,
    doc_type     TEXT,          -- pdf, video, audio, pptx, url, ...
    status       TEXT,          -- pending, processing, ready, failed
    s3_path      TEXT,
    chunk_count  INT,
    uploaded_by  UUID REFERENCES users(id),
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Ingestion jobs (async)
CREATE TABLE ingestion_jobs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id       UUID REFERENCES documents(id),
    celery_task  TEXT,
    stage        TEXT,          -- parsing → chunking → embedding → indexing
    error        TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
```

### 2.2 JWT + RBAC Middleware

```python
# FastAPI dependency: resolves user + enforces dept-level access
async def require_dept_access(
    dept_id: UUID,
    role: role_enum = role_enum.viewer,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> DepartmentMembership:
    membership = await db.execute(
        select(DepartmentMembership)
        .where(DepartmentMembership.user_id == current_user.id)
        .where(DepartmentMembership.dept_id == dept_id)
    )
    m = membership.scalar_one_or_none()
    if not m or role_rank(m.role) < role_rank(role):
        raise HTTPException(403, "Insufficient department access")
    return m
```

---

## 3. Parsing Strategy — Tool Selection Per Content Type

This is the most critical architectural decision. Wrong tool = garbage in, garbage out.

### 3.1 Decision Matrix

| Content Type | Primary Tool | Secondary Tool | Why |
|---|---|---|---|
| **PDF (native text)** | Docling | — | Preserves layout, headers, columns, reading order |
| **PDF (scanned/image)** | Docling (OCR mode) | Tesseract fallback | Docling uses layout-aware OCR, not just raw text extraction |
| **Tables (in PDF/PPTX)** | Docling TableFormer | — | TableFormer is a dedicated table structure recognition model |
| **PowerPoint** | Docling | python-pptx (metadata) | Docling reads slide structure; python-pptx for speaker notes |
| **Word (.docx)** | Docling | — | Preserves headings, lists, inline styles |
| **Plain Text / Markdown** | LangChain RecursiveCharacterTextSplitter | — | Simple, no layout complexity |
| **Images** | RAG-Anything + LLaVA/GPT-4V | — | Multimodal: caption + OCR + object detection |
| **Audio** | Whisper v3 Large → text | — | High-accuracy transcription; then treat as text |
| **Video** | VideoRAG pipeline (see §3.5) | — | Frame extraction + audio + transcript fusion |
| **Website URLs** | Crawl4AI | BeautifulSoup fallback | JS-rendered pages, dynamic content, anti-bot bypass |
| **Structured Data (CSV/XLS)** | pandas → markdown tables | — | Convert to table chunks for table-aware search |
| **Email** | Python email/MIME parser | — | Headers, body, attachment extraction |

### 3.2 Docling Pipeline Configuration

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PipelineOptions, TableFormerMode

def get_docling_converter(doc_type: str) -> DocumentConverter:
    opts = PipelineOptions()

    if doc_type in ("pdf", "pptx", "docx"):
        opts.do_ocr = True                          # enable for scanned docs
        opts.do_table_structure = True              # TableFormer for tables
        opts.table_structure_options.mode = TableFormerMode.ACCURATE
        opts.do_cell_matching = True                # match cells to table structure

    return DocumentConverter(pipeline_options=opts)

async def parse_document(file_path: str, doc_type: str) -> DoclingDocument:
    converter = get_docling_converter(doc_type)
    result = converter.convert(file_path)
    return result.document   # structured DoclingDocument with tables, sections
```

### 3.3 Crawl4AI Web Parsing Pipeline

```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

async def crawl_url(url: str) -> str:
    browser_cfg = BrowserConfig(headless=True, java_script_enabled=True)

    # Deep content extraction: remove nav, ads, boilerplate
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed")
    )
    run_cfg = CrawlerRunConfig(
        markdown_generator=md_generator,
        wait_for="css:.main-content",     # wait for dynamic render
        remove_forms=True,
        exclude_social_media_links=True,
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)
        return result.markdown_v2.fit_markdown   # clean, dense markdown
```

### 3.4 RAG-Anything Multimodal Orchestration

RAG-Anything acts as the orchestration layer for non-text modalities:

```python
from raganything import RAGAnything, RAGAnythingConfig

config = RAGAnythingConfig(
    working_dir="./dept_store",
    enable_image_processing=True,
    enable_audio_processing=True,
    vision_model_name="gpt-4o",         # or local LLaVA
    vision_model_provider="openai",
    audio_model_name="whisper-large-v3",
    embedding_func=bge_m3_embed,        # your dense embed function
)

rag = RAGAnything(config=config, llm_model_func=llm_func)

# Ingest any file type — RAGAnything routes internally
await rag.process_document_complete(
    file_path=file_path,
    output_dir=dept_output_dir
)
```

### 3.5 Video RAG Pipeline (The Hard Part)

Video RAG is genuinely the most complex modality. Here is the complete strategy:

```
VIDEO FILE
    │
    ├── 1. AUDIO TRACK ──── Whisper v3 ──── Transcript (with timestamps)
    │
    ├── 2. FRAME SAMPLING
    │       ├── Keyframe extraction (PySceneDetect: scene boundaries)
    │       ├── Uniform sampling (1 frame/5s for dense coverage)
    │       └── Motion-based sampling (high-motion = more frames)
    │
    ├── 3. FRAME UNDERSTANDING (per extracted frame)
    │       ├── LLaVA-1.6 / GPT-4o Vision → frame caption
    │       ├── OCR on frame (Docling) → text overlays, slides in video
    │       └── CLIP embeddings → visual semantic embedding (768-dim ViT-L/14)
    │
    ├── 4. TEMPORAL FUSION
    │       ├── Align transcript timestamps with frame timestamps
    │       ├── Group: [00:00-00:30] → {transcript_chunk + frame_captions}
    │       └── Build VideoSegment objects with multimodal context
    │
    ├── 5. LATE FUSION EMBEDDING (per VideoSegment)
    │       ├── Text vector:   BGE-M3(transcript + captions) → 1024-dim
    │       ├── Visual vector: mean(CLIP(keyframes in window)) → proj → 1024-dim
    │       └── Stored as two NAMED VECTORS in Qdrant (not one fused vector)
    │
    └── 6. DUAL INDEXING
            ├── Qdrant: named vectors {text, visual} + sparse → hybrid searchable
            └── Neo4j: VideoSegment nodes chained by NEXT_SEGMENT temporal edges
```

#### 3.5.1 Multimodal Embedding Strategy — Late Fusion

The core question for video embedding is: **single fused vector, or separate named vectors?**

This system uses **late fusion with Qdrant named vectors**. Each `VideoSegment` is indexed with two separate named vectors — `text` (BGE-M3 over transcript + frame captions) and `visual` (mean-pooled CLIP projected to 1024-dim). At query time, a text query hits `text`; an image query (user uploads a screenshot) hits `visual`; a hybrid video query prefetches both and fuses via RRF server-side.

This strictly outperforms early fusion (projecting CLIP + BGE-M3 into one vector at index time) because it preserves each modality's geometric structure and enables independent recall legs.

```python
import torch, clip, numpy as np
from FlagEmbedding import BGEM3FlagModel
from PIL import Image

class VideoSegmentEmbedder:
    def __init__(self):
        self.text_model  = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        # ViT-L/14: best CLIP checkpoint for retrieval quality
        self.clip_model, self.clip_prep = clip.load("ViT-L/14")
        # Project CLIP 768-dim → BGE-M3 1024-dim space
        self.proj = torch.nn.Linear(768, 1024, bias=False)

    def embed_segment(self, segment: "VideoSegment") -> "VideoSegmentVectors":
        # Text vector: full transcript + all frame captions for this window
        combined = segment.transcript + " " + " ".join(segment.frame_captions)
        text_vec = self.text_model.encode(
            [combined], return_dense=True
        )["dense_vecs"][0]                              # shape: (1024,)

        # Visual vector: CLIP per keyframe → mean pool → linear projection
        if segment.keyframes:
            clip_vecs = []
            for frame_np in segment.keyframes:
                img = self.clip_prep(Image.fromarray(frame_np)).unsqueeze(0)
                with torch.no_grad():
                    feat = self.clip_model.encode_image(img)  # (1, 768)
                    feat = self.proj(feat)                     # (1, 1024)
                clip_vecs.append(feat.squeeze().cpu().numpy())
            visual_vec = np.mean(clip_vecs, axis=0)           # mean pool
        else:
            visual_vec = text_vec                             # fallback

        # Unit-normalise both for cosine distance
        text_vec   = text_vec   / np.linalg.norm(text_vec)
        visual_vec = visual_vec / np.linalg.norm(visual_vec)
        return VideoSegmentVectors(text=text_vec, visual=visual_vec)
```

#### 3.5.2 VideoSegment Storage — Qdrant Named Vectors

```python
# Collection already has named vector slots {dense, sparse} for text docs.
# Add video-specific slots at collection creation:
#   "text":   for video transcript+caption text embedding  (1024-dim)
#   "visual": for mean-pooled CLIP visual embedding        (1024-dim)

from qdrant_client.models import PointStruct

vecs = embedder.embed_segment(segment)

client.upsert(
    collection_name=dept_collection,
    points=[PointStruct(
        id=segment.id,
        vector={
            "text":   vecs.text.tolist(),
            "visual": vecs.visual.tolist(),
            # sparse omitted for video — transcript sparsity is low quality
        },
        payload={
            "content_type":    "video_segment",
            "doc_type":        "video",
            "doc_id":          segment.doc_id,
            "dept_id":         segment.dept_id,
            "filename":        segment.filename,
            "timestamp_start": segment.start_sec,
            "timestamp_end":   segment.end_sec,
            "segment_index":   segment.index,        # ordinal position in video
            "transcript":      segment.transcript,
            "frame_captions":  segment.frame_captions,
            "keyframe_paths":  segment.keyframe_s3_paths,   # S3 keys for thumbnails
        }
    )]
)

# Query: text query → text vector; image query → visual vector; hybrid → both
async def video_search(query_text: str, collection: str, k: int = 10):
    text_vec = text_embedder.embed(query_text)
    return client.query_points(
        collection_name=collection,
        prefetch=[
            Query(query=text_vec, using="text", limit=k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=Filter(must=[FieldCondition(
            key="content_type", match=MatchValue(value="video_segment")
        )]),
        limit=k, with_payload=True,
    ).points
```

#### 3.5.3 VideoSegment Storage — Neo4j Temporal Graph

```cypher
// Node per segment
CREATE (:Dept_{dept_id}:VideoSegment {
    id:            $seg_id,
    doc_id:        $doc_id,
    start_sec:     $start,
    end_sec:       $end,
    transcript:    $text,
    thumb_path:    $s3_thumb
})

// Temporal chain — enables "what came just before X?" queries
MATCH (a:VideoSegment {id: $seg_a}), (b:VideoSegment {id: $seg_b})
CREATE (a)-[:NEXT_SEGMENT {gap_sec: 0}]->(b)

// Parent document
MATCH (s:VideoSegment {id: $sid}), (d:Document {id: $did})
CREATE (s)-[:PART_OF]->(d)

// Entity mentions from transcript (fed to LightRAG graph)
MATCH (s:VideoSegment {id: $sid}), (e:Entity {name: $entity_name})
CREATE (s)-[:MENTIONS {confidence: $conf}]->(e)
```

The `NEXT_SEGMENT` chain enables temporal context queries:

```python
# "What was discussed in the 5 minutes before the pricing announcement?"
await neo4j.run("""
    MATCH path = (target:VideoSegment {id: $seg_id})
                 <-[:NEXT_SEGMENT*1..10]-(prev)
    WHERE (target.start_sec - prev.start_sec) <= 300
    RETURN prev ORDER BY prev.start_sec ASC
""", seg_id=anchor_segment_id)
```

### 3.6 Audio Pipeline

```python
import whisper
from pydub import AudioSegment

async def process_audio(audio_path: str) -> list[TextChunk]:
    # Handle all formats: mp3, mp4, m4a, wav, flac
    audio = AudioSegment.from_file(audio_path)
    wav_path = audio_path.replace(audio_path.split(".")[-1], "wav")
    audio.export(wav_path, format="wav")

    model = whisper.load_model("large-v3")
    result = model.transcribe(wav_path, word_timestamps=True)

    # Create time-anchored chunks (~60s windows)
    chunks = []
    for seg in result["segments"]:
        chunks.append(TextChunk(
            text=seg["text"],
            metadata={"start": seg["start"], "end": seg["end"], "source": "audio"}
        ))
    return chunks
```

---

## 4. Chunking Strategy — Intelligence by Content Type

The single biggest lever for RAG quality. Every content type has a different optimal chunking strategy.

### 4.1 Chunking Strategy Matrix

| Content Type | Strategy | Tool | Chunk Size | Overlap | Why |
|---|---|---|---|---|---|
| **Long-form text / prose** | Recursive Character | LangChain `RecursiveCharacterTextSplitter` | 512 tokens | 64 tokens | Preserves sentence/paragraph boundaries |
| **Technical docs / manuals** | Semantic | LangChain `SemanticChunker` (embedding-based) | Variable | 0 | Groups semantically coherent ideas |
| **Markdown / structured text** | Markdown Header | LangChain `MarkdownHeaderTextSplitter` | Per section | 0 | Headers = natural boundaries |
| **Code** | Language-aware | LangChain `RecursiveCharacterTextSplitter(language=)` | 256 tokens | 32 tokens | Preserves function/class boundaries |
| **Tables** | Row-group + full-table | Custom | Full table + row groups of 10 | 2 rows | Two chunk types: full for context, rows for lookup |
| **PDF with mixed content** | Docling layout-aware | Docling `HierarchicalChunker` | Per logical block | 0 | Respects columns, captions, headings |
| **PowerPoint** | Slide-level | Custom (1 chunk = 1 slide + speaker notes) | ~300 tokens | 0 | Slide is the semantic unit |
| **Audio / Video transcripts** | Sentence window | LangChain `SentenceWindowNodeParser` | 1 sentence + 3 context | — | Enables sentence-level retrieval with context |
| **Web pages** | Recursive + Semantic | RecursiveCharacterTextSplitter → SemanticChunker | 400 tokens | 50 tokens | Web structure is variable |
| **Images** | 1 chunk per image | — | Caption + OCR text | — | Image is atomic |

### 4.2 Table Chunking: The Dual-Representation Pattern

Tables are the hardest to chunk. The solution is dual representation:

```python
def chunk_table(table: DoclingTable) -> list[Chunk]:
    chunks = []

    # Representation 1: Full table as markdown (for comparison queries)
    full_markdown = table.export_to_markdown()
    chunks.append(Chunk(
        text=full_markdown,
        metadata={"chunk_type": "full_table", "table_id": table.id}
    ))

    # Representation 2: Row-group chunks (for specific value lookup)
    rows = table.rows
    WINDOW = 10
    for i in range(0, len(rows), WINDOW - 2):   # 2-row overlap
        window = rows[i:i+WINDOW]
        row_md = table.header_row + "\n" + "\n".join(
            [" | ".join(r.cells) for r in window]
        )
        chunks.append(Chunk(
            text=row_md,
            metadata={"chunk_type": "row_group", "table_id": table.id,
                      "row_start": i, "row_end": i+WINDOW}
        ))

    return chunks
```

### 4.3 Semantic Chunking with LangChain

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings  # swap for BGE-M3 in practice

def semantic_chunk(text: str, embed_model) -> list[str]:
    splitter = SemanticChunker(
        embeddings=embed_model,
        breakpoint_threshold_type="percentile",   # split at semantic jumps
        breakpoint_threshold_amount=95,            # top 5% = breakpoints
    )
    return splitter.create_documents([text])
```

### 4.4 Sentence Window Chunking (for granular retrieval)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk

def sentence_window_chunk(text: str, window: int = 3) -> list[SentenceWindowChunk]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i, sent in enumerate(sentences):
        lo = max(0, i - window)
        hi = min(len(sentences), i + window + 1)
        chunks.append(SentenceWindowChunk(
            sentence=sent,                                 # what gets embedded
            context=" ".join(sentences[lo:hi]),           # what gets returned to LLM
            metadata={"index": i}
        ))
    return chunks
```

### 4.5 Hierarchical Chunking with Docling

```python
from docling.chunking import HierarchicalChunker

chunker = HierarchicalChunker(
    merge_peers=True,         # merge sibling chunks below min size
    tokenizer="BAAI/bge-m3"  # token counting consistent with embedder
)

def chunk_docling_doc(doc: DoclingDocument) -> list[DocChunk]:
    chunks = list(chunker.chunk(doc))
    # Each chunk has: text, headings[], page_no, bbox (for citation rendering)
    return chunks
```

---

## 5. Dual Embedding: Dense + Sparse

### 5.1 Why Both?

| | Dense (BGE-M3) | Sparse (SPLADE) |
|---|---|---|
| Strength | Semantic similarity, paraphrase matching | Exact keyword matching, rare terms |
| Weakness | Misses exact terms | Misses paraphrase/synonym |
| Best for | "Explain quantum entanglement" | "Section 4.2.1 clause B compliance" |

Together they achieve >95% recall on most enterprise queries.

### 5.2 BGE-M3 Dense Embedding

```python
from FlagEmbedding import BGEM3FlagModel

class DenseEmbedder:
    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        output = self.model.encode(
            texts,
            batch_size=12,
            max_length=8192,       # BGE-M3 handles up to 8192 tokens
            return_dense=True,
            return_sparse=False,
        )
        return output["dense_vecs"].tolist()
```

### 5.3 SPLADE Sparse Embedding

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

class SparseEmbedder:
    def __init__(self):
        model_id = "naver/splade-cocondenser-ensembledistil"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)

    def embed(self, text: str) -> dict[int, float]:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model(**tokens)
        logits = output.logits
        # SPLADE: RELU(LOG(1 + exp(logits))) max-pooled over tokens
        vecs = torch.log(1 + torch.relu(logits)).max(dim=1).values.squeeze()
        # Return sparse dict: {token_id: weight} filtering near-zero
        indices = vecs.nonzero().squeeze().cpu().tolist()
        values  = vecs[vecs != 0].cpu().tolist()
        return dict(zip(indices, values))
```

### 5.4 Multi-Tenant Vector Storage — Decision Framework

The original architecture hardcoded physical isolation (one Qdrant collection per department). That is correct for regulated, low-count deployments but wrong for most enterprise configurations: at 50 departments, physical isolation carries 2.5–5 GB of baseline RAM overhead before indexing a single vector. The right isolation mode is a function of regulatory pressure, tenant scale, cross-tenant query need, operational maturity, and data residency requirements.

#### 5.4.1 The Five Decision Dimensions

Score your deployment 1–5 on each axis before writing a single collection.

| Dimension | 1 (Simple) | 3 (Moderate) | 5 (Complex) |
|---|---|---|---|
| **Regulatory pressure** | Internal docs only | SOC 2, ISO 27001 | HIPAA, GDPR Art. 32, FedRAMP, ITAR |
| **Tenant scale** | < 20 departments | 20–200 departments | 200–10,000+ departments |
| **Cross-tenant query need** | Never | Monthly executive reports | Real-time cross-department analytics |
| **Operational maturity** | Startup, one SRE | Mid-size, dedicated platform team | Enterprise, 24/7 NOC |
| **Data residency variance** | Single region | Multi-region (EU + US) | Per-tenant country/region requirements |

**Sum interpretation**: 5–11 → logical isolation · 12–17 → hybrid · 18–25 → physical isolation default.

#### 5.4.2 Scoring Worksheet

Use this before provisioning. Sum the weighted column.

| Criterion | Weight | Score (1–5) | Weighted |
|---|---|---|---|
| Regulatory requirement for physical separation | 3.0 | ___ | ___ |
| Number of departments (now + 2yr forecast) | 2.5 | ___ | ___ |
| Cross-department query frequency | 2.0 | ___ | ___ |
| Operational team capacity | 1.5 | ___ | ___ |
| Data residency variance | 1.5 | ___ | ___ |
| Performance SLA variance per dept | 1.0 | ___ | ___ |
| Tenant churn rate | 1.0 | ___ | ___ |
| Audit granularity required | 1.0 | ___ | ___ |
| **TOTAL** | **13.5** | | **___** |

**Interpretation**: < 20 → Pattern A (logical) · 20–35 → Pattern C (hybrid) · > 35 → Pattern B (physical)

#### 5.4.3 Decision Tree

```
START
 │
 ├─ Regulatory requirement mandates PHYSICAL separation?
 │   ├─ YES → Physical isolation (per-dept collection or isolated cluster)
 │   │         └─ >100 sensitive depts? → Shard to multiple Qdrant clusters
 │   └─ NO → Continue
 │
 ├─ Total departments > 500?
 │   ├─ YES → Logical isolation mandatory (payload filter)
 │   │         └─ Any sensitive depts mixed in? → Hybrid: sensitive get physical
 │   └─ NO → Continue
 │
 ├─ Cross-department search needed by power users?
 │   ├─ YES → Logical or Hybrid (physical blocks cross-dept)
 │   └─ NO → Continue
 │
 ├─ Ops team < 3 engineers?
 │   ├─ YES → Logical (less operational surface area)
 │   └─ NO → Continue
 │
 └─ DEFAULT → Hybrid (Pattern C)
               Standard depts → company collection (logical)
               Sensitive depts (Legal/HR/Exec) → dedicated collection (physical)
```

#### 5.4.4 Pattern A — Pure Logical Isolation

**When**: Low regulatory pressure, high scale, cross-tenant queries needed.

Every department's vectors live in one collection per company, isolated by `dept_id` payload filter. The `BoundClient` wrapper makes the filter structurally impossible to omit.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, SparseVectorParams, SparseIndexParams,
    Distance, Filter, FieldCondition, MatchValue,
    PayloadSchemaType, HnswConfigDiff, OptimizersConfigDiff
)

class LogicalIsolation:
    def __init__(self, client: QdrantClient):
        self.client = client

    def create_company_collection(self, company_id: str) -> str:
        collection = f"company_{company_id}"
        self.client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE),
                "text":  VectorParams(size=1024, distance=Distance.COSINE),   # video
                "visual": VectorParams(size=1024, distance=Distance.COSINE),  # video CLIP
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            optimizers_config=OptimizersConfigDiff(default_segment_number=4),
        )
        # NON-NEGOTIABLE: keyword index on dept_id
        # Without this, every query is an O(n) full-collection scan.
        self.client.create_payload_index(
            collection_name=collection,
            field_name="dept_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        return collection

    def get_bound_client(self, company_id: str, dept_id: str) -> "BoundClient":
        return BoundClient(self.client, f"company_{company_id}", dept_id)
```

**Pros**: One backup, one schema migration, cross-dept search possible, scales to 10k+ tenants.
**Cons**: Single point of compromise if filter injection is bypassed — mitigated entirely by `BoundClient`.

#### 5.4.5 Pattern B — Pure Physical Isolation

**When**: HIPAA / FedRAMP / ITAR mandate infrastructure-level separation; < 50 sensitive departments; zero cross-tenant queries.

```python
class PhysicalIsolation:
    def __init__(self, client: QdrantClient):
        self.client = client

    def provision_department(self, dept_id: str,
                              tier: str = "standard") -> str:
        collection = f"dept_{dept_id}"
        if self.client.collection_exists(collection):
            return collection   # idempotent

        hnsw = (HnswConfigDiff(m=32, ef_construct=200, max_indexing_threads=4)
                if tier == "heavy"
                else HnswConfigDiff(m=16, ef_construct=100, max_indexing_threads=2))

        self.client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense":   VectorParams(size=1024, distance=Distance.COSINE),
                "text":    VectorParams(size=1024, distance=Distance.COSINE),
                "visual":  VectorParams(size=1024, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
            hnsw_config=hnsw,
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2,
                indexing_threshold=20000,  # defer index for small collections
            ),
        )
        return collection

    def decommission(self, dept_id: str):
        """Snapshot before delete — never delete without a snapshot."""
        collection = f"dept_{dept_id}"
        self.client.create_snapshot(collection_name=collection)
        self.client.delete_collection(collection_name=collection)

    def rebalance_alert(self, threshold: int = 1000) -> list[tuple[str, int]]:
        """
        Collections < threshold points waste baseline RAM (~50–100 MB each).
        Return a list for the ops team to consider migrating to logical isolation.
        """
        small = []
        for c in self.client.get_collections().collections:
            info = self.client.get_collection(c.name)
            if info.points_count < threshold and c.name.startswith("dept_"):
                small.append((c.name, info.points_count))
        return small
```

**Pros**: Provable isolation, independent per-dept scaling, easy per-dept SLAs.
**Cons**: ~50–100 MB RAM baseline per collection; operational cost explodes beyond 50 departments; no cross-dept search.

**Operational rule**: If you choose physical isolation, collection lifecycle MUST be automated. Manual provisioning/decommission is an incident waiting to happen.

#### 5.4.6 Pattern C — Hybrid Isolation (Recommended Default)

**When**: Mixed environment — most departments are standard, but Legal / HR / Finance Audit / Executive need hard walls.

Physical isolation is a **property of the department type**, not the department instance. All standard departments share a company collection (logical). All sensitive department types get dedicated collections (physical). The decision is made once at provisioning and stored in the `departments.isolation_mode` column.

```python
from enum import Enum

class DeptSensitivity(Enum):
    STANDARD  = "standard"    # → company collection (logical)
    SENSITIVE = "sensitive"   # → dedicated collection (physical, same cluster)
    ISOLATED  = "isolated"    # → dedicated collection + dedicated cluster / VPC

SENSITIVE_DEPT_TYPES = {
    "legal", "hr", "m_and_a", "executive",
    "compliance", "finance_audit",
}

class HybridResolver:
    """
    Single source of truth for collection naming.
    Called ONCE at department provisioning — result stored in Postgres.
    All subsequent lookups read from Postgres, not from this class.
    """
    def __init__(self, qdrant: QdrantClient):
        self.qdrant = qdrant
        self.logical  = LogicalIsolation(qdrant)
        self.physical = PhysicalIsolation(qdrant)

    def classify(self, dept_type: str,
                 regulatory_flags: list[str]) -> DeptSensitivity:
        if any(f in regulatory_flags for f in ("fedramp", "itar", "hipaa")):
            return DeptSensitivity.ISOLATED
        if dept_type in SENSITIVE_DEPT_TYPES:
            return DeptSensitivity.SENSITIVE
        return DeptSensitivity.STANDARD

    def provision(self, company_id: str, dept_id: str,
                  dept_type: str,
                  regulatory_flags: list[str]) -> tuple[str, DeptSensitivity]:
        """
        Returns (qdrant_collection_name, sensitivity).
        Caller persists both to Postgres departments table.
        """
        sensitivity = self.classify(dept_type, regulatory_flags)

        if sensitivity == DeptSensitivity.ISOLATED:
            collection = f"isolated_dept_{dept_id}"
            self.physical.provision_department(dept_id)  # dedicated cluster in prod
        elif sensitivity == DeptSensitivity.SENSITIVE:
            collection = f"sensitive_dept_{dept_id}"
            self.physical.provision_department(dept_id)
        else:
            collection = f"company_{company_id}"
            # Ensure company collection exists (idempotent)
            if not self.qdrant.collection_exists(collection):
                self.logical.create_company_collection(company_id)

        return collection, sensitivity

    def get_bound_client(self, dept: "Department") -> "BoundClient | SimpleBoundClient":
        """
        Called at query time. Reads isolation_mode from the Department ORM object
        (pre-loaded from Postgres) — never re-derives from dept_type.
        """
        if dept.isolation_mode == "logical":
            return BoundClient(self.qdrant, dept.qdrant_collection, str(dept.id))
        else:
            return SimpleBoundClient(self.qdrant, dept.qdrant_collection)
```

#### 5.4.7 BoundClient — SDK-Level Filter Enforcement

Filter injection in business logic is an anti-pattern: one missing call, one data breach. The `BoundClient` makes the `dept_id` filter structurally impossible to omit — every method signature omits the filter parameter because the filter is already built in.

```python
class BoundClient:
    """
    For logical isolation (company_ collections).
    Every operation is automatically scoped to dept_id.
    The raw QdrantClient is never exposed to callers.
    """
    def __init__(self, client: QdrantClient,
                 collection: str, dept_id: str):
        self._client     = client
        self._collection = collection
        self._base_filter = Filter(must=[
            FieldCondition(key="dept_id", match=MatchValue(value=dept_id))
        ])

    def _merge(self, extra: Filter | None) -> Filter:
        if not extra:
            return self._base_filter
        return Filter(must=[*self._base_filter.must, *extra.must])

    def query(self, prefetch: list, query, limit: int = 10,
              extra_filter: Filter | None = None):
        return self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=query,
            query_filter=self._merge(extra_filter),
            limit=limit,
            with_payload=True,
        )

    def upsert(self, points: list):
        # Ensure dept_id is present in every payload before writing
        for p in points:
            assert "dept_id" in p.payload, "dept_id missing from payload"
        return self._client.upsert(
            collection_name=self._collection, points=points
        )

    def delete_by_doc_id(self, doc_id: str):
        return self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter=Filter(must=[
                *self._base_filter.must,
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            ])),
        )


class SimpleBoundClient:
    """
    For physical/isolated collections.
    Collection name IS the boundary — no filter injection needed.
    """
    def __init__(self, client: QdrantClient, collection: str):
        self._client     = client
        self._collection = collection

    def query(self, prefetch: list, query, limit: int = 10,
              extra_filter: Filter | None = None):
        return self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=query,
            query_filter=extra_filter,
            limit=limit,
            with_payload=True,
        )

    def upsert(self, points: list):
        return self._client.upsert(
            collection_name=self._collection, points=points
        )

    def delete_by_doc_id(self, doc_id: str):
        return self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            ])),
        )
```

#### 5.4.8 Anti-Patterns

| Anti-pattern | Why it fails | The fix |
|---|---|---|
| "Start per-collection and migrate later" | Migration requires re-embedding every vector. At 10M vectors, that is days of compute. | Decide at Day 0 using the scoring worksheet. |
| "One collection, no payload index on `dept_id`" | Every query scans all vectors — latency scales O(n) with total documents, not per-department. | `create_payload_index` is non-negotiable for logical isolation. |
| "Inject `dept_id` filter in business logic" | One bug, one missing filter, one data breach. | Inject at the `BoundClient` level — never expose the raw client. |
| "Sensitive depts in their own collection, same cluster" | Same cluster = shared RAM, disk, network. True regulatory isolation (FedRAMP/ITAR) needs cluster-level separation. | `ISOLATED` departments → dedicated Qdrant cluster or VPC. |
| "One collection per user instead of per department" | Qdrant collections have ~50–100 MB baseline RAM overhead. 1,000 users = 50–100 GB RAM before indexing. | Collections are for department/company boundaries only. |
| "Derive collection name from dept_type at query time" | If dept_type changes (e.g. Sales becomes Finance Audit), queries silently route to the wrong collection. | Collection name is stored in `departments.qdrant_collection` at provisioning and never re-derived. |

#### 5.4.9 Migration Paths

**Path A — Logical → Physical** (compliance audit failure, auditor mandates physical separation)

Shadow indexing + atomic resolver cutover. Zero downtime.

```python
async def migrate_logical_to_physical(
    client: QdrantClient, db: AsyncSession,
    company_id: str, dept_id: str
):
    source = f"company_{company_id}"
    target = f"sensitive_dept_{dept_id}"

    # 1. Provision physical collection
    PhysicalIsolation(client).provision_department(dept_id)

    # 2. Scroll + copy in batches (never load all into memory)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=source, offset=offset, limit=500,
            with_vectors=True, with_payload=True,
            filter=Filter(must=[
                FieldCondition(key="dept_id", match=MatchValue(value=dept_id))
            ]),
        )
        if not points:
            break
        for p in points:
            p.payload.pop("dept_id", None)   # redundant in physical collection
        client.upsert(collection_name=target, points=points)

    # 3. Atomic switch in Postgres — new queries immediately use physical collection
    await db.execute(
        """UPDATE departments
           SET isolation_mode = 'physical', qdrant_collection = $1
           WHERE id = $2""",
        target, dept_id
    )
    await db.commit()

    # 4. After 24h verification window, delete from logical collection
    client.delete(
        collection_name=source,
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="dept_id", match=MatchValue(value=dept_id))
        ])),
    )
```

**Path B — Physical → Logical** (scale pressure: 500+ collections, RAM exhaustion)

Gradual consolidation by company. Inject `dept_id` into payload during migration.

```python
async def migrate_physical_to_logical(
    client: QdrantClient, db: AsyncSession,
    company_id: str, dept_ids: list[str]
):
    target = f"company_{company_id}"
    logical = LogicalIsolation(client)
    if not client.collection_exists(target):
        logical.create_company_collection(company_id)

    for dept_id in dept_ids:
        source = f"dept_{dept_id}"
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=source, offset=offset, limit=500,
                with_vectors=True, with_payload=True,
            )
            if not points:
                break
            for p in points:
                p.payload["dept_id"] = dept_id   # inject for payload filter
            client.upsert(collection_name=target, points=points)

        # Update Postgres atomically, then decommission
        await db.execute(
            """UPDATE departments
               SET isolation_mode = 'logical', qdrant_collection = $1
               WHERE id = $2""",
            target, dept_id
        )
        await db.commit()
        client.create_snapshot(collection_name=source)   # archive
        client.delete_collection(collection_name=source)
```

#### 5.4.10 Non-Negotiables Checklist

Before indexing a single embedding:

- [ ] **Regulatory review**: Does any data class require physical or cluster-level isolation?
- [ ] **Scale forecast**: Department count at 12 months? 24 months?
- [ ] **Cross-department use cases**: Do executives search across departments?
- [ ] **Operational capacity**: Can your team handle N collection backups and migrations?
- [ ] **`create_payload_index` on `dept_id`**: Applied to every logical/company collection before first upsert.
- [ ] **`BoundClient` audit**: Is the raw `QdrantClient` accessible from any route handler? It must not be.
- [ ] **Naming convention documented**: `company_{uuid}` · `sensitive_dept_{uuid}` · `isolated_dept_{uuid}` · `dept_{uuid}`
- [ ] **`isolation_mode` stored in Postgres**: Collection name is looked up from `departments.qdrant_collection` — never re-derived at query time.
- [ ] **Migration runbook exists**: If you pick the wrong model, can you reach the other model with zero downtime?

**Recommended starting point for this Document Intelligence System**: **Pattern C — Hybrid Isolation**. Standard departments (Sales, Engineering, Marketing, Product) share a company collection. Legal, HR, Finance Audit, M&A, and Executive get dedicated physical collections. This satisfies auditors for the 10–15% of departments that matter most while keeping the other 85% operationally simple. Individual departments can be migrated between modes without touching the rest of the system.

---

## 6. Hybrid Search with Qdrant Query API

Qdrant 1.10+ allows server-side fusion — no client-side RRF needed.

### 6.1 Query Flow

```
User Query
    │
    ├─→ Dense embed (BGE-M3) ──────────────────────┐
    ├─→ Sparse embed (SPLADE) ─────────────────────┤
    └─→ BM25 keyword (Qdrant built-in) ────────────┤
                                                    │
                                         Qdrant Query API
                                    (Reciprocal Rank Fusion)
                                                    │
                                          Top-K fused results
                                                    │
                                     Cross-encoder re-ranking
                                                    │
                                     MMR diversity filtering
                                                    │
                                    LightRAG graph expansion
                                                    │
                                         Final context window
```

### 6.2 Qdrant Hybrid Query (Dense + Sparse + RRF)

```python
from qdrant_client.models import (
    Query, FusionQuery, NearestQuery, SparseVector, Fusion
)

async def hybrid_search(
    client: QdrantClient,
    collection: str,
    query_text: str,
    dense_embedder: DenseEmbedder,
    sparse_embedder: SparseEmbedder,
    top_k: int = 20
) -> list[ScoredPoint]:

    dense_vec = dense_embedder.embed_batch([query_text])[0]
    sparse_vec = sparse_embedder.embed(query_text)

    results = client.query_points(
        collection_name=collection,
        prefetch=[
            # Dense search leg
            Query(query=dense_vec, using="dense", limit=top_k),
            # Sparse search leg
            Query(
                query=SparseVector(
                    indices=list(sparse_vec.keys()),
                    values=list(sparse_vec.values())
                ),
                using="sparse",
                limit=top_k
            ),
        ],
        # Server-side Reciprocal Rank Fusion
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return results.points
```

---

## 7. Re-Ranking Pipeline (Cross-Encoder + MMR)

### 7.1 Cross-Encoder Re-Ranking

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self):
        # Best open-source cross-encoder for passage retrieval
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    def rerank(self, query: str, passages: list[str], top_n: int = 8) -> list[int]:
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_n]
```

### 7.2 MMR Diversity Filtering

MMR prevents returning 8 copies of the same information. It balances relevance vs. novelty:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mmr_filter(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: list,
    k: int = 6,
    lambda_val: float = 0.5   # 0=max diversity, 1=max relevance
) -> list:
    selected = []
    selected_embeds = []
    remaining = list(range(len(candidates)))

    for _ in range(k):
        if not remaining:
            break

        # Score = λ * relevance - (1-λ) * max_similarity_to_selected
        query_sims = cosine_similarity([query_embedding], candidate_embeddings[remaining])[0]

        if selected_embeds:
            sel_sims = cosine_similarity(candidate_embeddings[remaining], selected_embeds)
            max_sel_sims = sel_sims.max(axis=1)
        else:
            max_sel_sims = np.zeros(len(remaining))

        mmr_scores = lambda_val * query_sims - (1 - lambda_val) * max_sel_sims
        best = remaining[np.argmax(mmr_scores)]
        selected.append(candidates[best])
        selected_embeds.append(candidate_embeddings[best])
        remaining.remove(best)

    return selected
```

---

## 8. Graph RAG with LightRAG + Neo4j

### 8.1 Why Graph RAG?

Vector search retrieves similar chunks. Graph RAG retrieves **connected knowledge** — relationships between entities across documents. For enterprise docs, this is invaluable:

- "What are all the compliance requirements that affect our European operations?" → requires traversing: Dept → Documents → Regulations → Clauses → Affected Operations
- "Which employees are mentioned across Q1 and Q4 reports?" → entity co-reference across documents

### 8.2 LightRAG Integration

```python
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete  # or any LLM
from lightrag.utils import EmbeddingFunc

async def build_lightrag_index(
    texts: list[str],
    working_dir: str,
    embed_func: EmbeddingFunc,
    llm_func
) -> LightRAG:
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=embed_func,
        graph_storage="Neo4JStorage",       # use Neo4j backend
        vector_storage="QdrantVectorDBStorge",
        kv_storage="JsonKVStorage",
    )
    for text in texts:
        await rag.ainsert(text)
    return rag
```

### 8.3 LightRAG Query Modes

```python
# Hybrid mode: combines local (entity-centric) + global (graph-wide) retrieval
result = await rag.aquery(
    "What compliance risks exist across all departments?",
    param=QueryParam(mode="hybrid")   # "local" | "global" | "hybrid" | "naive"
)
```

### 8.4 Neo4j Graph Schema

```cypher
-- Nodes per department (prefixed by dept label)
(:DeptFinance_a1b2:Entity {name: "GDPR", type: "Regulation"})
(:DeptFinance_a1b2:Document {id: "uuid", title: "Q3 Report"})
(:DeptFinance_a1b2:Chunk {id: "uuid", text: "...", page: 3})

-- Edges
(entity)-[:APPEARS_IN]->(chunk)
(chunk)-[:PART_OF]->(document)
(entity)-[:RELATED_TO {relation: "governs"}]->(entity)
(entity)-[:COOCCURS_WITH {freq: 7}]->(entity)
```

### 8.5 Neo4j Scaling Strategy

> **Coupling with Qdrant isolation**: The `HybridResolver` (§5.4.6) drives both stores. When it provisions a department, the `neo4j_label` is always `Dept_{dept_id}` regardless of Qdrant isolation mode — Neo4j isolation is always label-scoped within a single database (or company database in multi-database Enterprise mode). The two isolation decisions are independent: a `SENSITIVE` department gets a dedicated Qdrant collection but shares the same Neo4j database as standard departments, separated only by its node label prefix.

#### The Problem with One Database Per Department

Spinning up a separate Neo4j database per department (e.g. via `CREATE DATABASE dept_finance`) is operationally expensive: each database runs its own set of background threads, memory pools, and transaction logs. At 50+ departments on a single Neo4j instance this becomes unmanageable — memory pressure, slow restarts, and a DBA nightmare. **The recommended architecture is a single Neo4j database with strong node-label isolation and APOC-enforced security.**

#### Option A — Single Database, Label-Scoped Isolation (Recommended)

All departments share one Neo4j database. Isolation is enforced by the `dept_id`-prefixed node label on every node, plus database constraints and APOC role-based subgraph security. Every Cypher query in the application layer is constructed with a hard-coded label filter — it is impossible to write a query that retrieves nodes from a different department without explicitly using that department's label.

```cypher
-- Uniqueness constraint per dept per entity name
CREATE CONSTRAINT dept_entity_unique IF NOT EXISTS
FOR (e:DeptFinance_a1b2:Entity)
REQUIRE e.name IS UNIQUE;

-- Node key: ensures no dept can share a chunk_id with another dept
CREATE CONSTRAINT dept_chunk_key IF NOT EXISTS
FOR (c:DeptFinance_a1b2:Chunk)
REQUIRE c.id IS NODE KEY;

-- Index for fast entity lookup within a dept
CREATE INDEX dept_entity_name IF NOT EXISTS
FOR (e:DeptFinance_a1b2:Entity) ON (e.name);
```

Every application-layer Cypher query MUST include the dept label in the MATCH pattern. This is enforced in the `Neo4jService` wrapper — raw Cypher is never exposed to callers:

```python
class Neo4jService:
    def __init__(self, driver: AsyncDriver):
        self.driver = driver

    def _dept_label(self, dept_id: str) -> str:
        # Sanitise: only alphanumeric + underscore — prevents Cypher injection
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", dept_id)
        return f"Dept_{safe}"

    async def find_related_entities(self, dept_id: str, entity_name: str,
                                    hops: int = 2) -> list[dict]:
        label = self._dept_label(dept_id)
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH (start:{label}:Entity {{name: $name}})
                      -[:RELATED_TO*1..{hops}]-(related:{label}:Entity)
                RETURN DISTINCT related.name AS name, related.type AS type
                LIMIT 50
            """, name=entity_name)
            return [r.data() async for r in result]

    async def get_temporal_context(self, dept_id: str, seg_id: str,
                                   window_sec: int = 300) -> list[dict]:
        label = self._dept_label(dept_id)
        async with self.driver.session() as session:
            result = await session.run(f"""
                MATCH path = (target:{label}:VideoSegment {{id: $seg_id}})
                             <-[:NEXT_SEGMENT*1..10]-(prev:{label}:VideoSegment)
                WHERE (target.start_sec - prev.start_sec) <= $window
                RETURN prev ORDER BY prev.start_sec ASC
            """, seg_id=seg_id, window=window_sec)
            return [r.data() async for r in result]
```

#### Option B — APOC Subgraph Security (Enterprise Add-On)

For licensed Neo4j Enterprise, APOC provides fine-grained subgraph-level read restrictions via custom security procedures. This prevents even a misconfigured query from leaking cross-dept nodes:

```cypher
-- APOC: restrict a Neo4j role to only see nodes with a specific label
CALL apoc.security.addRolePermission(
    "dept_finance_role",
    "read_nodes",
    ["DeptFinance_a1b2"]   -- only nodes with this label are visible
)
```

Map each department to a dedicated Neo4j role. The FastAPI service authenticates to Neo4j using per-dept credentials, so even a raw Cypher injection can only read that department's subgraph.

#### Option C — Neo4j Multi-Database (Enterprise, Large Scale)

For organizations with 100+ departments and Neo4j Enterprise licensed:

```python
# Create a dedicated Neo4j database per company (not per dept)
# Departments within the same company share one database (label isolation)
# Different companies get different databases — stronger isolation boundary

async def provision_company_database(company_id: str, driver: AsyncDriver):
    db_name = f"company_{company_id.replace('-', '_')}"
    async with driver.session(database="system") as session:
        await session.run(f"CREATE DATABASE {db_name} IF NOT EXISTS")
    return db_name
```

This gives company-level database isolation (hard boundary) with department-level label isolation within each database — a two-tier model that scales to thousands of departments across hundreds of companies.

#### Scaling Decision Matrix

| Scenario | Recommended approach |
|---|---|
| < 100 departments, any Neo4j edition | Single database, label isolation + constraints |
| Regulatory requirement for hard data boundaries per tenant | Multi-database (one DB per company) + label isolation per dept |
| Security-critical with Neo4j Enterprise | APOC subgraph security + label isolation |
| Self-hosted, cost-sensitive | Single database, label isolation only (free Community edition) |

---

## 9. Streaming Structured LLM Responses

This is what separates this system from toy RAG demos. The LLM doesn't stream plain text — it streams **structured components** that render progressively.

### 9.1 The Problem with Plain-Text RAG

A query like: *"Compare our Q1 and Q4 revenue performance across all product lines, including key drivers"*

Plain RAG returns: a 500-word paragraph (hard to read, no structure).

This system streams: a live table + annotation + chart spec + citations.

### 9.2 Structured Streaming Protocol

We use a **tagged-chunk protocol** over SSE (Server-Sent Events):

```
event: block_start
data: {"block_id": "tbl_1", "type": "table", "title": "Revenue Comparison"}

event: block_chunk
data: {"block_id": "tbl_1", "row": {"Product": "Widget A", "Q1": "$2.1M", "Q4": "$3.4M", "Delta": "+62%"}}

event: block_chunk
data: {"block_id": "tbl_1", "row": {"Product": "Widget B", "Q1": "$0.8M", "Q4": "$0.6M", "Delta": "-25%"}}

event: block_end
data: {"block_id": "tbl_1"}

event: block_start
data: {"block_id": "txt_1", "type": "text", "heading": "Key Drivers"}

event: block_chunk
data: {"block_id": "txt_1", "delta": "Widget A growth was driven by..."}

event: citation
data: {"source": "Q4_Report.pdf", "page": 12, "chunk_id": "abc123"}
```

### 9.3 FastAPI Streaming Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import json

@router.post("/dept/{dept_id}/chat/stream")
async def stream_chat(
    dept_id: UUID,
    query: ChatQuery,
    _: DepartmentMembership = Depends(require_dept_access)
):
    async def event_generator():
        # 1. Retrieval
        chunks = await full_retrieval_pipeline(dept_id, query.text)

        # 2. Classify response type needed
        response_plan = await plan_response_structure(query.text, chunks)
        # e.g.: [{"type": "table"}, {"type": "text"}, {"type": "citation_list"}]

        # 3. Stream each block
        for block in response_plan:
            yield {
                "event": "block_start",
                "data": json.dumps({"block_id": block.id, "type": block.type})
            }

            async for token in llm_stream_block(block, chunks, query.text):
                yield {
                    "event": "block_chunk",
                    "data": json.dumps({"block_id": block.id, "delta": token})
                }

            yield {"event": "block_end", "data": json.dumps({"block_id": block.id})}

        # 4. Citations
        for c in chunks[:5]:
            yield {
                "event": "citation",
                "data": json.dumps({
                    "source": c.metadata["filename"],
                    "page": c.metadata.get("page"),
                    "chunk_id": c.id
                })
            }

    return EventSourceResponse(event_generator())
```

### 9.4 Structured Prompt for Comprehensive Responses

The key to getting the LLM to produce rich structured output (not short answers) is a carefully engineered system prompt:

```python
SYSTEM_PROMPT = """You are an enterprise document intelligence assistant.

RESPONSE STRUCTURE RULES:
1. For comparison queries → ALWAYS produce a markdown table first, then analysis
2. For multi-faceted queries → produce H2 sections for each facet
3. For data queries → extract numbers into a structured table, NEVER bury them in prose
4. For "explain" queries → use progressive disclosure: summary → detail → examples
5. NEVER produce a response shorter than the complexity demands

CITATION FORMAT:
After every factual claim, append [SOURCE: <filename>, p.<page>]

EQUATION FORMAT (for formulas):
Use LaTeX: $$ CAGR = \left(\frac{V_f}{V_i}\right)^{1/t} - 1 $$

TABLE FORMAT:
Always use proper markdown tables with alignment:
| Metric | Q1 | Q4 | Change |
|:-------|---:|---:|-------:|

MINIMUM RESPONSE REQUIREMENTS:
- Comparison query: 1+ table + 2+ paragraphs + citations
- Summary query: executive summary + bullet points + citations
- Analysis query: sections + data evidence + conclusion

Retrieved context:
{context}
"""
```

---

## 10. The Comprehensive Answer Problem — Use Case

### Use Case: "Give me a full competitive analysis of our product lines vs market benchmarks"

**What basic RAG returns**: 3 sentences. Why? Because it retrieves top-3 chunks and stuffs them into a short prompt.

**Root Causes of Short RAG Answers**:
1. Too few chunks retrieved (k=3)
2. No instruction to be comprehensive
3. Token budget not allocated to the answer
4. No response plan — LLM guesses format

**This System's Solution**:

```python
async def comprehensive_query_pipeline(query: str, dept_id: UUID):
    # Step 1: Decompose complex query into sub-questions
    sub_questions = await decompose_query(query)
    # e.g.: ["What are our product line revenues?",
    #         "What are market benchmark revenues?",
    #         "What are our growth rates vs market?"]

    # Step 2: Retrieve per sub-question (k=10 each, then deduplicate)
    all_chunks = []
    for sq in sub_questions:
        chunks = await hybrid_search(dept_collection, sq, top_k=10)
        all_chunks.extend(chunks)
    unique_chunks = deduplicate_by_chunk_id(all_chunks)

    # Step 3: Re-rank the full set against the ORIGINAL query
    reranked = cross_encoder_rerank(query, unique_chunks, top_n=15)

    # Step 4: MMR for diversity (avoid 5 chunks from the same doc)
    diverse_chunks = mmr_filter(query_embed, reranked, k=10)

    # Step 5: Graph expansion — add related entity chunks from Neo4j
    graph_chunks = await lightrag_expand(query, diverse_chunks)
    final_context = diverse_chunks + graph_chunks  # ~12-14 total chunks

    # Step 6: Response planning — tell the LLM what sections to produce
    response_plan = await plan_response(query, final_context)

    # Step 7: Stream structured response
    async for event in stream_structured_response(query, final_context, response_plan):
        yield event
```

### The Query Decomposition Step (Critical)

```python
async def decompose_query(query: str) -> list[str]:
    prompt = f"""Break this query into 3-5 specific sub-questions that together answer it completely.
Return ONLY a JSON array of strings.
Query: {query}"""

    response = await llm(prompt)
    return json.loads(response)
```

This single step is what transforms "3 sentence answer" into "full competitive analysis with tables."

---

## 11. Complete FastAPI Backend Structure

```
backend/
├── main.py                          # FastAPI app entry
├── core/
│   ├── config.py                    # Settings (env vars)
│   ├── security.py                  # JWT + password hashing
│   └── database.py                  # Async SQLAlchemy engine
├── models/
│   ├── user.py
│   ├── department.py
│   └── document.py
├── api/
│   ├── auth.py                      # /auth/login, /auth/register
│   ├── departments.py               # /depts/ CRUD
│   ├── ingest.py                    # /ingest/ upload + async job
│   ├── search.py                    # /search/ hybrid query
│   └── chat.py                      # /chat/stream SSE endpoint
├── services/
│   ├── parsing/
│   │   ├── docling_parser.py
│   │   ├── crawl4ai_parser.py
│   │   ├── audio_parser.py
│   │   ├── video_parser.py
│   │   └── multimodal_orchestrator.py  # RAG-Anything
│   ├── chunking/
│   │   ├── strategies.py            # All chunking functions
│   │   └── table_chunker.py
│   ├── embedding/
│   │   ├── dense_embedder.py        # BGE-M3
│   │   └── sparse_embedder.py       # SPLADE
│   ├── indexing/
│   │   ├── qdrant_service.py
│   │   └── neo4j_service.py
│   ├── retrieval/
│   │   ├── hybrid_retriever.py      # Qdrant Query API
│   │   ├── graph_retriever.py       # LightRAG
│   │   ├── reranker.py              # Cross-encoder
│   │   └── mmr.py
│   └── generation/
│       ├── query_decomposer.py
│       ├── response_planner.py
│       └── structured_streamer.py
├── workers/
│   └── ingestion_worker.py          # Celery tasks
└── tests/
```

---

## 12. Key Engineering Decisions & Why

| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| Dense embedder | BGE-M3 | text-embedding-3-large | Free, 8192 context, multilingual, same quality |
| Sparse embedder | SPLADE | BM25 | SPLADE learns term importance from training; BM25 is pure frequency |
| Fusion | Qdrant RRF (server) | Client-side RRF | Zero latency overhead; server knows internal scores |
| Graph store | Neo4j | NetworkX | Production-grade, persistent, Cypher query language |
| Re-ranker | ms-marco-MiniLM-L-12 | Cohere Rerank API | Free, fast, production quality |
| PDF parser | Docling | PyMuPDF / pdfplumber | TableFormer for tables; layout-aware reading order |
| Web parser | Crawl4AI | Scrapy / requests | JS rendering, anti-bot, clean markdown output |
| Streaming | SSE + tagged blocks | WebSocket | SSE is simpler, HTTP-compatible, resumable |
| Task queue | Celery + Redis | FastAPI Background | Ingestion can take minutes; Celery is restartable |

---

## 13. Production Deployment Architecture

```yaml
# docker-compose (simplified)
services:
  api:
    image: doc-intel-api
    ports: ["8000:8000"]
    depends_on: [postgres, qdrant, neo4j, redis]

  worker:
    image: doc-intel-api
    command: celery -A workers.ingestion_worker worker --loglevel=info
    deploy:
      replicas: 4      # Scale ingestion horizontally

  postgres:
    image: postgres:16
    volumes: [pgdata:/var/lib/postgresql/data]

  qdrant:
    image: qdrant/qdrant:latest
    volumes: [qdrant_data:/qdrant/storage]
    ports: ["6333:6333"]

  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/password
    volumes: [neo4j_data:/data]
    ports: ["7474:7474", "7687:7687"]

  redis:
    image: redis:7-alpine
```

---

## 14. Retrieval Quality Metrics (How to Evaluate)

Never deploy without measuring:

| Metric | Tool | Target |
|---|---|---|
| Recall@10 | RAGAS | > 0.85 |
| Precision@5 | RAGAS | > 0.75 |
| Answer Faithfulness | RAGAS | > 0.90 |
| Answer Relevance | RAGAS | > 0.85 |
| Context Precision | RAGAS | > 0.80 |
| End-to-end latency | Custom | < 3s first token |

---

## 15. Metadata Tagging Strategy

Metadata is not documentation — it is a retrieval filter. Every field stored must answer: *"what query should this chunk be excluded from?"* not *"what is nice to know about this chunk?"*

### 15.1 Complete Chunk Metadata Schema

```python
@dataclass
class ChunkMetadata:
    # --- Identity (RBAC + deduplication) ---
    chunk_id:         str          # "chnk_a3f9b2"
    doc_id:           str          # parent document
    dept_id:          str          # enforced at query time as Qdrant must-filter

    # --- Structure (free — extracted by Docling, zero LLM cost) ---
    doc_type:         str          # pdf | pptx | video | audio | url | csv
    content_type:     str          # text | table | image | equation | code | slide
    chunk_strategy:   str          # how this chunk was created (row_group, semantic, etc.)
    hierarchy_level:  int          # 0=body, 1=h1, 2=h2 — heading depth
    parent_section:   str          # "Revenue Analysis"
    headings_path:    list[str]    # ["Q4 Report", "Revenue Analysis", "EMEA"]

    # --- Location (enables deep-link citations) ---
    page_number:      int | None   # PDF page
    slide_number:     int | None   # PPTX slide
    timestamp_start:  float | None # video/audio seconds
    timestamp_end:    float | None
    bbox:             list[int] | None  # [x0, y0, x1, y1] for PDF highlight overlay

    # --- Semantic (one small LLM call per chunk at index time) ---
    language:         str          # "en", "de", etc.
    entities:         list[str]    # ["GDPR", "European Commission", "Q4 2024"]
    topics:           list[str]    # ["compliance", "revenue", "europe"]
    semantic_density: float        # unique_meaningful_tokens / total_tokens  (0–1)
    has_numbers:      bool         # regex for digits — boosts on data queries
    has_citations:    bool         # chunk itself contains source references

    # --- Administrative ---
    token_count:      int
    filename:         str
    upload_date:      str
    created_by:       str
    recency_score:    float        # exp(-days_since_upload / 180) — decays over time
```

### 15.2 Why Each Field Earns Its Place

`content_type` is the single highest-leverage tag. Pre-filtering to `content_type = table AND has_numbers = true` before vector search shrinks the candidate pool from 50,000 mixed chunks to ~200 table chunks. Semantic scores matter far more inside a homogeneous candidate set. This filter runs as a Qdrant `must` condition before any distance calculation — it is free.

`headings_path` enables two things: citations that say *"Q4 Report → Revenue Analysis → EMEA"* instead of just *"p.12"*, and parent-section retrieval — if no chunk directly answers a query but a section heading matches semantically, the full section can be fetched.

`bbox` (bounding box from Docling) is what separates *"PDF, p.12"* from a citation that highlights the exact paragraph in a PDF viewer. The frontend fetches the PDF and renders a highlight overlay at those coordinates. For slides, `slide_number` renders a thumbnail. For video, `timestamp_start` renders a *"play from 4:32"* button.

`recency_score` uses exponential decay: `exp(-days_since_upload / 180)`. Applied as a payload-weighted reranking boost in Qdrant. For regulatory or policy queries this is critical — a 2024 policy update must outrank a 2019 one even if the 2019 text is semantically closer.

`semantic_density = unique_meaningful_tokens / total_tokens` filters boilerplate. Footer text, table-of-contents pages, and repeated headers all score below 0.3. Setting a minimum threshold eliminates this noise before embedding, saving index space and improving search precision.

### 15.3 Query-Time Metadata Pre-Filtering Examples

| Query type | Qdrant `must` filter applied | Effect |
|---|---|---|
| "Show me the Q4 revenue table" | `content_type = table AND has_numbers = true` | 10× smaller candidate set |
| "What does slide 8 say about pricing?" | `doc_type = pptx AND slide_number = 8` | Exact lookup — no vector needed |
| "Find recent policy updates" | `recency_score > 0.7 AND topics contains policy` | Temporal boost + topic filter |
| "What did GDPR say about data retention?" | `entities contains GDPR` | Entity pre-filter before semantic search |
| "Find the auth code snippet" | `content_type = code` | Eliminates all prose, tables, slides |
| "Anything from the legal team this quarter?" | `created_by IN legal_team_ids AND upload_date > 90_days_ago` | Authorship + recency scope |

### 15.4 Metadata Extraction Pipeline (Ingestion Time)

```python
async def extract_metadata(chunk: RawChunk, doc: DoclingDocument) -> ChunkMetadata:

    # Step 1 — Structural fields: free, from Docling (zero LLM cost)
    structural = {
        "page_number":     chunk.meta.page_no,
        "bbox":            chunk.meta.bbox,
        "hierarchy_level": chunk.meta.heading_level,
        "headings_path":   chunk.meta.headings,
        "content_type":    infer_content_type(chunk),   # table/text/image/code
        "token_count":     count_tokens(chunk.text),
    }

    # Step 2 — Computed fields: pure Python, free
    computed = {
        "has_numbers":      bool(re.search(r'\d+\.?\d*', chunk.text)),
        "has_citations":    bool(re.search(r'\[\d+\]|\(.*\d{4}\)', chunk.text)),
        "semantic_density": len(set(meaningful_tokens(chunk.text))) / max(len(chunk.text.split()), 1),
        "recency_score":    math.exp(-days_since(doc.upload_date) / 180),
        "language":         detect_language(chunk.text),
    }

    # Step 3 — Semantic fields: one fast LLM call per chunk (haiku/gpt-4o-mini)
    semantic_prompt = f"""Extract from this text:
1. Named entities (people, orgs, regulations, dates) as a JSON list
2. Topic tags (3 max) as a JSON list
Return: {{"entities": [...], "topics": [...]}}
Text: {chunk.text[:800]}"""

    semantic = json.loads(await llm_fast(semantic_prompt))

    return ChunkMetadata(**structural, **computed, **semantic,
                         chunk_id=new_uuid(), doc_id=doc.id, dept_id=doc.dept_id,
                         doc_type=doc.doc_type, filename=doc.filename,
                         upload_date=doc.upload_date, created_by=doc.uploaded_by)
```

### 15.5 Storage: Qdrant Payload vs PostgreSQL

Store the full metadata as a Qdrant point payload — filterable without a separate DB lookup. Keep the payload lean (< 2KB per chunk): include all filterable fields but never store large text blobs. Key identity fields (`doc_id`, `dept_id`, `filename`) are also mirrored in PostgreSQL for joins and admin queries.

```python
client.upsert(
    collection_name=dept_collection,
    points=[PointStruct(
        id=chunk.chunk_id,
        vector={"dense": dense_vec, "sparse": sparse_vec},
        payload=asdict(chunk_metadata)   # full metadata as Qdrant payload
    )]
)
```

---

## 16. Citation and Reference Architecture

### 16.1 Three Citation Layers

| Layer | Unit | What it enables |
|---|---|---|
| Chunk-level | `chunk_id` | Atomic citation — maps to exact passage, table row group, or slide |
| Document-level | `doc_id` | Groups chunk citations per document → "pp. 4, 7, 12" instead of 3 entries |
| Deep-link | `bbox` / `timestamp_start` / `slide_number` | PDF highlight overlay, video seek, slide preview |

### 16.2 CitationTracker — Assigning Numbers Before the LLM Prompt

The LLM must be given pre-numbered chunks, not asked to figure out citations itself. If you send 10 chunks and ask the LLM to cite sources, it will hallucinate citation numbers and conflate sources. The correct approach: assign `[1]`, `[2]`, `[3]` before constructing the prompt.

```python
class CitationTracker:
    def __init__(self):
        self.used_chunks: list[ChunkWithMeta] = []
        self.citation_map: dict[str, int] = {}  # chunk_id → number

    def register(self, chunk: ChunkWithMeta) -> int:
        if chunk.id not in self.citation_map:
            n = len(self.used_chunks) + 1
            self.citation_map[chunk.id] = n
            self.used_chunks.append(chunk)
        return self.citation_map[chunk.id]

    def build_context_block(self) -> str:
        """Numbered context for the LLM prompt."""
        lines = []
        for chunk in self.used_chunks:
            n = self.citation_map[chunk.id]
            loc = f"p.{chunk.metadata['page_number']}" if chunk.metadata.get('page_number') \
                  else f"slide {chunk.metadata['slide_number']}" if chunk.metadata.get('slide_number') \
                  else f"t={chunk.metadata['timestamp_start']:.0f}s" if chunk.metadata.get('timestamp_start') \
                  else chunk.metadata.get('parent_section', '')
            lines.append(f"[{n}] {chunk.text}\n    (Source: {chunk.metadata['filename']}, {loc})")
        return "\n\n".join(lines)

    def render_citation_strip(self) -> list[CitationCard]:
        """Grouped by doc_id, pages merged."""
        by_doc: dict[str, list[ChunkWithMeta]] = {}
        for chunk in self.used_chunks:
            by_doc.setdefault(chunk.metadata['doc_id'], []).append(chunk)

        cards = []
        for doc_chunks in by_doc.values():
            pages = sorted(set(
                c.metadata['page_number'] for c in doc_chunks
                if c.metadata.get('page_number')
            ))
            cards.append(CitationCard(
                numbers=[self.citation_map[c.id] for c in doc_chunks],
                filename=doc_chunks[0].metadata['filename'],
                pages=pages,                              # "pp. 4, 7, 12"
                content_type=doc_chunks[0].metadata['content_type'],
                bbox=doc_chunks[0].metadata.get('bbox'),
                timestamp=doc_chunks[0].metadata.get('timestamp_start'),
                slide_number=doc_chunks[0].metadata.get('slide_number'),
            ))
        return cards
```

### 16.3 LLM Prompt for Citation-Grounded Responses

```python
CITATION_SYSTEM_PROMPT = """You are answering exclusively from the retrieved document chunks below.

CITATION RULES (non-negotiable):
- After EVERY factual claim, append the citation number: [1] or [1][2] for multiple
- For comparison tables: add a footer row "Sources: [1][2]"
- For equations extracted from documents: cite as [N, eq. 3.2]
- Never state a fact not present in the context below
- If chunks conflict on a fact, cite both and note the discrepancy

RESPONSE FORMAT:
- Comparison queries → table first, then analysis, then citations
- Data queries → extract numbers into a structured table, never bury in prose
- Every section heading must be followed by at least one cited claim

CONTEXT:
{tracker.build_context_block()}
"""
```

### 16.4 Deep-Link Citation Rendering (Frontend Contract)

Each `CitationCard` streamed to the frontend contains everything needed to render an interactive citation:

```python
@dataclass
class CitationCard:
    numbers:      list[int]       # [1] or [1, 3] if same doc cited multiple times
    filename:     str             # "Q4_Board_Report_2024.pdf"
    pages:        list[int]       # [4, 7, 12] → rendered as "pp. 4, 7, 12"
    content_type: str             # "table" → badge shown
    # Deep-link fields (mutually exclusive by doc type):
    bbox:         list[int] | None   # PDF: [x0,y0,x1,y1] for highlight overlay
    timestamp:    float | None       # Video: seek to this second
    slide_number: int | None         # PPTX: render slide thumbnail
```

The frontend renders:
- PDF → open PDF viewer at `page`, draw highlight rect from `bbox`
- Video → seek to `timestamp`, show transcript panel
- PPTX → show slide thumbnail in a popover

---

## 17. Adaptive Latency — Matching Pipeline Depth to Query Complexity

### 17.1 The Core Insight

Most RAG systems run every query through the full pipeline: embed → hybrid search → rerank → MMR → graph → LLM. For *"what is our company name?"* that is 2–4 seconds of wasted compute. For *"give me a full competitive analysis"* skipping graph expansion reduces quality. The fix is a **query classifier** that routes each query to the appropriate depth before any retrieval starts.

### 17.2 The Four Routing Tiers

| Tier | Trigger | Pipeline steps | Target first-token latency |
|---|---|---|---|
| **Instant** | Cache hit, exact metadata lookup | Redis → return | < 100ms |
| **Fast** | Single concept, narrow scope, simple factual | Dense-only search, k=5, direct LLM | 0.8–1.5s |
| **Standard** | Multi-concept, moderate complexity | Hybrid search, k=10, cross-encoder, MMR | 2–4s |
| **Deep** | Analytical, comparison, comprehensive, "all/across/every" | Decompose → multi-retrieval → graph expand → plan → stream | 5–12s (first token at ~1s via planning block) |

### 17.3 Query Classifier

```python
import re
from enum import Enum

class QueryTier(Enum):
    FAST     = "fast"
    STANDARD = "standard"
    DEEP     = "deep"

class QueryClassifier:
    FAST_PATTERNS = [
        r"what is \w+\??$", r"define \w+", r"who is",
        r"when (was|did|is)", r"how many \w+\??$",
        r"what does \w+ stand for",
    ]
    DEEP_KEYWORDS = [
        "compare", "versus", "vs", "comprehensive", "analysis",
        "all", "every", "across", "trend", "overview", "summary of all",
        "impact", "implications", "strategic", "risks and opportunities",
    ]
    DEEP_PATTERNS = [
        r"how does .+ affect",
        r"what are (all|the) .+ (across|between|for each)",
        r"give me a (full|complete|detailed|thorough)",
    ]

    async def classify(self, query: str, dept_id: str) -> QueryTier:
        q = query.lower().strip()

        # 1. Fast-path pattern match — free, < 1ms
        if any(re.search(p, q) for p in self.FAST_PATTERNS):
            return QueryTier.FAST

        # 2. Deep-path keyword/pattern match — free, < 1ms
        if any(kw in q for kw in self.DEEP_KEYWORDS):
            return QueryTier.DEEP
        if any(re.search(p, q) for p in self.DEEP_PATTERNS):
            return QueryTier.DEEP

        # 3. LLM complexity score — fast model, ~50ms
        score = await self._llm_complexity_score(query)
        if score < 0.30: return QueryTier.FAST
        if score < 0.65: return QueryTier.STANDARD
        return QueryTier.DEEP

    async def _llm_complexity_score(self, query: str) -> float:
        prompt = f"""Score this query's answer complexity from 0.0 to 1.0.
0.0 = single fact answer. 1.0 = requires multi-document synthesis, comparison, or analysis.
Return ONLY a float. Query: {query}"""
        return float(await llm_fast(prompt))  # haiku / gpt-4o-mini
```

### 17.4 Tier-Specific Pipeline Execution

```python
async def route_and_execute(query: str, dept_id: str) -> AsyncGenerator:
    tier = await classifier.classify(query, dept_id)
    collection = resolve_collection(dept_id)

    if tier == QueryTier.FAST:
        # Dense-only, small model, no rerank, no graph
        chunks = await qdrant_dense_search(collection, query, k=5)
        async for token in llm_stream(query, chunks, model="haiku"):
            yield token

    elif tier == QueryTier.STANDARD:
        # Full hybrid, cross-encoder, MMR — skip graph
        chunks = await hybrid_search(collection, query, k=10)
        chunks = cross_encoder_rerank(query, chunks, top_n=6)
        chunks = mmr_filter(query_embed, chunks, k=5)
        async for block in structured_stream(query, chunks):
            yield block

    elif tier == QueryTier.DEEP:
        # Stream a planning signal immediately so the UI feels live
        yield stream_event("planning", {"message":
            f"Analyzing across multiple documents — decomposing your query..."})

        # Full deep pipeline
        sub_qs = await decompose_query(query)
        all_chunks = await asyncio.gather(*[
            hybrid_search(collection, sq, k=10) for sq in sub_qs
        ])
        unique = deduplicate(flatten(all_chunks))
        reranked = cross_encoder_rerank(query, unique, top_n=15)
        diverse  = mmr_filter(query_embed, reranked, k=10)
        graph    = await lightrag_expand(query, diverse)
        final    = diverse + graph

        plan = await plan_response_structure(query, final)
        async for block in structured_stream(query, final, plan):
            yield block
```

### 17.5 Parallel Retrieval — The Single Biggest Latency Win

The most impactful latency optimization is running dense and sparse retrieval concurrently. Most implementations await them sequentially — 80ms + 50ms = 130ms. The correct approach:

```python
# WRONG — sequential (130ms total)
dense_vec  = await embed_dense(query)    # 80ms
dense_hits = await qdrant_dense(dense_vec)  # 50ms
sparse_vec = await embed_sparse(query)   # 20ms (after dense finishes)

# RIGHT — parallel (80ms total, bottleneck is the slower leg only)
async def hybrid_search_parallel(query: str, collection: str, k: int):
    async def dense_leg():
        vec = await embed_dense(query)
        return await qdrant_search(collection, "dense", vec, k)

    async def sparse_leg():
        vec = await embed_sparse(query)
        return await qdrant_search(collection, "sparse", vec, k)

    dense_hits, sparse_hits = await asyncio.gather(dense_leg(), sparse_leg())
    return qdrant_rrf_fuse(dense_hits, sparse_hits)
```

### 17.6 Latency Budget by Tier

| Step | Fast | Standard | Deep |
|---|---|---|---|
| Classification | 1ms | 25ms | 50ms |
| Query decomposition | — | — | 200ms |
| Dense embedding | 15ms (small) | 80ms | 80ms |
| Sparse embedding | — | 20ms ‖ dense | 20ms ‖ dense |
| Qdrant search | 30ms | 50ms | 80ms |
| Cross-encoder rerank | — | 250ms | 250ms |
| MMR filtering | — | 30ms | 30ms |
| Graph expansion | — | — | 600ms |
| LLM first token | 300ms | 400ms | 500ms |
| **Total** | **~350ms** | **~860ms** | **~1.8s** (planning token at ~100ms) |

> The "Deep" total is time-to-first-content-token. The planning event streams at ~100ms so the UI is never blank. Full response completes in 5–12s depending on answer length.

### 17.7 Tiered Embedding Models

Fast tier uses `bge-small-en-v1.5` (384-dim, ~15ms, 512-token context). Standard and Deep use `bge-m3` (1024-dim, ~80ms, 8192-token context, multilingual). Fast-tier queries are narrow and specific — they don't need multilingual support or long context. Using the smaller model halves embedding latency for ~40% of all queries.

```python
def select_embedder(tier: QueryTier) -> Embedder:
    if tier == QueryTier.FAST:
        return bge_small_embedder   # 384-dim, fast
    return bge_m3_embedder          # 1024-dim, full quality
```

---

## 18. Updated FastAPI Backend Structure

The new modules slot into the existing service layer:

```
backend/services/
├── vector_store/                        ← NEW: isolation framework (§5.4)
│   ├── resolver.py                      # HybridResolver + DeptSensitivity
│   ├── bound_client.py                  # BoundClient + SimpleBoundClient
│   ├── logical_isolation.py             # LogicalIsolation (Pattern A)
│   ├── physical_isolation.py            # PhysicalIsolation + lifecycle (Pattern B)
│   └── migration.py                     # migrate_logical_to_physical + reverse
├── metadata/
│   ├── extractor.py             # ChunkMetadata extraction pipeline
│   ├── schema.py                # ChunkMetadata dataclass
│   └── recency.py               # Recency score computation
├── citations/
│   ├── tracker.py               # CitationTracker class
│   ├── renderer.py              # CitationCard → SSE event
│   └── deep_links.py            # PDF bbox / video timestamp / slide logic
├── routing/
│   ├── classifier.py            # QueryClassifier + QueryTier
│   └── pipeline_executor.py    # route_and_execute() dispatcher
├── parsing/  ...                # (unchanged)
├── chunking/ ...                # (unchanged)
├── embedding/
│   ├── dense_embedder.py        # BGE-M3 (standard/deep)
│   ├── small_embedder.py        # BGE-small (fast tier) ← new
│   └── sparse_embedder.py       # SPLADE
├── retrieval/ ...               # (unchanged)
└── generation/ ...              # (unchanged)
```

---

## 19. Orchestration Layer — Doc-Type-Specific Worker Queues

RAG-Anything is a powerful multimodal orchestrator but it does not natively know about Docling's structured output, your custom chunking strategies, or LightRAG's graph indexing. A single `IngestionOrchestrator` class that handles every doc type in one flow is the wrong shape: video ingestion (GPU-heavy, minutes-long per file) and PDF ingestion (CPU-bound, seconds-long) cannot share a worker pool without one starving the other.

### 19.1 The Problem with a Monolithic Worker

A single Celery queue with homogeneous workers forces a bad tradeoff:

- **CPU-sized workers** (4 vCPU, no GPU): video ingestion blocks for 10–20 minutes per file, queuing all PDF jobs behind it.
- **GPU workers**: massively over-provisioned and expensive for PDF parsing, which never touches the GPU.
- **I/O workers**: wrong profile entirely for Whisper or CLIP inference.

The fix is **three separate Celery queues with resource-matched worker pools**, routing each doc type at submission time.

### 19.2 Queue Architecture

```
Upload API
    │
    ├── doc_type ∈ {pdf, pptx, docx, txt, csv}
    │       └──→ queue=pdf   (CPU workers, Docling)
    │
    ├── doc_type ∈ {mp4, mov, avi, mp3, wav, m4a}
    │       └──→ queue=video  (GPU workers, Whisper + CLIP + LLaVA)
    │
    └── doc_type ∈ {url}
            └──→ queue=web   (I/O workers, Crawl4AI)

All queues share a final step: queue=embed → queue=index
(embedding and Qdrant indexing are always CPU + network, same worker profile)
```

### 19.3 Shared IngestionContext

The `IngestionContext` dataclass is unchanged — it is still the single state object threaded through every stage. What changes is which Celery task picks it up and which worker pool executes it.

```python
from dataclasses import dataclass, field
from enum import Enum

class IngestionStage(Enum):
    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"
    GRAPH = "graph"
    DONE  = "done"

@dataclass
class IngestionContext:
    doc_id:     str
    dept_id:    str
    doc_type:   str
    file_path:  str
    s3_key:     str
    raw_chunks: list = field(default_factory=list)
    chunks:     list = field(default_factory=list)
    stage:      IngestionStage = IngestionStage.PARSE
    error:      str | None = None
```

### 19.4 Doc-Type-Specific Celery Tasks

```python
# workers/ingestion_worker.py
from celery import Celery, chain

app = Celery(
    "doc_intel",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

# ── PDF / PPTX / DOCX / text ─────────────────────────────────────────────────

@app.task(bind=True, max_retries=3, default_retry_delay=30,
          queue="pdf", acks_late=True)
def parse_document_task(self, ctx_dict: dict) -> dict:
    """CPU worker — Docling parsing, layout analysis, TableFormer."""
    ctx = IngestionContext(**ctx_dict)
    try:
        ctx.stage = IngestionStage.PARSE
        doc = docling_parser.parse(ctx.file_path, ctx.doc_type)
        ctx.raw_chunks = doc.export_chunks()          # bbox, headings preserved
        ctx.stage = IngestionStage.CHUNK
        ctx.chunks = chunk_and_tag(ctx.raw_chunks, ctx)
        persist_stage(ctx)
        return asdict(ctx)
    except Exception as exc:
        self.retry(exc=exc)

# ── Video / Audio ─────────────────────────────────────────────────────────────

@app.task(bind=True, max_retries=2, default_retry_delay=120,
          queue="video", acks_late=True,
          soft_time_limit=1800, time_limit=2100)   # 30-min soft, 35-min hard
def parse_video_task(self, ctx_dict: dict) -> dict:
    """GPU worker — Whisper, CLIP, LLaVA captioning. Longest-running task."""
    ctx = IngestionContext(**ctx_dict)
    try:
        ctx.stage = IngestionStage.PARSE
        ctx.raw_chunks = video_pipeline.process_video(
            ctx.file_path, ctx.doc_id, ctx.dept_id
        )
        ctx.stage = IngestionStage.CHUNK
        ctx.chunks = chunk_and_tag(ctx.raw_chunks, ctx)
        persist_stage(ctx)
        return asdict(ctx)
    except SoftTimeLimitExceeded:
        # Partial progress: index what was completed so far
        if ctx.raw_chunks:
            ctx.chunks = chunk_and_tag(ctx.raw_chunks, ctx)
            persist_stage(ctx)
            return asdict(ctx)
        raise

@app.task(bind=True, max_retries=3, default_retry_delay=30,
          queue="video", acks_late=True)
def parse_audio_task(self, ctx_dict: dict) -> dict:
    """GPU worker — Whisper transcription. Shares the GPU queue with video."""
    ctx = IngestionContext(**ctx_dict)
    ctx.raw_chunks = audio_pipeline.process_audio(ctx.file_path)
    ctx.chunks = chunk_and_tag(ctx.raw_chunks, ctx)
    persist_stage(ctx)
    return asdict(ctx)

# ── Web / URL ─────────────────────────────────────────────────────────────────

@app.task(bind=True, max_retries=5, default_retry_delay=15,
          queue="web", acks_late=True)
def parse_url_task(self, ctx_dict: dict) -> dict:
    """I/O worker — Crawl4AI. High concurrency, no GPU, tolerates retries."""
    ctx = IngestionContext(**ctx_dict)
    text = crawl4ai_parser.crawl(ctx.file_path)      # file_path = URL string
    ctx.raw_chunks = [RawChunk(text=text)]
    ctx.chunks = chunk_and_tag(ctx.raw_chunks, ctx)
    persist_stage(ctx)
    return asdict(ctx)

# ── Shared final stages (always CPU + network) ────────────────────────────────

@app.task(bind=True, max_retries=3, default_retry_delay=30,
          queue="embed", acks_late=True)
def embed_and_index_task(self, ctx_dict: dict) -> dict:
    """CPU + network worker — BGE-M3, SPLADE, Qdrant upsert, LightRAG graph."""
    ctx = IngestionContext(**ctx_dict)
    try:
        # Embedding
        ctx.stage = IngestionStage.EMBED
        texts = [c.text for c in ctx.chunks]
        dense_vecs  = dense_embedder.embed_batch(texts)
        sparse_vecs = [sparse_embedder.embed(t) for t in texts]
        for i, chunk in enumerate(ctx.chunks):
            chunk.dense_vec  = dense_vecs[i]
            chunk.sparse_vec = sparse_vecs[i]

        # Qdrant indexing
        ctx.stage = IngestionStage.INDEX
        points = [build_qdrant_point(c) for c in ctx.chunks]
        qdrant_service.upsert_batch(ctx.dept_id, points)

        # LightRAG graph — same text slice as Qdrant, consistent entities
        ctx.stage = IngestionStage.GRAPH
        full_text = "\n\n".join(c.text for c in ctx.chunks)
        lightrag.ainsert_sync(full_text, dept_id=ctx.dept_id)

        ctx.stage = IngestionStage.DONE
        persist_stage(ctx)
        return asdict(ctx)
    except Exception as exc:
        # Rollback: delete any partially indexed Qdrant points for this doc
        qdrant_service.delete_by_doc_id(ctx.dept_id, ctx.doc_id)
        self.retry(exc=exc)
```

### 19.5 Routing at Upload Time

```python
# api/ingest.py
DOC_TYPE_QUEUE = {
    "pdf": parse_document_task,
    "pptx": parse_document_task,
    "docx": parse_document_task,
    "txt": parse_document_task,
    "csv": parse_document_task,
    "mp4": parse_video_task,
    "mov": parse_video_task,
    "avi": parse_video_task,
    "mp3": parse_audio_task,
    "wav": parse_audio_task,
    "m4a": parse_audio_task,
    "url": parse_url_task,
}

@router.post("/dept/{dept_id}/ingest")
async def ingest_document(dept_id: UUID, file: UploadFile, ...):
    await check_ingestion_quota(dept_id, file_size_gb, doc_type)

    ctx = IngestionContext(
        doc_id=str(uuid4()), dept_id=str(dept_id),
        doc_type=doc_type, file_path=local_tmp_path, s3_key=s3_key
    )

    parse_task = DOC_TYPE_QUEUE[doc_type]

    # Chain: parse (type-specific queue) → embed+index (shared embed queue)
    pipeline = chain(
        parse_task.s(asdict(ctx)),
        embed_and_index_task.s(),
    )
    result = pipeline.apply_async()
    return {"job_id": result.id, "queue": parse_task.queue}
```

### 19.6 Worker Pool Sizing

```yaml
# docker-compose.yml worker services
services:
  worker-pdf:
    image: doc-intel-api
    command: celery -A workers.ingestion_worker worker
             --queues pdf,embed --concurrency 8 --loglevel info
    deploy:
      replicas: 2    # 2 × 8 concurrent = 16 parallel PDF jobs
    # No GPU needed — Docling is CPU only

  worker-video:
    image: doc-intel-api-gpu
    command: celery -A workers.ingestion_worker worker
             --queues video --concurrency 2 --loglevel info
    deploy:
      replicas: 1    # 1 GPU node; concurrency=2 (Whisper + CLIP pipeline)
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]

  worker-web:
    image: doc-intel-api
    command: celery -A workers.ingestion_worker worker
             --queues web --concurrency 16 --loglevel info
    deploy:
      replicas: 2    # High-concurrency I/O; Crawl4AI is mostly network-bound
    # No GPU, low CPU — concurrency can be very high
```

### 19.7 RAG-Anything Integration Point

RAG-Anything is invoked narrowly — only for image captioning (the one task where its multimodal orchestration adds value over a direct LLaVA call). It is not the entry point for the pipeline.

```python
# Within parse_document_task, after Docling extracts embedded images:
for image_block in doc.image_blocks:
    caption = await rag_anything.caption_image(
        image_bytes=image_block.data,
        context=image_block.surrounding_text,   # caption with document context
    )
    ctx.raw_chunks.append(RawChunk(
        text=caption,
        metadata={"content_type": "image", "page": image_block.page,
                  "bbox": image_block.bbox}
    ))
```

This keeps RAG-Anything's role narrow and replaceable — swapping to a different vision model only changes this one call site.

---

## 20. Cost and Scale Controls

Enterprise deployments fail not on launch day but at month 3, when a single department has ingested 200GB of video, a team of 40 is hammering the search endpoint, and the embedding bill arrives. These controls must be designed in from the start.

### 21.1 PostgreSQL Schema Additions

```sql
-- Per-department resource quotas
CREATE TABLE department_quotas (
    dept_id           UUID PRIMARY KEY REFERENCES departments(id),
    max_storage_gb    NUMERIC(10,2) DEFAULT 50.0,
    max_docs          INT DEFAULT 10000,
    max_ingestion_rpm INT DEFAULT 10,     -- ingestion requests per minute
    max_search_rpm    INT DEFAULT 120,    -- search requests per minute
    max_video_hours   NUMERIC(10,2) DEFAULT 20.0,  -- total video duration
    embedding_budget_usd NUMERIC(10,4) DEFAULT 100.0  -- monthly spend cap
);

-- Running usage counters (updated by workers)
CREATE TABLE department_usage (
    dept_id           UUID PRIMARY KEY REFERENCES departments(id),
    storage_gb_used   NUMERIC(10,2) DEFAULT 0,
    doc_count         INT DEFAULT 0,
    video_hours_used  NUMERIC(10,2) DEFAULT 0,
    embedding_cost_usd NUMERIC(10,4) DEFAULT 0,
    period_start      TIMESTAMPTZ DEFAULT date_trunc('month', NOW())
);
```

### 21.2 Rate Limiting Middleware (FastAPI + Redis)

```python
from fastapi import Request, HTTPException
import redis.asyncio as aioredis
import time

class DeptRateLimiter:
    """
    Sliding window rate limiter per department, stored in Redis.
    Separate limits for ingestion (expensive) and search (cheap).
    """
    def __init__(self, redis: aioredis.Redis):
        self.redis = redis

    async def check(self, dept_id: str, action: str, limit: int, window_sec: int = 60):
        key = f"ratelimit:{dept_id}:{action}"
        now = time.time()
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window_sec)  # evict old entries
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window_sec)
        _, _, count, _ = await pipe.execute()
        if count > limit:
            raise HTTPException(429, f"Rate limit exceeded for {action}. "
                                     f"Max {limit} requests per {window_sec}s.")

# Inject into search and ingest endpoints
rate_limiter = DeptRateLimiter(redis_client)

@router.post("/dept/{dept_id}/ingest")
async def ingest(dept_id: UUID, ...):
    await rate_limiter.check(str(dept_id), "ingest", limit=10, window_sec=60)
    ...

@router.post("/dept/{dept_id}/chat/stream")
async def chat(dept_id: UUID, ...):
    await rate_limiter.check(str(dept_id), "search", limit=120, window_sec=60)
    ...
```

### 21.3 Ingestion Quota Enforcement

```python
async def check_ingestion_quota(dept_id: str, file_size_gb: float,
                                doc_type: str, db: AsyncSession):
    quota, usage = await db.execute(
        select(DepartmentQuota, DepartmentUsage)
        .where(DepartmentQuota.dept_id == dept_id)
        .where(DepartmentUsage.dept_id == dept_id)
    ).one()

    if usage.storage_gb_used + file_size_gb > quota.max_storage_gb:
        raise HTTPException(402, "Storage quota exceeded. "
                                 f"Used {usage.storage_gb_used:.1f}GB / "
                                 f"{quota.max_storage_gb:.0f}GB limit.")
    if usage.doc_count >= quota.max_docs:
        raise HTTPException(402, f"Document quota reached: {quota.max_docs} docs.")
    if doc_type == "video":
        video_hours = estimate_video_hours(file_size_gb)
        if usage.video_hours_used + video_hours > quota.max_video_hours:
            raise HTTPException(402, "Video processing quota exceeded.")
```

### 21.4 Embedding Cache (Redis)

Re-embedding identical text on every re-index or query variant wastes both latency and money. Cache dense embeddings keyed by a hash of the normalized text.

```python
import hashlib, json
import redis.asyncio as aioredis
import numpy as np

class EmbeddingCache:
    """
    Redis cache for dense embeddings.
    Key: sha256(normalized_text) → value: serialized float32 array.
    TTL: 7 days (embeddings don't change unless model changes).
    """
    def __init__(self, redis: aioredis.Redis, model_version: str = "bge-m3-v1"):
        self.redis = redis
        self.model_version = model_version  # include in key to bust on upgrade

    def _key(self, text: str) -> str:
        normalized = " ".join(text.lower().split())
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"emb:{self.model_version}:{digest}"

    async def get(self, text: str) -> np.ndarray | None:
        raw = await self.redis.get(self._key(text))
        return np.frombuffer(raw, dtype=np.float32) if raw else None

    async def set(self, text: str, vec: np.ndarray, ttl: int = 604800):
        await self.redis.setex(self._key(text), ttl, vec.astype(np.float32).tobytes())

    async def embed_with_cache(self, texts: list[str],
                               embedder: DenseEmbedder) -> list[np.ndarray]:
        vecs, misses, miss_idx = [], [], []
        for i, t in enumerate(texts):
            cached = await self.get(t)
            if cached is not None:
                vecs.append(cached)
            else:
                vecs.append(None)
                misses.append(t)
                miss_idx.append(i)
        if misses:
            fresh = embedder.embed_batch(misses)
            for i, (idx, vec) in enumerate(zip(miss_idx, fresh)):
                vecs[idx] = vec
                await self.set(misses[i], vec)
        return vecs
```

### 21.5 Cost Estimation by Operation

| Operation | Model / Service | Unit cost (approx.) | Per 1,000 chunks |
|---|---|---|---|
| Dense embedding | BGE-M3 (self-hosted) | ~$0.000 | $0 |
| Dense embedding | OpenAI text-embedding-3-large | $0.13 / 1M tokens | ~$0.07 |
| Sparse embedding | SPLADE (self-hosted) | ~$0.000 | $0 |
| Frame captioning | GPT-4o Vision | $0.00765 / image | $7.65 per 1k frames |
| Entity extraction | GPT-4o-mini | $0.15 / 1M tokens | ~$0.08 |
| Whisper transcription | OpenAI API | $0.006 / min audio | depends on duration |
| Qdrant (cloud) | 1M vectors, 1024-dim | ~$70/month | — |
| Neo4j (AuraDB) | 1GB graph | $65/month | — |

> Self-hosting BGE-M3 on a single A10G GPU eliminates the largest recurring cost. The embedding cache (§21.4) reduces re-embedding calls by 60–80% for stable corpora.

---

## 21. Evaluation Framework

A RAG system without ongoing evaluation degrades silently. New documents change the retrieval distribution; model updates shift embedding geometry; edge-case queries accumulate. Evaluation must be a continuous pipeline, not a one-time benchmark.

### 22.1 Evaluation Stack

| Layer | Tool | What it measures |
|---|---|---|
| Retrieval quality | RAGAS | Context recall, context precision, answer faithfulness, answer relevance |
| Answer quality | DeepEval | Hallucination detection, G-Eval scoring, custom rubrics |
| Adversarial robustness | ARES | Adversarial query generation + automated relevance scoring |
| Regression suite | Custom pytest | Per-department golden Q&A pairs run on every deploy |
| Latency benchmarks | Locust | p50/p95/p99 first-token latency under concurrent load |

### 22.2 RAGAS Integration

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

async def run_ragas_eval(
    dept_id: str,
    golden_dataset: list[dict],   # [{question, ground_truth, contexts, answer}]
) -> dict:
    """
    golden_dataset is pre-built per department during onboarding.
    Run after every significant ingestion batch or weekly as a cron job.
    """
    ds = Dataset.from_list(golden_dataset)
    result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ragas_llm,               # judge LLM (GPT-4o or Claude)
        embeddings=ragas_embeddings, # BGE-M3 for consistency with retrieval
    )
    scores = result.to_pandas().mean().to_dict()

    # Persist scores to Postgres for trend tracking
    await db.execute(insert(EvalRun).values(
        dept_id=dept_id,
        faithfulness=scores["faithfulness"],
        answer_relevancy=scores["answer_relevancy"],
        context_recall=scores["context_recall"],
        context_precision=scores["context_precision"],
        run_at=datetime.utcnow(),
    ))
    return scores
```

### 22.3 DeepEval Hallucination Detection

```python
from deepeval import evaluate as deval
from deepeval.metrics import HallucinationMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def build_test_cases(rag_outputs: list[dict]) -> list[LLMTestCase]:
    return [
        LLMTestCase(
            input=o["question"],
            actual_output=o["answer"],
            retrieval_context=o["contexts"],  # the chunks passed to the LLM
        )
        for o in rag_outputs
    ]

hallucination = HallucinationMetric(threshold=0.1)  # fail if >10% hallucinated
faithfulness  = FaithfulnessMetric(threshold=0.85)

# Run as part of CI/CD gate — block deploy if scores regress
deval(test_cases=build_test_cases(sample_outputs),
      metrics=[hallucination, faithfulness])
```

### 22.4 Golden Dataset Construction (Per Department)

```python
async def build_golden_dataset(dept_id: str, n_questions: int = 100) -> list[dict]:
    """
    Auto-generate golden Q&A pairs from the department's actual documents.
    Human-reviewed subset (~20%) validated before use as eval ground truth.
    """
    sample_chunks = await qdrant.sample(dept_id, n=n_questions)
    golden = []
    for chunk in sample_chunks:
        # Generate a question the chunk should answer
        q = await llm(f"Generate one specific factual question answered by:\n{chunk.text}")
        # Generate a reference answer grounded in the chunk
        a = await llm(f"Answer this question using only: {chunk.text}\nQuestion: {q}")
        golden.append({
            "question":     q,
            "ground_truth": a,
            "contexts":     [chunk.text],
            "doc_source":   chunk.metadata["filename"],
        })
    return golden
```

### 22.5 Evaluation Thresholds (Deployment Gates)

| Metric | Warning | Block deploy |
|---|---|---|
| Context recall@10 | < 0.82 | < 0.75 |
| Context precision | < 0.75 | < 0.65 |
| Answer faithfulness | < 0.88 | < 0.80 |
| Answer relevancy | < 0.82 | < 0.75 |
| Hallucination rate | > 0.08 | > 0.15 |
| p95 first-token latency | > 4s | > 8s |

---

## 22. Frontend SSE Contract

This section is the authoritative spec for frontend developers. Every event type, field, and ordering guarantee is defined here. The backend MUST NOT emit events outside this schema without a versioned contract update.

### 23.1 Connection and Protocol

```
POST /dept/{dept_id}/chat/stream
Content-Type: application/json
Authorization: Bearer {jwt}

Body: {"query": "...", "session_id": "sess_abc123", "stream_version": "1"}

Response: text/event-stream (SSE)
```

The `stream_version` field lets the frontend detect schema mismatches and render a graceful fallback.

### 23.2 Event Type Reference

Every SSE message has the form:
```
event: {event_type}
data: {json_payload}

```
(blank line terminates each event, per SSE spec)

#### `stream_start`
Emitted once, first, before any blocks. Contains the routing tier and plan so the UI can show a progress indicator.

```json
{
  "session_id":    "sess_abc123",
  "query_tier":    "deep",
  "stream_version": "1",
  "plan": [
    {"block_id": "tbl_01", "type": "table",      "title": "Revenue comparison"},
    {"block_id": "txt_01", "type": "text",        "heading": "Key drivers"},
    {"block_id": "cit_01", "type": "citation_strip"}
  ],
  "sub_questions": [
    "What are our Q1 revenue figures by product?",
    "What are Q4 revenue figures by product?",
    "What are market benchmark figures?"
  ]
}
```

#### `block_start`
Emitted once per block, before any `block_chunk` for that block.

```json
{
  "block_id":  "tbl_01",
  "type":      "table",
  "title":     "Revenue comparison",
  "columns":   ["Product", "Q1", "Q4", "Market Q4", "Delta"]
}
```

Supported `type` values: `table` · `text` · `equation` · `comparison` · `image_ref` · `code` · `citation_strip`

#### `block_chunk`
The streaming payload for a block. Schema differs by `type`:

```json
// type: table — one row at a time
{"block_id": "tbl_01", "row": {"Product": "Widget A", "Q1": "$2.1M", "Q4": "$3.4M", "Market Q4": "$2.8M", "Delta": "+21%"}}

// type: text — token-by-token delta (matches OpenAI streaming convention)
{"block_id": "txt_01", "delta": "Widget A growth was driven primarily by..."}

// type: equation — full LaTeX string, emitted as one chunk (not streamed token by token)
{"block_id": "eq_01", "latex": "CAGR = \\left(\\frac{V_f}{V_i}\\right)^{1/t} - 1"}

// type: comparison — one field-value pair at a time
{"block_id": "cmp_01", "field": "Erasure window", "left": "30 days", "right": "90 days", "delta": "+60 days", "status": "risk"}

// type: image_ref — a single event (no further chunks)
{"block_id": "img_01", "s3_key": "dept_finance/docs/q4_chart.png", "caption": "Q4 revenue waterfall chart", "alt": "Bar chart showing..."}

// type: code — full code block, emitted as one chunk
{"block_id": "code_01", "language": "python", "code": "def compute_cagr(vf, vi, t):\n    return (vf/vi)**(1/t) - 1"}
```

#### `block_end`
Emitted once per block after all its chunks. Signals the frontend to finalize rendering (e.g., apply table sorting controls, render LaTeX).

```json
{"block_id": "tbl_01"}
```

#### `citation`
Emitted after all content blocks, one per cited source. Rendered as the citation strip at the bottom.

```json
{
  "number":       1,
  "filename":     "Q4_Board_Report_2024.pdf",
  "doc_type":     "pdf",
  "pages":        [4, 7, 12],
  "section":      "Revenue Analysis > EMEA",
  "content_type": "table",
  "bbox":         [88, 234, 520, 410],
  "timestamp":    null,
  "slide_number": null,
  "s3_key":       "dept_finance/docs/Q4_Board_Report_2024.pdf"
}
```

#### `stream_end`
Emitted once, last. Contains final quality signals.

```json
{
  "session_id":       "sess_abc123",
  "total_blocks":     3,
  "chunks_retrieved": 12,
  "graph_expanded":   true,
  "cache_hit":        false,
  "query_tier":       "deep",
  "latency_ms": {
    "retrieval":   843,
    "rerank":      241,
    "graph":       612,
    "llm_total":  3820
  }
}
```

#### `error`
Emitted instead of `stream_end` on failure. The frontend should display the `user_message` and log `detail`.

```json
{
  "code":         "RETRIEVAL_FAILED",
  "user_message": "Unable to search this department's documents. Please try again.",
  "detail":       "Qdrant connection timeout after 5000ms",
  "retryable":    true
}
```

### 23.3 Event Ordering Guarantee

```
stream_start
  (one or more):
    block_start
    block_chunk × N
    block_end
  (zero or more):
    citation × M
stream_end  |  error
```

The frontend MUST NOT assume block order matches the `plan` array — blocks may arrive out of order if different sections complete at different times. Use `block_id` to route each event to the correct UI component.

### 23.4 Frontend Pseudocode

```typescript
const source = new EventSource(`/dept/${deptId}/chat/stream`, {
  headers: { Authorization: `Bearer ${token}` }
});

const blocks: Map<string, BlockState> = new Map();

source.addEventListener('stream_start', (e) => {
  const payload = JSON.parse(e.data);
  payload.plan.forEach(p => blocks.set(p.block_id, { type: p.type, content: [] }));
  renderPlanSkeleton(payload.plan);   // show loading skeleton for each block
});

source.addEventListener('block_start', (e) => {
  const { block_id, type, title, columns } = JSON.parse(e.data);
  mountBlock(block_id, type, { title, columns });
});

source.addEventListener('block_chunk', (e) => {
  const payload = JSON.parse(e.data);
  appendToBlock(payload.block_id, payload);  // dispatch by block type
});

source.addEventListener('block_end', (e) => {
  const { block_id } = JSON.parse(e.data);
  finalizeBlock(block_id);   // enable sorting, render LaTeX, etc.
});

source.addEventListener('citation', (e) => {
  appendCitation(JSON.parse(e.data));
});

source.addEventListener('stream_end', (e) => {
  source.close();
  showLatencyDebug(JSON.parse(e.data));
});

source.addEventListener('error', (e) => {
  source.close();
  showError(JSON.parse(e.data));
});
```

---

## 23. Observability and Monitoring

A production RAG system has three distinct observability concerns that typical web-service monitoring misses: LLM trace-level debugging (why did the model cite the wrong source?), vector database performance (why did recall degrade after ingesting 10k new documents?), and cross-component latency attribution (is slowness in the embedder, Qdrant, or the LLM?). Each requires a different tool.

### 24.1 RAG Trace-Level Observability — LangFuse (Recommended)

LangFuse is the most operationally mature open-source option for RAG tracing. It captures the full chain: query → retrieval → rerank → LLM prompt → response, with scores attached at each step. LangSmith (LangChain's hosted offering) and Arize Phoenix are strong alternatives.

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST,      # self-host at https://langfuse.com/docs/deployment/self-host
)

@observe(name="rag_pipeline")
async def traced_rag_pipeline(query: str, dept_id: str):
    # Retrieval span
    with langfuse_context.update_current_observation(
        name="hybrid_retrieval",
        metadata={"dept_id": dept_id, "query_tier": "standard"}
    ):
        chunks = await hybrid_search(dept_id, query, k=10)

    # Reranking span
    with langfuse_context.update_current_observation(name="cross_encoder_rerank"):
        ranked = cross_encoder_rerank(query, chunks, top_n=6)

    # LLM generation span — automatically captures prompt + response
    with langfuse_context.update_current_observation(name="llm_generation"):
        response = await llm_stream(query, ranked)

    # Attach retrieval quality score
    langfuse_context.score_current_observation(
        name="context_relevance",
        value=compute_context_relevance(query, ranked),
    )
    return response

# Log user feedback (thumbs up/down) to close the eval loop
async def log_feedback(trace_id: str, score: float, comment: str = ""):
    langfuse.score(
        trace_id=trace_id,
        name="user_satisfaction",
        value=score,           # 1.0 = thumbs up, 0.0 = thumbs down
        comment=comment,
    )
```

LangFuse self-hosted deploys as a Docker Compose stack (Postgres + ClickHouse + Next.js) and stores all traces in your own infrastructure — required for enterprise deployments where query content cannot leave the premises.

### 24.2 Infrastructure Metrics — Prometheus + Grafana

```python
# FastAPI: instrument with prometheus-fastapi-instrumentator
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter, Gauge

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Custom RAG-specific metrics
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "End-to-end retrieval latency",
    ["dept_id", "query_tier"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
)
CACHE_HIT_RATE = Counter(
    "rag_cache_hits_total",
    "Semantic cache hits",
    ["dept_id"]
)
INGESTION_FAILURES = Counter(
    "rag_ingestion_failures_total",
    "Ingestion job failures by stage",
    ["dept_id", "stage", "doc_type"]
)
ACTIVE_STREAMS = Gauge(
    "rag_active_streams",
    "Currently active SSE streaming connections"
)
CHUNK_COUNT = Gauge(
    "rag_indexed_chunks_total",
    "Total indexed chunks per department",
    ["dept_id"]
)

# Use in pipeline
async def hybrid_search_instrumented(dept_id: str, query: str, tier: str, **kwargs):
    with RETRIEVAL_LATENCY.labels(dept_id=dept_id, query_tier=tier).time():
        return await hybrid_search(dept_id, query, **kwargs)
```

**Recommended Grafana dashboards:**

| Dashboard | Key panels |
|---|---|
| RAG health | p50/p95/p99 retrieval latency, cache hit rate, active streams, error rate |
| Ingestion pipeline | Jobs queued / in-progress / failed, latency by stage and doc type, quota utilization |
| Qdrant | Collection sizes, indexing throughput, search RPS, memory usage per collection |
| Neo4j | Query execution time, node/relationship counts, bolt connection pool usage |
| Cost | Embedding API calls/hour, LLM token spend, video processing GPU utilization |

Qdrant exposes a native Prometheus endpoint at `/metrics` — no exporter needed. Neo4j requires the `neo4j-prometheus` plugin (included in Enterprise; community plugin available for Community edition).

### 24.3 Audit Logging

Enterprise document systems handle sensitive data — HR records, M&A documents, legal communications. Every access must be logged with enough detail to answer: *who accessed what document, from which chunk, in which query, at what time, with what result.*

```python
# models/audit.py
from sqlalchemy import Column, Text, UUID, TIMESTAMPTZ, JSON
from sqlalchemy.orm import DeclarativeBase

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id            = Column(UUID, primary_key=True, default=uuid4)
    event_type    = Column(Text, nullable=False)   # QUERY | INGEST | EXPORT | DELETE | LOGIN
    user_id       = Column(UUID, nullable=False)
    dept_id       = Column(UUID, nullable=True)
    session_id    = Column(Text)
    query_text    = Column(Text)                   # hashed for PII-sensitive depts
    doc_ids       = Column(JSON)                   # documents surfaced in response
    chunk_ids     = Column(JSON)                   # specific chunks cited
    query_tier    = Column(Text)                   # fast | standard | deep
    ip_address    = Column(Text)
    user_agent    = Column(Text)
    response_ms   = Column(Integer)
    cache_hit     = Column(Boolean)
    error         = Column(Text)
    created_at    = Column(TIMESTAMPTZ, default=datetime.utcnow)
    # Immutable — never UPDATE, only INSERT
```

```python
# FastAPI middleware: log every request regardless of outcome
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    if request.url.path.startswith("/dept/"):
        await audit_log_service.write(AuditLog(
            event_type="QUERY" if "chat" in request.url.path else "SEARCH",
            user_id=request.state.user_id,
            dept_id=request.state.dept_id,
            session_id=request.headers.get("X-Session-ID"),
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent"),
            response_ms=elapsed_ms,
        ))
    return response
```

Audit logs are append-only (no UPDATE/DELETE on the table — enforced via Postgres row-level security). For regulated industries (HIPAA, SOC 2, GDPR), ship audit logs to an immutable store (AWS CloudTrail, Azure Monitor, or an append-only S3 bucket with object lock).

### 24.4 Tool Comparison

| Tool | Best for | Deployment | Cost |
|---|---|---|---|
| LangFuse | RAG trace debugging, prompt versioning, user feedback loop | Self-hosted or cloud | Free (self-hosted) |
| LangSmith | LangChain-native tracing, datasets, evals | Cloud only | Usage-based |
| Arize Phoenix | ML observability, embedding drift detection | Self-hosted or cloud | Free (self-hosted) |
| Prometheus | Infrastructure metrics, alerting | Self-hosted | Free |
| Grafana | Dashboard visualisation over Prometheus | Self-hosted | Free (OSS) |
| Grafana Loki | Log aggregation (replaces ELK for most use cases) | Self-hosted | Free (OSS) |

---

---

## 24. What Makes This System Elite

1. **Every content type has a purpose-built parser** — not a generic "extract text" approach
2. **Table chunking is dual-representation** — tables answer both "show me the table" AND "what is widget A's Q4 value"
3. **Hybrid search is server-side** — Qdrant RRF eliminates client-side complexity
4. **Graph RAG expands retrieval** — finds related knowledge the vector search misses
5. **Query decomposition** — complex queries become multiple targeted retrievals
6. **Streaming is structured** — tables, equations, citations render live, not after full generation
7. **Isolation is a scored decision, not a hardcoded assumption** — `HybridResolver` scores regulatory pressure, scale, cross-tenant query need, operational maturity, and data residency at provisioning time; standard departments share a company collection (logical, payload-filtered), sensitive departments (Legal/HR/Finance Audit/Executive) get dedicated collections (physical), regulated workloads (FedRAMP/ITAR/HIPAA) get isolated clusters; `BoundClient` makes dept_id filter injection structurally impossible to omit; migration paths between modes are zero-downtime
8. **RBAC is query-time enforced** — you cannot query a department collection without membership verification at the API level
9. **Metadata is a retrieval filter, not documentation** — every field pre-filters candidates before embedding comparison, cutting latency and irrelevant results simultaneously
10. **Citations are chunk-tracked, not LLM-hallucinated** — pre-numbered context with `CitationTracker` ensures every `[N]` maps to a real source with page, bbox, or timestamp for deep linking
11. **Latency adapts to query complexity** — three routing tiers (fast / standard / deep) with parallel `asyncio.gather()` retrieval; add semantic caching only after metrics show >15% repeated queries
12. **Video uses late-fusion named vectors** — CLIP visual embeddings and BGE-M3 text embeddings stored as separate Qdrant named vectors, preserving each modality's geometry for independent recall legs
13. **Ingestion uses resource-matched Celery queues, not a monolithic worker** — `queue=pdf` (CPU, Docling), `queue=video` (GPU, Whisper + CLIP), `queue=web` (I/O, Crawl4AI); all funnel into a shared `queue=embed` for Qdrant indexing; stage-level rollback on failure; RAG-Anything scoped narrowly to image captioning only
14. **Cost and scale controls built in** — per-department ingestion quotas, sliding-window rate limiters in Redis, and an embedding cache that cuts re-embedding calls by 60–80%; video and storage quotas enforced before a single byte is processed
15. **Continuous evaluation via RAGAS + DeepEval** — golden datasets auto-generated per department, hallucination detection as a CI/CD gate, deployment blocked when faithfulness drops below 0.80 or hallucination rate exceeds 15%
16. **Typed SSE contract with versioning** — every frontend event has a published JSON schema, ordering guarantees, and a `stream_version` field for graceful degradation
17. **Neo4j scales without per-dept databases** — single instance, dept-prefixed node labels, uniqueness constraints, optional APOC subgraph security or multi-database (Enterprise) for hard tenant boundaries; decoupled from Qdrant isolation mode
18. **Three-tier observability stack** — LangFuse traces every retrieval → rerank → LLM chain; Prometheus + Grafana cover infrastructure metrics; append-only audit logs for SOC 2 / HIPAA compliance
19. **Storage is a one-line config swap** — pick AWS S3 for cloud or MinIO for on-prem at deployment time; the `StorageClient` boto3 wrapper with `endpoint_url` is the only change required

This is production-grade, not a Jupyter notebook demo.