# Pipelines Stage

Converts a `ParsedDocument` (from the parsing stage) into retrieval-ready chunks with rich metadata. Three self-contained pipeline modules handle the three content streams independently.

This is the second stage of the RAG pipeline. It sits between parsing and embedding.

```
ParsedDocument.texts  → text_chunker.py   → list[TextChunk]
ParsedDocument.tables → table_pipeline.py → list[TableChunk]
ParsedDocument.images → image_pipeline.py → list[ImageChunk]
                                                    ↓
                                           embedding → vector store
```

---

## Quick start

```python
import sys
sys.path.insert(0, "src")

from backend.parsing import PDFParser
from backend.pipelines import chunk_text, process_tables, process_images

doc_meta = {
    "doc_id":      "doc_abc123",
    "dept_id":     "dept_finance",
    "filename":    "Q4_Report.pdf",
    "upload_date": "2026-01-15",
    "created_by":  "user_001",
}

# Parse
doc = PDFParser(extract_images=True, strategy="hi_res").parse("Q4_Report.pdf")

# Chunk all three content streams
text_chunks  = chunk_text(doc.texts, source_type="pdf", doc_meta=doc_meta)
table_chunks = process_tables(doc.tables, source_type="pdf", doc_meta=doc_meta,
                               llm_client={"backend": "ollama", "model": "qwen3"})
image_chunks = process_images(doc.images, source_type="pdf", doc_meta=doc_meta,
                               vlm_backend="ollama",
                               vlm_config={"vision_model": "qwen2.5-vl", "text_model": "qwen3"})

# Each chunk is ready to embed
for c in text_chunks:
    print(c.chunk_id, c.metadata["chunk_strategy"], c.text[:80])
```

---

## `text_chunker.py`

### Strategy routing

The chunker detects structure automatically and routes to the best strategy:

| Source / detected structure | Strategy | Description |
|---|---|---|
| `source_type="pptx"` | **slide-level** | One chunk per slide text block |
| `source_type="audio"` or `"video"` | **sentence window** | 1 sentence embedded + ±3-sentence context |
| Text contains ≥ 2 markdown headers (`#`, `##`, `###`) | **markdown-header** | Split at headings first, recursively sub-split long sections |
| Everything else | **recursive** | `RecursiveCharacterTextSplitter` — 512 tokens, 64 overlap |

Token counting uses `tiktoken` (`cl100k_base`), matching the BGE-M3 embedding model's tokenizer.

### `TextChunk` fields

```python
@dataclass
class TextChunk:
    chunk_id:     str    # "chnk_a3f9b2xx"
    text:         str    # content to embed
    context_text: str    # extended context for sentence-window; else same as text
    metadata:     dict
```

### Metadata (all fields, zero LLM cost)

| Field | Type | Description |
|---|---|---|
| `chunk_id` | str | Unique chunk identifier |
| `doc_id` / `dept_id` | str | From `doc_meta` |
| `doc_type` | str | Source format ("pdf", "url", etc.) |
| `content_type` | `"text"` | Always "text" for this pipeline |
| `chunk_strategy` | str | Which strategy was used |
| `headings_path` | list[str] | Heading breadcrumb, e.g. `["Q4 Report", "Revenue", "EMEA"]` |
| `hierarchy_level` | int | Depth of `headings_path` |
| `parent_section` | str | Last heading in the path |
| `page_number` | int\|None | Source page (if available) |
| `slide_number` | int\|None | Source slide (PPTX) |
| `language` | str | Detected language (`langdetect`), fallback `"en"` |
| `has_numbers` | bool | Whether chunk contains digits |
| `has_citations` | bool | Whether chunk contains `[1]`-style or `(2024)`-style refs |
| `semantic_density` | float | `unique_meaningful_tokens / total_tokens` (0–1) |
| `token_count` | int | Token count via `tiktoken` |
| `recency_score` | float | `exp(-days_since_upload / 180)` — decays over 6 months |
| `filename` / `upload_date` / `created_by` | str | From `doc_meta` |

### API

```python
chunk_text(
    texts: list[str],    # from ParsedDocument.texts
    source_type: str,    # "pdf" | "docx" | "txt" | "pptx" | "url" | "audio"
    doc_meta: dict,
) -> list[TextChunk]
```

---

## `table_pipeline.py`

### Dual-representation chunking

Every table produces two types of chunks. Both types contain the LLM summary.

```
Table (N rows)
  │
  ├── full_table chunk   — entire table markdown + LLM summary
  │                        best for: comparison queries, aggregate questions
  │
  └── row_group chunks   — 10-row windows with 2-row overlap (only if N > 10)
                           best for: specific value lookup, exact cell retrieval
```

### `TableChunk` fields

```python
@dataclass
class TableChunk:
    chunk_id:    str
    text:        str    # markdown table (full or row-group)
    chunk_type:  str    # "full_table" | "row_group"
    llm_summary: str    # LLM-generated summary ("" if skipped)
    metadata:    dict
```

### Metadata

| Field | Description |
|---|---|
| `content_type` | `"table"` |
| `chunk_type` | `"full_table"` or `"row_group"` |
| `table_index` | Which table in the document (0-indexed) |
| `row_count` / `col_count` | Table dimensions |
| `has_header` | Whether a header row was detected |
| `header_names` | List of column header strings |
| `column_types` | `"numeric"` or `"text"` per column |
| `row_start` / `row_end` | Row range for row_group chunks |
| `has_numbers` | Whether table contains numeric data |
| `llm_summary` | Table description from LLM |

### LLM summary backend

```python
# Skip LLM summary (fastest)
process_tables(tables, source_type, doc_meta, llm_client=None)

# Ollama with qwen3
process_tables(tables, source_type, doc_meta,
               llm_client={"backend": "ollama", "model": "qwen3"})

# Or shorthand for Ollama default
process_tables(tables, source_type, doc_meta, llm_client="ollama")

# Custom callable
process_tables(tables, source_type, doc_meta, llm_client=my_llm_fn)
```

### API

```python
process_tables(
    tables: list[str],   # from ParsedDocument.tables (markdown strings)
    source_type: str,
    doc_meta: dict,
    llm_client=None,     # see above
) -> list[TableChunk]
```

---

## `image_pipeline.py`

### Two-pass VLM analysis

Each image goes through two Ollama calls:

```
Image
  │
  ├── Pass 1: Classification
  │     prompt → "classify: chart | diagram | table | photo | equation | general"
  │     model  → qwen2.5-vl (vision) if image bytes available
  │              qwen3 (text) as fallback using OCR text
  │
  └── Pass 2: Deep analysis (type-tailored prompt)
        chart:   axes, legend, all data points, trend direction
        diagram: all nodes, all connections, process flow
        table:   all column headers, all visible values, units
        photo:   all objects, all visible text, environment
        equation: full transcription in LaTeX, variable meanings
        general: exhaustive description of all visible content
```

### Structured metadata by image type

Chart images get `chart_type`, `axis_labels`, `key_values`, `trend`.
Diagram images get `entities_mentioned`.
All images get `image_type`, `description`, `ocr_text`, `vlm_model`.

### `ImageChunk` fields

```python
@dataclass
class ImageChunk:
    chunk_id:   str
    text:       str    # VLM description — the embeddable content
    image_type: str    # chart | diagram | table | photo | equation | general
    ocr_text:   str    # raw OCR / unstructured text from the source image
    metadata:   dict
```

### Metadata

| Field | Description |
|---|---|
| `content_type` | `"image"` |
| `image_type` | Classified type |
| `image_index` | Position in document (0-indexed) |
| `vlm_backend` | Which backend was used (`"ollama"`) |
| `vlm_model` | Exact model name used |
| `has_image_bytes` | Whether actual image bytes were available |
| `description` | Full VLM-generated description |
| `ocr_text` | Raw OCR text from parsing stage |
| `chart_type` | (charts only) bar / line / pie / scatter / etc. |
| `axis_labels` | (charts only) `{"x": "Year", "y": "Revenue"}` |
| `key_values` | (charts only) prominent numbers extracted |
| `trend` | (charts only) increasing / decreasing / stable |
| `entities_mentioned` | (diagrams only) capitalized component names |
| `page_number` / `slide_number` | Source location |
| `image_path` | Path to extracted image file (if available) |

### API

```python
process_images(
    images: list[dict],       # from ParsedDocument.images
    source_type: str,
    doc_meta: dict,
    vlm_backend: str = "ollama",
    vlm_config: dict = None,  # {"vision_model": "qwen2.5-vl", "text_model": "qwen3"}
) -> list[ImageChunk]
```

### Image dict format (from ParsedDocument)

```python
{
    "metadata": {
        "page_number":   1,           # source page
        "image_path":    "/tmp/...",  # path to extracted image file (if present)
        "image_base64":  "...",       # base64 bytes (if present)
        "coordinates":   {...},       # bounding box from unstructured
    },
    "text": "some OCR text"           # from unstructured extraction
}
```

If neither `image_path` nor `image_base64` is present, the pipeline falls back to text-only analysis using the `text_model` on the OCR text.

---

## `doc_meta` reference

All three pipelines accept the same `doc_meta` dict:

```python
doc_meta = {
    "doc_id":      str,   # unique document identifier
    "dept_id":     str,   # department scope (used for Qdrant/Neo4j filtering)
    "filename":    str,   # original filename
    "upload_date": str,   # ISO date string, e.g. "2026-01-15"
    "created_by":  str,   # user ID who uploaded the document
}
```

---

## Dependencies

All available in the project venv:

```
langchain-text-splitters   # MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
tiktoken                   # token counting
langdetect                 # language detection
nltk                       # sentence tokenization (sentence-window strategy)
ollama                     # Ollama Python client (table + image LLM calls)
pillow                     # PIL image loading
```

Ollama models needed (pull once):

```bash
ollama pull qwen3          # table summarization (text model)
ollama pull qwen2.5-vl     # image analysis (vision model)
```
