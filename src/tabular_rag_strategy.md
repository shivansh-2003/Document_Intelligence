# Tabular Data RAG Strategy
## Tables · CSV · Excel

**Scope:** Parsing layer strategy for structured/tabular data in a multi-modal RAG pipeline.
**Decision:** No code yet — this is the architectural strategy before implementation.

---

## Why Tabular Data Is Different

Standard text RAG pipelines fail on structured data for three core reasons:

1. **Row fragmentation** — Generic chunkers split mid-record, destroying row integrity.
2. **Header loss** — Chunks get separated from column names; the LLM can no longer interpret values without knowing what column they belong to.
3. **Context blindness** — A single cell value like `"2.4M"` is meaningless without knowing it belongs to `Revenue → Q2 2023`.

The fundamental fix: **every chunk must be a self-describing unit** — it must carry enough context (headers, sheet name, table caption) to be understood in isolation.

---

## The Three-Representation Model

For each table, CSV, or Excel sheet, we produce **three parallel representations**. Each serves a different retrieval mode.

```
Raw Table
    │
    ├──► [1] Structural Repr   → SQL / pandas queries   (precision)
    ├──► [2] Semantic Repr     → Vector embedding        (similarity)
    └──► [3] Metadata Envelope → Routing + context       (both)
```

### 1. Structural Representation (Precision Path)
The original data is preserved in a queryable form — a pandas DataFrame, in-memory SQLite table, or serialized HTML table stored in vector metadata. This path handles: *"What was revenue in Q3 2022?"* — queries that need exact lookup, aggregation, or filtering.

### 2. Semantic Representation (Similarity Path)
An LLM-generated natural language summary of the table is embedded into the vector store. Example: *"This table contains quarterly revenue data for FY2020–2023 broken down by product line, with columns: Quarter, Product, Revenue (USD), Growth %."* This path handles discovery: *"Find me anything about product revenue trends."* The unstructured.io Table Description enrichment produces exactly this — it overwrites the text field with a summary while preserving the original structure.

### 3. Metadata Envelope (Routing Layer)
Every chunk carries metadata that enables the retrieval router to decide which path to use:
- `source_file`, `sheet_name`, `table_index`
- `column_names`, `row_count`, `data_types`
- `repr_type`: `"structural"` or `"semantic"`
- `date_range` (if detected), `numeric_columns`

---

## File-Type-Specific Parsing Strategy

### CSV Files
**Core problems:** delimiter variance, no schema, flat structure, large row counts.

**Approach:**
- Auto-detect delimiter (comma, semicolon, tab, pipe) — do not hardcode.
- Treat the first row as schema (column headers). Store headers in metadata.
- **Chunking unit:** Row groups, not arbitrary token windows. Default 10–50 rows per chunk depending on column count. Every chunk re-includes the header row.
- **No mid-row splits** under any circumstances.
- For small CSVs (< 200 rows): keep as one chunk + one semantic summary.
- For large CSVs: chunk by row groups + one global summary chunk that describes the full dataset.

**Two retrieval modes (based on query intent):**
- Lookup / filter / aggregate → route to structural (pandas query on original DataFrame)
- Discovery / context → route to vector search on semantic summary

### Excel Files (.xlsx / .xls)
**Core problems:** multi-sheet workbooks, merged cells, mixed data types, charts, named ranges.

**Approach:**
- Process **each sheet independently** — sheet name becomes a first-class metadata field.
- Per-sheet: detect if the sheet is data (tabular), narrative (mostly text), or mixed.
- Data sheets → apply the same row-group chunking as CSV.
- Narrative sheets → treat as text, route through standard text parser.
- **Do not flatten workbooks** into a single document — losing sheet structure destroys the semantic organization authors intended.
- For each sheet, produce: one semantic summary chunk + N structural row-group chunks.

### Tables Inside Documents (PDF / DOCX / HTML)
**Core problems:** tables extracted as raw cell strings, no spatial context, captions lost.

**Approach (via unstructured.io partitioning):**
- Use `partition_pdf(strategy="hi_res")` or `partition_docx()` — both return typed `Table` elements.
- Each `Table` element is processed separately from surrounding text.
- Apply LLM-based table description (unstructured.io Table Description enrichment) to generate the semantic summary.
- Store the original HTML/markdown rendering of the table in chunk metadata for structural access.
- TableChunks each receive their own summary — do not merge summaries across tables.

---

## Chunking Rules (Universal)

| Rule | Rationale |
|---|---|
| Never split mid-row | A partial row has no meaning |
| Always include column headers in every chunk | LLM needs schema to interpret values |
| Keep related rows together | Group by logical entity (same product, same date range) |
| Small tables (< 20 rows): one chunk | Splitting gains nothing |
| Large tables: chunk by logical groupings, not token count | Domain coherence > token budget |
| One semantic summary per table/sheet (not per chunk) | Summary describes the whole; chunks describe parts |

---

## Retrieval Architecture

Two retrieval paths, selected by a lightweight router based on query type:

```
User Query
    │
    ▼
[Query Classifier]
    │
    ├── Analytical / precise  ──► Structural path → pandas/SQL on original data
    │   ("what is...", "how many...", "which year...")
    │
    └── Exploratory / semantic ──► Vector path → similarity search on summaries
        ("find anything about...", "what does X say about...")
```

For complex queries that need both (e.g., *"summarize revenue trends and give me the exact Q3 figure"*), an **agent pattern** is used: the agent calls both paths as tools and synthesizes the response.

This is the LangChain map-reduce pattern in practice — each table gets independently queried, then results are combined in a final synthesis step.

---

## Key Decisions

| Decision | Choice | Why |
|---|---|---|
| Chunking unit for CSV | Row groups (not tokens) | Token-based chunking breaks rows |
| Delimiter detection | Auto-detect | CSV files are not always comma-separated |
| Excel multi-sheet | Per-sheet processing | Sheet structure is semantic, not just formatting |
| Table representation | Dual (structural + semantic) | No single repr handles both lookup and discovery |
| Table summarization | LLM-generated NL description | Raw cell strings are not embeddable meaningfully |
| Retrieval for analytics | SQL/pandas on original data | Vector search cannot do aggregation or exact math |
| Large table strategy | Chunk + global summary | Chunks for precision, summary for discovery |

---

## What NOT to Do

- **Do not pass raw CSV rows directly to a text embedder** — embedding `"Laptop,Electronics,999.99,15"` produces meaningless vectors.
- **Do not use a single flat text representation** — you lose the ability to do precise structured queries.
- **Do not apply generic token-based chunking** — it will split rows mid-record.
- **Do not flatten Excel workbooks** into one document — you lose sheet-level context.
- **Do not skip table description enrichment** — without it, table retrieval relies on matching query terms to raw cell values, which fails for analytical questions.

---

## Implementation Order (for Parsing Layer)

1. **CSV Parser** — delimiter detection, header extraction, row-group chunking, dual repr
2. **Excel Parser** — sheet enumeration, per-sheet classification, same dual repr
3. **Table Enricher** — LLM summary generation for embedded tables (PDF/DOCX tables via unstructured)
4. **Retrieval Router** — query classifier that selects structural vs semantic path
5. **Agent Wrapper** — for complex queries that need both paths

Steps 1–3 are the parsing layer (current focus). Steps 4–5 are retrieval layer (next phase).

---

## Sources
- [LangChain: Summarizing Excel with eparse + LLM](https://www.langchain.com/blog/summarizing-and-querying-data-from-excel-spreadsheets-using-eparse-and-a-large-language-model)
- [Unstructured.io: Table Description Enrichment](https://docs.unstructured.io/concepts/enriching/table-descriptions)
- [Open WebUI Discussion: Tabular RAG Issues](https://github.com/open-webui/open-webui/discussions/22337)
- [YouTube: Master Series RAG #13 — CSV and Excel Parsing](https://www.youtube.com/watch?v=TdMFen8ieB8)
- [YouTube: RAG on Excel Using LlamaIndex](https://www.youtube.com/watch?v=Tlb6Lkc9A2A)
