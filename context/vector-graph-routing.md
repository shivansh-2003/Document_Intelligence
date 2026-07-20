# Vector vs. Graph Routing — Decision

> Which parsed chunks go to Qdrant (dense+sparse), which go to the graph (Neo4j /
> LightRAG-style entity extraction), and why. Decided 2026-07-20, not yet implemented —
> see `status.md` for what's actually built. Chunk taxonomy referenced here
> (`Chunk.kind`, `ChunkMetadata.doc_type`) is defined in `pipelines/text_pipeline.py`.

## Decision

**Qdrant (dense + sparse) — every chunk, no exceptions.**

Text, table summaries, image descriptions, audio segments all get embedded and
indexed, both legs. This is the retrieval backbone — excluding a type breaks "find
this thing" for that type. Dense catches paraphrase/semantic queries, sparse catches
exact terms (a number in a table, a named entity, a filename). No content-type split
here; dense+sparse is a blanket policy, not a routing decision.

**Graph — two layers, decided independently:**

### Layer 1 — structural edges (free, no LLM call, build for everything)

`PART_OF` (chunk→doc), `NEXT_SEGMENT` (audio chunk ordering by `start_sec`), anchor
edges (table/image → nearest Title). Comes straight off metadata already on every
`Chunk` — `idx`, `doc_id`, `start_sec`/`end_sec` — so there's no reason to skip any
type here.

### Layer 2 — entity/relation extraction (costs one LLM call per chunk — route selectively)

| Chunk type | Extract entities/relations? | Why |
|---|---|---|
| Text (`kind="text"`, `doc_type` = pdf/docx/pptx/txt/md/html) | Yes, always | Highest entity/relationship density — names, orgs, dates, cross-references |
| Audio (`kind="text"`, `doc_type="audio"`) | Yes, always | Same reasoning, plus the temporal chain is a natural fit for graph traversal ("what was discussed 5 min before X") |
| Table (`kind="table"`) | No, by default | A table's relationships are already structural (row↔column) — vector+sparse exact-value lookup serves "what was Q3 revenue" far better than a graph traversal. Revisit only if a concrete cross-table entity-linking need shows up |
| Image (`kind="image"`) | Only if `ImageDescription.type` is `diagram` or `flowchart` | That classification is already produced at parse time by `image_pipeline.py` — free filter, no new work. Diagrams/flowcharts literally encode a graph (labeled nodes, flow edges); photos/screenshots/charts don't have that structure to extract |

## Net effect

Vector store is universal and unconditional. Graph gets cheap structural edges for
every chunk, but the expensive entity-extraction pass only runs where it pays for
itself: text, audio, and diagram/flowchart-type images.

## Open questions for later

- Whether audio's entity-extraction prompt needs different framing than document text
  (speaker turns, temporal references) — same LLM call, different prompt
- Whether the table/image skip rules need a manual override per-document (e.g. an org
  chart delivered as a table)
