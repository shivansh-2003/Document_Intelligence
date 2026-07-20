# Graph Builder Guide — Document Intelligence

> Companion to `context/vector-graph-routing.md` (what goes to the graph and why) and
> `backend-development-guide.md` (auth, indexing, vector store). This guide covers
> `graph/` only — building the entity/relation knowledge graph on top of chunks that
> are already parsed, tagged, and embedded.

## 1. What this module is for, in one sentence

The `graph/` module turns chunks into a queryable web of entities and relationships,
scoped so no department can see another's — it exists to answer "what else is
connected to this?", which vector search alone can't answer.

**Do not start this module before `identity/` and `vectorstore/` are real.** The
isolation pattern here directly mirrors `vectorstore/bound_client.py`, and copying a
pattern that doesn't exist yet produces a worse version of it. Build order: backend
guide phases 1–4, then this module.

## 2. Framework and database decision (recap)

- **LightRAG**, not Graphiti. LightRAG is built for batch corpus ingestion with
  incremental appends — new knowledge unions into the existing graph rather than
  triggering a full rebuild — which matches continuous multi-user document upload.
  Graphiti's core value (bi-temporal fact validity, "this preference changed") is
  built for conversational agent memory, not a static document corpus. Reconsider
  Graphiti only if a per-user chat-memory layer gets added later — it can coexist
  with LightRAG rather than replace it.
- **Neo4j**, not FalkorDB or Memgraph. Neo4j is LightRAG's first-class, mature
  storage backend. FalkorDB is Graphiti's ecosystem default — pairing it with
  LightRAG means maintaining a non-standard adapter. Memgraph's in-memory speed
  solves a latency problem this system doesn't have (LLM calls dominate latency, not
  graph traversal).
- **Single Neo4j database, label-scoped isolation** — not one database per
  department. Database-per-tenant is operationally unmanageable past ~50
  departments (each database runs its own memory pools, background threads,
  transaction logs). Isolation is enforced by a `Dept_{dept_id}` label prefix plus a
  bound client, exactly like the Qdrant `BoundClient` pattern.

## 3. Target folder layout

```
src/backend/graph/
├── graph_store.py             # Neo4j driver/session mgmt — no isolation logic
├── bound_graph_client.py      # label-injection isolation, every write/read goes through here
├── structural_edges.py        # PART_OF, NEXT_SEGMENT, anchors — Layer 1, no LLM, always
├── entity_extraction.py       # per-dept LightRAG instance — Layer 2, one LLM call per chunk
├── entity_resolution.py       # dedupe/merge check before a new Entity node is created
└── ontology.py                # per-dept entity-type hints fed into extraction prompts
```

**File responsibilities, in pipeline order:**

| File | Job |
|---|---|
| `graph_store.py` | Raw connection to Neo4j — driver, sessions, retries. Nothing else talks to Neo4j directly. |
| `bound_graph_client.py` | Injects `Dept_{dept_id}` into every write and read; refuses unlabeled queries outright. |
| `structural_edges.py` | Free, always-on graph layer — built straight from metadata already on the chunk. |
| `entity_extraction.py` | Expensive, selective layer — LLM call per eligible chunk, builds the per-dept `LightRAG` instance. |
| `entity_resolution.py` | Runs before extraction output is written — prevents duplicate entity nodes across uploaders. |
| `ontology.py` | Pure config — per-dept expected entity types, reduces extraction noise for free. |

Naming note: this replaces earlier drafts named `neo4j_client.py` and
`lightrag_factory.py`. Both named the file after the vendor/library instead of the
role it plays — if LightRAG or Neo4j is ever swapped, `entity_extraction.py` and
`graph_store.py` shouldn't need to change name or import path.

## 4. The two-layer decision (from `vector-graph-routing.md`)

**Layer 1 — structural edges. Build for every chunk, no exceptions.**

`PART_OF` (chunk → doc), `NEXT_SEGMENT` (audio chunk ordering by `start_sec`), anchor
edges (table/image → nearest Title). This comes straight off metadata already present
on every `Chunk` (`idx`, `doc_id`, `start_sec`/`end_sec`) — zero LLM calls, so there's
no reason to gate it or queue it separately from the main ingestion path.

**Layer 2 — entity/relation extraction. Costs one LLM call per chunk — route
selectively.**

| Chunk type | Extract? | Why |
|---|---|---|
| Text | Always | Highest entity/relationship density |
| Audio | Always | Same reasoning, plus temporal chain fits graph traversal |
| Table | No (default) | Row↔column relationships are already structural; vector+sparse serves exact-value lookup better |
| Image | Only if `type` is `diagram`/`flowchart` | That classification is free — already produced by `image_pipeline.py` at parse time |

`indexing/routing_service.py` (backend guide, Phase 4) makes this decision as a pure
function. `graph/` consumes the decision, it doesn't make it — keep that boundary
clean so the routing rule lives in exactly one place.

## 5. Per-chunk pipeline

```
Chunk ingested (dept_id, doc_id tagged)
        │
        ▼
Structural edges built (always — PART_OF, NEXT_SEGMENT, anchors)
        │
        ▼
Extraction eligible? ──No──┐
        │ Yes              │
        ▼                  │
Extract & resolve entities │
(LLM call, dedupe vs       │
 dept subgraph)            │
        │                  │
        ▼                  ▼
        Write to Neo4j (dept-labeled subgraph)
```

**Step-by-step:**

1. **Chunk arrives** already tagged with `dept_id`/`doc_id` — nothing new to derive.
2. **Structural edges write synchronously**, in the same path as the vectorstore
   write. No queue needed — this is cheap.
3. **Eligibility check** runs `routing_service`'s decision (§4).
4. **If eligible**, the chunk goes onto a dedicated queue lane
   (`workers/graph_extract_worker.py`, `queue=graph_extract`) — **not** the same
   queue as embedding. Entity extraction is an LLM call; embedding is comparatively
   fast. Sharing a queue means a document burst starves the fast path. Set
   concurrency (`LIGHTRAG_MAX_ASYNC`) per department, not globally, so one
   department's ingestion spike can't starve another's extraction.
5. **Extraction runs** with a dept-scoped ontology hint from `ontology.py` — same LLM
   call, less entity-type noise, for free.
6. **Resolution runs before write** — `entity_resolution.py` checks the dept subgraph
   for a near-match (normalized name, alias table, or embedding similarity on entity
   descriptions) before creating a new `Entity` node. This is what stops "Acme Corp"
   and "Acme Corporation," uploaded by two different people, from forking into two
   nodes. **This is the step most systems skip under deadline pressure — it's the one
   that determines whether the graph is still useful at 500k chunks or has degenerated
   into duplicate-entity noise.**
7. **Write happens through `bound_graph_client.py`** — label injected automatically,
   cross-dept queries structurally refused.
8. **Idempotency and rollback mirror the vectorstore pattern**: key extraction
   results by `chunk_id` so a retried job doesn't duplicate edges; on ingestion
   failure, delete-by-`doc_id`.

## 6. Multi-tenant isolation model

**One Neo4j database. Label-scoped subgraphs per department. One `LightRAG` instance
per department**, each with its own working directory, its own concurrency budget, and
its own ontology hints — but all writing into the same database, separated only by
label prefix.

```
Neo4j — single database
├── Dept_finance:*   (entities, chunks, docs — fed by LightRAG "finance" instance)
└── Dept_legal:*     (entities, chunks, docs — fed by LightRAG "legal" instance)

All writes pass through bound_graph_client.py — label injected, cross-dept queries refused.
```

**Isolation is a boundary, identity is provenance — don't confuse the two:**

- **Department = isolation boundary, structurally enforced.** The label prefix plus
  `bound_graph_client.py` make cross-dept access impossible by construction, not by
  convention.
- **User = provenance, not isolation.** Don't shard the graph by user — that would
  destroy the value of a departmental graph (entities linking across a team's
  documents is the point). Instead, carry `uploaded_by`/`uploaded_at` as properties
  on `Document`/`Chunk` nodes only. Entity nodes stay user-agnostic: an entity like
  "GDPR" belongs to the department's knowledge, not to whoever's document mentioned
  it first. This still gives clean per-user audit and offboarding (delete a user's
  doc/chunk nodes, then garbage-collect entities with zero remaining `APPEARS_IN`
  edges) without fragmenting the knowledge itself.
- **Cross-department linking is a future, explicitly permissioned layer, not a
  default.** If Finance and Legal both mention "Acme Corp," that's two separate
  entity nodes in two separate subgraphs — correct behavior for isolation. If
  org-wide intelligence is needed later, add an admin-only `SAME_AS` cross-dept edge
  computed by a separate entity-resolution pass, rather than merging the graphs.

## 7. Two things to keep as periodic batch jobs, not per-chunk work

- **Entity resolution sweep** — catches merges the per-write check in step 6 missed.
  Runs on `workers/maintenance_worker.py` alongside the vectorstore's usage-counter
  and orphan-cleanup jobs.
- **Community/summary detection** — only needed if LightRAG's global-mode query
  (broad thematic retrieval across the whole subgraph) gets used later. Graph-wide,
  expensive, does not belong in the request-time path.

## 8. Build order

1. `graph_store.py` — confirm a raw Neo4j connection works, no isolation yet.
2. `bound_graph_client.py` — build the label-injection wrapper immediately after;
   don't let any other file in this module talk to `graph_store.py` directly.
3. `structural_edges.py` — the free layer, wire it into `indexing_pipeline.py`
   (backend guide) so it runs for every chunk from day one.
4. `ontology.py` — static config, low risk, unblocks step 5.
5. `entity_extraction.py` — the `LightRAG` factory, one instance per department.
6. `entity_resolution.py` — build this *before* wiring extraction output to writes,
   not after. Retrofitting dedup onto an already-duplicated graph is much harder
   than preventing duplicates from the start.
7. `workers/graph_extract_worker.py` — separate queue lane, per-dept concurrency cap.

## 9. Checklist before calling this module "built"

- [ ] Structural edges exist for 100% of ingested chunks, regardless of extraction
      eligibility
- [ ] A chunk from one department's LightRAG instance never appears in another
      department's Cypher query results, even with a hand-crafted unlabeled query
      (test this directly — don't just trust the wrapper)
- [ ] Re-running extraction on an already-processed `chunk_id` does not create
      duplicate edges
- [ ] Two documents from different uploaders mentioning the same real-world entity
      under slightly different names resolve to one node, not two
- [ ] `graph_extract` queue depth doesn't grow unbounded when the `embed` queue is
      idle — confirms the two lanes are actually independent
- [ ] Deleting a document removes its `Document`/`Chunk` nodes and any now-orphaned
      entities, without touching entities still referenced by other documents