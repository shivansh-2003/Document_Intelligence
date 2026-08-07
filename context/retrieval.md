# Retrieval Guide — Document Intelligence

> Companion to `context/backend.md` (Phases 1-5) and `context/status.md` (what's
> actually built). Backend.md ends at "chunks are searchable and access-controlled" —
> this guide is the build plan for everything between "chunk is searchable" and
> "a user asks a question and gets a cited, access-scoped answer." Scope here:
> `retrieval/`, `caching/`, `ingestion_versioning/`, `api/retrieval_router.py`,
> `services/retrieval_service.py`.

## 1. Where this sits today

Phase 4 (`backend.md`) is done: every chunk is embedded (dense + sparse) and
upserted into Qdrant, `dept_id`-filtered, via `vectorstore.vector_store.hybrid_search()`.
Nothing calls that function yet — no `/query` route, no reranking, no query
understanding, no generation. This is **Phase 6** (Phase 5 is Celery/workers,
still unbuilt and not a dependency of this phase — retrieval is synchronous
request/response, it doesn't need a queue).

**Build this in order** — each step is independently testable, same convention
as `backend.md`:

```
Step 1: schema + config          → new columns/env vars exist, nothing uses them yet
Step 2: retrieval/ core pipeline → POST /query works end to end, no caching/memory
Step 3: conversation_slots       → multi-turn coherence
Step 4: caching/                 → 3-tier cache on top of a working pipeline
Step 5: ingestion_versioning/    → corpus versioning + supersession, ingest-side
```

## 2. Request lifecycle

```
query text ──> query_transform (skip-router or 1 structured LLM call)
                   │  QueryPlan { resolved_query, sub_queries[], hints }
                   ▼
             scope_resolver (dept_hint ∩ live memberships)
                   │  dept_ids[]
                   ▼
             [tier 2 retrieval_cache check — corpus_version-keyed]
                   │ miss
                   ▼
             fanout (asyncio.gather: sub_query × dept_id → hybrid_search)
                   │  [tier 1 embedding_cache feeds hybrid_search's vector= param]
                   ▼
             merge (two-level RRF, dedupe by chunk_id, per-dept quota)
                   ▼
             rerank (single cross-encoder pass vs resolved_query)
                   ▼
             recency (post-rerank score decay by indexed_at)
                   ▼
             confidence_gate (threshold check)
              │ below            │ above
              ▼                  ▼
        return "not          [tier 3 response_cache check — cosine match]
        confident" +               │ miss
        raw citations              ▼
                              generation (1 cited-answer LLM call)
                                   ▼
                          write tier 2 + tier 3 cache, update conversation_slots
                                   ▼
                              cited answer + citations
```

## 3. Query understanding

- **`query_transform.py`** — `QueryPlan` / `SubQuery` Pydantic models (same
  shape convention as `pipelines/text_pipeline.ChunkMetadata`/`Chunk`).
  A cheap skip-router (short, single-intent queries — no multi-part
  conjunctions, under a word-count threshold) builds a trivial one-`SubQuery`
  plan with no LLM call. Otherwise, one structured call through the existing
  `services.llm_service.generate(prompt, schema=QueryPlan)` does
  classify+decompose+resolve together — not three separate calls.
- **§3.3 `scope_resolver.py`** — `resolve_scope(db, user, dept_hint) -> list[dept_id]`.
  Fresh `DepartmentMembership` query per request — no caching of role/membership,
  same anti-staleness principle as `identity.rbac_service._check_access`. Any
  membership row already clears the viewer floor (`Role.VIEWER` is rank 0,
  nothing ranks below it). **Empty-scope policy**: zero memberships → `403`
  (same shape `require_dept_access` already uses). A `dept_hint` that doesn't
  intersect live memberships is *dropped*, not rejected — the hint comes from
  `query_transform`'s LLM call, it's a suggestion, not an access grant. It
  narrows within the real boundary (live memberships) and never expands past it.
- **§3.4 `conversation_slots.py`** — semantic-only slot store: `period`,
  `topic`, `doc_type`, and department **names** (never `dept_id`, never
  `role`). Keyed by `conversation_id` in Valkey, TTL'd — ephemeral session
  state, not a durable record, so no Postgres table/migration for it.
  `query_transform` may read a department *name* out of a slot as a hint for
  coreference resolution ("what about last quarter" → prior topic), but
  `scope_resolver` re-validates every hint against live memberships on every
  single call regardless. The slot store is never itself a source of truth
  for access.

## 4. Cross-department fanout & merge

A query can span every department the user belongs to — `dept_hint` narrows,
it doesn't restrict to one. That means results from different departments
(different corpora, different score distributions) have to be fused fairly:
**§4.2** quota-guaranteed merge ensures a department with any matches keeps at
least one slot in the merged top-K, so one department's score distribution
can't fully starve another's genuinely relevant chunk. See §5.3.

## 5. Per-scope retrieval mechanics

- **§5.1 `fanout.py`** — thin `asyncio.gather` over `(sub_query × dept_id)`
  pairs, each a direct call to the existing `vectorstore.vector_store.hybrid_search()`.
  No new search logic here, orchestration only.
- **§5.2 `rerank.py`** — single pass vs. `resolved_query` (not per-sub_query),
  using `fastembed`'s `TextCrossEncoder` (`BAAI/bge-reranker-base` — already
  installed, same vendor family as the dense embedder, no new dependency).
  Lazy singleton + `run_in_executor`, identical shape to
  `indexing.embedding_service`'s `_get_dense()`/`_get_sparse()` and the
  blocking-call reasoning already documented in `vector_store.upsert_chunks`.
- **§5.3 `merge.py`** — two-level RRF. Level 1 (dense+sparse fusion) already
  happens *inside* `hybrid_search()` via Qdrant's `FusionQuery`. Level 2 fuses
  across sub-query/department variants (`score = Σ 1/(k + rank)`, `k=60`),
  dedupes by `chunk_id` (the same chunk can surface from multiple variants),
  then applies the §4.2 quota pass before truncating to `RERANK_TOP_K` input size.
- **§5.4 `recency.py`** — post-rerank score decay:
  `adjusted = rerank_score * exp(-ln2 * age_days / RECENCY_HALF_LIFE_DAYS)`,
  reading the `indexed_at` Qdrant payload field (added in §6 of `backend.md`'s
  vectorstore section — see schema changes below).

## 6. Confidence gate & generation

- **§6.1 `confidence_gate.py`** — threshold check on the top adjusted score
  against `CONFIDENCE_THRESHOLD`. Gates two independent things: whether
  `generation.py` runs at all, and whether tier 2/3 caches get written. A
  low-confidence result is not cached and reused as if it were trustworthy.
- **§6.2 `generation.py`** — one cited-answer call through the existing
  `services.llm_service.generate()`. Prompt-template convention copied from
  `pipelines/table_pipeline.py` (module-level template constant +
  `build_*_prompt()` function): every chunk passed in carries its `chunk_id`
  and source (`filename`, `page_number`), the prompt requires inline citations
  referencing those.

## 7. Ingestion & versioning

- **§7.1 `corpus_version.py`** — `bump_corpus_version(db, dept_id)` via
  Postgres's own atomic increment (`UPDATE departments SET corpus_version =
  corpus_version + 1 WHERE id = :dept_id RETURNING corpus_version`), not
  hand-rolled locking. Called from `indexing.indexing_pipeline.index_chunks()`
  right after `upsert_chunks` succeeds. This single counter is the entire
  tier 2/3 cache invalidation mechanism (§8) — it's baked into the cache key,
  so a re-ingest naturally misses old cache entries with no separate
  cache-bust step.
- **§7.2 `supersession.py`** — nearest-neighbor candidate detection (dense
  similarity within the same `dept_id`) runs automatically right after ingest
  and surfaces `possible_duplicates` in the ingest response — **informational
  only**. Confirming supersession is a separate, explicit, editor-gated route
  (`Depends(require_dept_access(Role.EDITOR))`, same pattern as every other
  write route) that sets `Document.is_current=False` / `superseded_by` in
  Postgres *and* flips the `is_current` Qdrant payload field via `set_payload`
  (no full re-upsert). Never automatic — silently hiding a document from
  retrieval is a hard-to-reverse action, it needs an explicit confirming call.
- **§7.3 table sparse-encode fix** — `chunk_embed_text()` currently embeds the
  LLM table summary for *both* dense and sparse vectors. SPLADE (sparse) is a
  term-matching model; raw cell text ("$4.2M", "Q3") matches exact-value
  queries far better than prose does. Fix: derive cleaned cell text from
  `ChunkMetadata.text_as_html` (already carried on table chunks) and use it
  for the sparse encode specifically, while dense keeps using the summary.

## 8. Caching — 3 tiers on Valkey

No Redis/Celery infra exists in this repo yet (`status.md`); this introduces
Valkey (Redis-protocol-compatible) via a single lazy client
(`core/valkey_client.py`, same singleton shape as `vector_store.py`'s
module-level `_client`). All three tiers key off `caching/cache_keys.py`,
which — critically — bakes `corpus_version` into tier 2/3 keys (see §7.1).

- **Tier 1 `embedding_cache.py`** — get-or-compute around a query's dense+sparse
  vectors, feeding `hybrid_search()`'s new optional `vector=` param (§ schema
  changes below) so a cache hit skips the ONNX embed call entirely.
- **Tier 2 `retrieval_cache.py`** — get-or-compute around the
  fanout→merge→rerank→recency result list.
- **Tier 3 `response_cache.py`** — the one genuinely new mechanism: **cosine
  match against stored query embeddings**, not an exact key match. Valkey has
  no vector index, but the candidate set per department-scope is small (a
  cache, not a corpus) — `SCAN` the scope's key prefix and brute-force
  cosine-compare in Python against `RESPONSE_CACHE_SIM_THRESHOLD`. Reusing
  Qdrant for this would be reaching for a search index to solve a cache
  problem — rejected.

## 9. End-to-end wiring

- **`services/retrieval_service.py`** — orchestrator mirroring
  `services/parsing_service.py`'s role: one public `answer_query(db, user,
  query, conversation_id, dept_hint)` calling every module above in the §2
  order, including cache get-or-compute calls at the tier 1/2/3 seams and the
  confidence gate.
- **`api/retrieval_router.py`** — `POST /query`, `Depends(get_current_user)`
  only (no path-param `dept_id` — a query can span every department the user
  belongs to; `require_dept_access`/`require_dept_access_form` are single-dept
  and don't fit here).

## Schema changes this phase needs

- `models/department.py`: `corpus_version: int` (default `0`).
- `models/document.py`: `is_current: bool` (default `True`), `superseded_by: str | None`.
- No Alembic in this repo (documented divergence in `status.md`) — these need
  a hand-run `ALTER TABLE` against Neon, same as the earlier `Document.id` fix.
- `vectorstore/vector_store.py` payload (`_to_payload`): add `is_current`
  (default `True`) and `indexed_at` (ISO timestamp at upsert time) — recency
  and supersession filtering both need these in the Qdrant payload directly,
  not a Postgres join on the retrieval hot path (same denormalization
  philosophy already used for `dept_id`/`doc_id`/`doc_type`).
- `vectorstore/vector_store.py` `hybrid_search()`: new optional `vector: dict
  | None = None` param — when given, skip the internal `embed_batch` call.
  Backward compatible; this is the seam tier-1 caching plugs into.

## What's deliberately out of scope here

Celery/workers (`backend.md` Phase 5) — retrieval is synchronous
request/response and doesn't need a queue; this phase doesn't block on it.
`BoundClient`/`HybridResolver` (physical/isolated Qdrant collections) — still
logical-mode only, unaffected by this phase. LangFuse/Prometheus/RAGAS
observability — not touched here, same gap `status.md` already notes.
