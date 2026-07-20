# Backend Development Guide — Document Intelligence

> Companion to `context/architecture.md` (target blueprint) and `context/status.md`
> (what's actually built). This guide covers everything **except** the graph builder —
> see `graph-builder-guide.md` for that. Scope here: `core/`, `models/`, `identity/`,
> `api/`, `indexing/`, `vectorstore/`, `workers/`.

## 1. Where this sits today

Parsing is done: six document formats plus audio parse into a normalized `Chunk`
schema. Nothing past parsing exists yet — no auth, no vector store, no async queue.
This guide is the build plan for everything between "chunk exists" and "chunk is
searchable and access-controlled."

**Build this in order.** Each phase is a hard dependency for the next — don't start
`vectorstore/` before `identity/` is real, or you'll retrofit `dept_id` validation
into code that currently just trusts the caller.

```
Phase 1: models + core        → schema and infra exist, nothing uses them yet
Phase 2: identity              → login works, roles are checkable
Phase 3: wire identity into    → /parse stops trusting dept_id on faith
         existing routers
Phase 4: indexing + vectorstore→ chunks become searchable
Phase 5: workers               → ingestion stops blocking the request thread
```

## 2. Target folder layout (this guide's scope)

```
src/backend/
├── core/
│   ├── config.py                  # env vars: DB_*, JWT_*, QDRANT_*, CELERY_*, plus existing OLLAMA_*/WHISPER_*
│   ├── security.py                # JWT encode/decode, password hashing
│   ├── database.py                # async SQLAlchemy engine + session factory
│   └── logging.py                 # structured logging, request-id correlation
│
├── models/
│   ├── company.py
│   ├── user.py
│   ├── department.py              # dept_type, isolation_mode, regulatory_flags
│   ├── department_membership.py   # user × dept × role — the RBAC join table
│   ├── document.py
│   ├── ingestion_job.py
│   └── audit_log.py               # append-only, never UPDATE
│
├── identity/
│   ├── auth_service.py            # login, token issuance, password verification
│   ├── rbac_service.py            # role_rank(), require_dept_access() dependency
│   └── membership_service.py      # add/remove member, change role, list members
│
├── api/
│   ├── auth_router.py             # POST /auth/login, /auth/register
│   ├── company_router.py          # provision a company/tenant
│   ├── department_router.py       # create dept, set isolation_mode/dept_type
│   ├── membership_router.py       # add/remove members, assign roles
│   ├── parsing_router.py          # existing — gets require_dept_access() added
│   └── indexing_router.py         # ingestion status/trigger endpoints
│
├── indexing/
│   ├── embedding_service.py       # dense (BGE-M3) + sparse (SPLADE) — chunk → vectors
│   ├── routing_service.py         # implements vector-graph-routing.md L1/L2 eligibility
│   └── indexing_pipeline.py       # orchestrator: chunk → embed → vectorstore write → enqueue graph
│
├── vectorstore/
│   ├── vector_store.py            # Qdrant collection mgmt, hybrid upsert/query
│   ├── bound_client.py            # BoundClient / SimpleBoundClient — dept isolation
│   ├── resolver.py                # HybridResolver — decides isolation_mode at provisioning
│   └── schema.py                  # payload schema, named vectors, RRF fusion config
│
└── workers/
    ├── celery_app.py
    ├── embed_worker.py            # queue=embed
    └── maintenance_worker.py      # periodic cleanup (usage counters, orphans)
```

Naming rule carried over from the existing codebase: **name files by what they do, not
by which library does it.** `vector_store.py`, not `qdrant_client.py`. If the vendor
changes, the file name shouldn't have to.

## 3. Phase 1 — `models/` + `core/`

Get the schema real before writing any service around it. Use Alembic for migrations
from the start — retrofitting migrations onto a schema that's already in production is
much more painful than starting with them.

**Core tables and why each exists:**

| Table | Purpose |
|---|---|
| `companies` | top-level tenant boundary |
| `users` | one row per person, scoped to a company |
| `departments` | isolation boundary — `dept_type`, `isolation_mode`, `regulatory_flags` |
| `department_memberships` | the actual RBAC join: `user_id × dept_id → role` |
| `documents` | one row per uploaded file, `dept_id` + `uploaded_by` |
| `ingestion_jobs` | async job tracking once `workers/` exists |
| `audit_log` | append-only — every access-relevant action, never `UPDATE`, only `INSERT` |

**Decide `isolation_mode` schema now, even before the resolver logic exists.**
Retrofitting this field after Qdrant collections are already created per-department is
painful — collections would need renaming/migrating. `isolation_mode` is one of
`logical | physical | isolated`, set once at provisioning time by
`vectorstore/resolver.py`, and treated as fixed afterward — never recomputed
per-request. `dept_type` (standard/legal/hr/finance_audit/m_and_a/executive/compliance)
and `regulatory_flags` (e.g. `hipaa`, `fedramp`) are the inputs that decision reads.

`core/database.py` and `core/security.py` come next — connection pooling, JWT
primitives, password hashing. No business logic here, just infrastructure.

**Checklist before moving to Phase 2:**
- [ ] Alembic migrations run cleanly against a fresh Postgres instance
- [ ] `department_membership` has a composite primary key `(user_id, dept_id)` — one
      role per user per department, not a list
- [ ] `audit_log` has no `updated_at` column — if you're tempted to add one, that's a
      sign something is trying to mutate an audit row, which should never happen

## 4. Phase 2 — `identity/`

This is where auth becomes real, but stays deliberately dept-unaware at first — just
"can this person log in and get a token."

- **`auth_service.py`** — login, register, token issuance/verification. No RBAC logic
  here; this file doesn't know what a department is.
- **`rbac_service.py`** — `role_rank(role)` for comparing `admin > editor > viewer`,
  and `require_dept_access(dept_id, min_role)` as a FastAPI dependency that resolves
  the current user's membership and raises `403` if their role is insufficient. This
  is the single choke point every future dept-scoped route imports.
- **`membership_service.py`** — add/remove a member, change a role, list a
  department's members. This is what `api/membership_router.py` calls.

**A design decision to make explicitly here, not later:** does `dept_id` on an
uploaded chunk get validated against `department_memberships` at upload time, or only
enforced at query time? Right now `parsing_service._check_identity` only checks that
`dept_id` is *present*, not that the uploader actually belongs to that department.
Closing that gap is what Phase 3 is for — decide the answer now so Phase 3 isn't a
redesign.

**Checklist:**
- [ ] A user with no membership row for a department gets `403`, not a silent empty
      result
- [ ] `role_rank` comparison is `>=`, not `==` — an admin should pass a `viewer`
      check
- [ ] Tokens carry `user_id` only, never `dept_id` or `role` baked in — those are
      looked up fresh per request from `department_memberships`, so a role change
      takes effect without requiring re-login

## 5. Phase 3 — wire identity into existing routers

`parsing_router.py` currently accepts `dept_id` as an unchecked parameter. This phase
adds `Depends(require_dept_access(dept_id, role_enum.editor))` (or `viewer` for reads)
to both `/parse` and `/parse/audio/stream`. This is a small, mechanical change but it's
the actual point where the system stops being trust-based — don't skip it or defer it
past this phase, since every route built after this point will copy whichever pattern
is already there.

Also add `company_router.py` and `department_router.py` here — provisioning needs to
exist before there's anywhere real to `/parse` into. `department_router.py` is where
`isolation_mode` gets set (see §3) — this is a one-time decision per department, made
at creation, not editable through a normal `PATCH`.

## 6. Phase 4 — `indexing/` + `vectorstore/`

This is the first point where chunks become searchable. Two things happen to every
chunk, in order:

1. **Embed.** `embedding_service.py` produces both a dense vector (BGE-M3) and a
   sparse vector (SPLADE) for every chunk — text, table summary, image description,
   audio segment alike. Per `vector-graph-routing.md`, this is unconditional: no
   content-type split at the vector layer.
2. **Route.** `routing_service.py` is a pure function reading `chunk.kind` /
   `ChunkMetadata.doc_type` to decide graph-layer eligibility (Layer 1 always, Layer 2
   selectively) — see the graph builder guide for what happens with that decision.

`indexing_pipeline.py` is the orchestrator tying these together: chunk → embed →
`vectorstore` write → hand off to `graph/` (once that module exists) for structural
edges and, if eligible, extraction queueing.

**`vectorstore/` is the Qdrant isolation layer**, structured to mirror what `graph/`
will need later:

- `resolver.py` — `HybridResolver`. At department-provisioning time, decides
  `isolation_mode` from `dept_type` + `regulatory_flags`: `isolated` (dedicated
  cluster, regulated data), `physical` (dedicated collection, sensitive dept types),
  or `logical` (shared `company_{company_id}` collection, payload-filtered by
  `dept_id`). This result gets written once to `departments.isolation_mode` and
  `departments.qdrant_collection` — never recomputed.
- `bound_client.py` — `BoundClient` (logical mode, injects `dept_id` filter into every
  operation) and `SimpleBoundClient` (physical/isolated mode, collection itself is the
  boundary). **The raw Qdrant client is never exposed to callers outside this file.**
  Filter injection in business logic is the anti-pattern this exists to prevent — one
  missed filter call is a data breach; a bound client makes omitting it impossible.
- `schema.py` — named vectors (`text`, `visual` if/when video lands), payload fields,
  RRF fusion config for combining dense + sparse results.

**Checklist:**
- [ ] Every chunk write goes through a `BoundClient`/`SimpleBoundClient` — no direct
      `qdrant_client.upsert()` calls anywhere outside `vectorstore/vector_store.py`
- [ ] A `logical`-mode dept's queries are structurally incapable of returning another
      dept's points, even if `dept_id` is omitted from the call site
- [ ] Embedding runs for every chunk kind — verify table summaries and image
      descriptions are embedded, not just raw text

## 7. Phase 5 — `workers/`

Closes the divergence noted in `status.md`: ingestion currently runs synchronously
in-request. This phase moves embedding (and later, graph extraction) onto Celery.

- `celery_app.py` — broker/backend config (Redis).
- `embed_worker.py` — `queue=embed`. Consumes chunks, calls `embedding_service.py`,
  writes to `vectorstore`.
- `maintenance_worker.py` — periodic jobs: usage counter updates
  (`department_usage`), orphan cleanup (points for documents deleted mid-ingestion).

**Keep `embed` and `graph_extract` on separate queue lanes from day one**, even before
`graph/` exists — this is a scale decision made in the folder structure now, not
something to retrofit. Embedding is fast; entity extraction is an LLM call and
materially slower. If they share a queue, a burst of documents starves the fast path.

**Checklist:**
- [ ] A failed embed job retries with backoff, doesn't silently drop the chunk
- [ ] `ingestion_jobs.stage` is updated at each transition (`parsing → chunking →
      embedding → indexing`) so `/indexing_router.py`'s status endpoint has something
      real to report
- [ ] Rollback on failure deletes vectorstore points by `doc_id` — mirror this pattern
      exactly when `graph/` adds its own rollback later

## 8. What's deliberately out of scope here

Don't start building `/search` or `/chat` endpoints in this pass, even though they're
tempting once vectors exist. `status.md` is explicit that reranking, MMR, hybrid
search fusion, and generation don't exist yet — build indexing solid first. Similarly,
don't build the graph module here — see `graph-builder-guide.md`.