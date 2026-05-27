# Skill: schema-migrate

**Trigger:** user says "add field … to model …" or "create migration for …"

## What This Skill Does
Guides the three-step process of extending a Pydantic model and keeping the DB schema in sync:
model change → Alembic migration → test update.

## Steps

### 1. Update the Pydantic model
File: `src/models/schemas.py` (request/response) and/or `src/models/db.py` (ORM mapped class).

```python
# Example: adding `language` field to DocumentSchema
class DocumentSchema(BaseModel):
    id: UUID
    status: JobStatus
    language: str | None = None    # ← add here, with default for backwards compat
```

### 2. Generate Alembic migration
```bash
alembic revision --autogenerate -m "add_language_to_documents"
# Review the generated file in alembic/versions/
# Verify upgrade() and downgrade() are correct before proceeding
```

### 3. Apply migration (dev)
```bash
alembic upgrade head
```

### 4. Update affected tests
```bash
# Find tests that construct DocumentSchema directly
grep -r "DocumentSchema(" tests/
# Update fixtures / assertions for the new field
```

### 5. Run scoped tests
```bash
pytest tests/ -x -q -k "document"
```

## Safety Checklist

- [ ] New fields have a default (`None` or a value) so existing rows are valid after migration
- [ ] `downgrade()` in the migration reverses `upgrade()` cleanly
- [ ] No `NOT NULL` column added without a server default or backfill in the migration
- [ ] `mypy src/` passes after the change
