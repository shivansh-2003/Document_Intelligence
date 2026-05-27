# Skill: doc-pipeline

**Trigger:** user says "run the pipeline on …" or "process this document …"

## What This Skill Does
Guides end-to-end execution of the document intelligence pipeline on a given input file:
ingestion → extraction → analysis → result retrieval.

## Steps

### 1. Verify environment
```bash
# Check stack is running
docker compose ps
# Expected: api, worker, db, redis all "Up"
```
If not running: `docker compose up -d` and wait ~10s.

### 2. Upload the document
```bash
curl -X POST http://localhost:8000/upload \
  -H "X-API-Key: $TEST_API_KEY" \
  -F "file=@<PATH_TO_FILE>" \
  | jq .
# Note the returned document_id
```

### 3. Poll until complete
```bash
DOC_ID=<document_id>
while true; do
  STATUS=$(curl -s http://localhost:8000/status/$DOC_ID \
    -H "X-API-Key: $TEST_API_KEY" | jq -r '.status')
  echo "Status: $STATUS"
  [[ "$STATUS" == "COMPLETE" || "$STATUS" == "FAILED" ]] && break
  sleep 2
done
```

### 4. Retrieve result
```bash
curl http://localhost:8000/result/$DOC_ID \
  -H "X-API-Key: $TEST_API_KEY" \
  | jq .
```

### 5. If status is FAILED
```bash
# Check worker logs for the specific task error
docker compose logs worker --tail=50 | grep -A 10 "$DOC_ID"
```

## Scoped Tests for This Flow
```bash
pytest tests/api/test_documents.py tests/ingestion/ tests/extraction/ -x -q
```
