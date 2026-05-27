# Skill: api-testing

**Trigger:** user says "test the API endpoint …" or "write integration tests for …"

## What This Skill Does
Generates and runs `httpx`-based integration tests for FastAPI endpoints, following project
test conventions in `tests/CLAUDE.md`.

## Template: New Endpoint Test

```python
# tests/api/test_<resource>.py
import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

async def test_<endpoint>_success(api_client: AsyncClient):
    response = await api_client.post(
        "/<endpoint>",
        headers={"X-API-Key": "test-key"},
        json={...},
    )
    assert response.status_code == 202
    data = response.json()
    assert "id" in data

async def test_<endpoint>_unauthenticated(api_client: AsyncClient):
    response = await api_client.post("/<endpoint>", json={...})
    assert response.status_code == 401

async def test_<endpoint>_invalid_input(api_client: AsyncClient):
    response = await api_client.post(
        "/<endpoint>",
        headers={"X-API-Key": "test-key"},
        json={"missing": "required_fields"},
    )
    assert response.status_code == 422
```

## Steps When Adding Tests for a New Endpoint

1. Identify the router file: `src/api/routers/<resource>.py`
2. List all paths + methods + expected status codes
3. Create `tests/api/test_<resource>.py` using the template above
4. Run scoped: `pytest tests/api/test_<resource>.py -x -q`
5. Check coverage: `pytest tests/api/ --cov=src/api --cov-report=term-missing`

## Run Command
```bash
pytest tests/api/ -x -q
```
