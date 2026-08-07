# main.py
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

from api.auth_router import router as auth_router
from api.company_router import router as company_router
from api.department_router import router as department_router
from api.documents_router import router as documents_router
from api.membership_router import router as membership_router
from api.parsing_router import router as parsing_router
from api.retrieval_router import router as retrieval_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI()
app.include_router(auth_router)
app.include_router(company_router)
app.include_router(department_router)
app.include_router(documents_router)
app.include_router(membership_router)
app.include_router(parsing_router)
app.include_router(retrieval_router)

# Dev convenience: a real <input type="file" multiple> page for /parse/batch, since
# Swagger UI's file-picker rendering was flaky pre-fix below -- same-origin mount
# avoids any CORS setup for what's just a local testing page.
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


def _openapi_with_file_arrays() -> dict:
    """FastAPI/Pydantic v2 emit OpenAPI 3.1-style {"contentMediaType": ...} for
    list[UploadFile] fields (e.g. /parse/batch's `files`). swagger-ui-dist@5 doesn't
    reliably recognize that on array items and falls back to rendering a plain
    array<string> text-box widget instead of file pickers. Swagger's file-picker
    rendering is keyed off the older OpenAPI 3.0 "format: binary" instead, which it
    does support reliably -- so this walks every generated component schema once and
    swaps one for the other. Purely documentation-level: actual request parsing is
    governed by each route's `list[UploadFile] = File(...)` signature, untouched here.
    """
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    for component in schema.get("components", {}).get("schemas", {}).values():
        for prop in component.get("properties", {}).values():
            items = prop.get("items")
            if isinstance(items, dict) and items.pop("contentMediaType", None):
                items["format"] = "binary"
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _openapi_with_file_arrays