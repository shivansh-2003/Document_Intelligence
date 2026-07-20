# main.py
import logging

from fastapi import FastAPI

from api.auth_router import router as auth_router
from api.company_router import router as company_router
from api.department_router import router as department_router
from api.membership_router import router as membership_router
from api.parsing_router import router as parsing_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI()
app.include_router(auth_router)
app.include_router(company_router)
app.include_router(department_router)
app.include_router(membership_router)
app.include_router(parsing_router)