# main.py
import logging

from fastapi import FastAPI
from api.parsing_router import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI()
app.include_router(router)