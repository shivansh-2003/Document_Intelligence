# main.py
from fastapi import FastAPI
from api.parsing_router import router

app = FastAPI()
app.include_router(router)