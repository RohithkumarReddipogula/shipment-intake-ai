from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.db import connect, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = connect(get_settings().db_path)
    init_db(conn)
    conn.close()
    yield


app = FastAPI(title="shipment-intake-ai", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
