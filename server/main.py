from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database import init_db

app = FastAPI(
    title="Credit Score and Loan Recommendation",
    description="This is a API for credit score and loan recommendation",
    version="0.0.1",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    pass

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

