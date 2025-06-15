from fastapi import FastAPI

app = FastAPI(
    title="Credit Score and Loan Recommendation",
    description="This is a API for credit score and loan recommendation",
    version="0.0.1",
)

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

