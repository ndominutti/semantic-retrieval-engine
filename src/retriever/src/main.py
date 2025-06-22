from fastapi import FastAPI
from app.routers import main_router

app = FastAPI()

app.include_router(main_router)


@app.get("/health")
def health():
    return {"status": "ok"}
