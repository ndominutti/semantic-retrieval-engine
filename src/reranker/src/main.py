from app.routers import main_router
from fastapi import FastAPI

app = FastAPI()

app.include_router(main_router)


@app.get("/health")
def health():
    return {"status": "ok"}
