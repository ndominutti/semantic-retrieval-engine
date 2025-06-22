from fastapi import APIRouter

from .reranker import router as rerank_route

main_router = APIRouter()

main_router.include_router(rerank_route, prefix="/rerank", tags=["reranker endpoint"])
