from fastapi import APIRouter

from .retriever import router as retrieve_route

main_router = APIRouter()

main_router.include_router(
    retrieve_route, prefix="/retrieval", tags=["retrieval endpoint"]
)
