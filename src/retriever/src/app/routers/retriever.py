from ..services import RetrievalService
from fastapi import APIRouter
from .. import schemas


router = APIRouter()

retrieval_service: RetrievalService = RetrievalService()


@router.post(
    "/retrieve_ids",
    response_model=schemas.RetrievalIDResponse,
    openapi_extra={},
)
async def retrieve_docs_ids(
    request: schemas.RetrievalRequest,
) -> schemas.RetrievalIDResponse:
    doc_ids = await retrieval_service.retrieve_ids(request.query, return_score=False)
    response = schemas.RetrievalIDResponse(ids=doc_ids)
    return response
