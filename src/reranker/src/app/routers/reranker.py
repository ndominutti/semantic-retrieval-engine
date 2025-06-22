from ..services import RerankService
from fastapi import APIRouter
from .. import schemas
from utils import logger
import numpy as np

from .examples import rerank_ids_examples, rerank_docs_examples

router = APIRouter()

rerank_service: RerankService = RerankService()


@router.post(
    "/rerank_get_ids",
    response_model=schemas.RerankIDResponse,
    openapi_extra=rerank_ids_examples,
)
async def rerank_docs_ids(
    request: schemas.RerankRequest,
) -> schemas.RerankIDResponse:
    logger.info(f"New request: {request.query}")
    r_docs = await rerank_service.rerank_get_ids(
        request.query, request.documents, request.top_n
    )
    response = schemas.RerankIDResponse(ids_and_scores=r_docs)
    return response


@router.post(
    "/rerank_get_docs",
    response_model=schemas.RerankDocsResponse,
    openapi_extra=rerank_docs_examples,
)
async def rerank_docs(
    request: schemas.RerankRequest,
) -> schemas.RerankDocsResponse:
    docs = await rerank_service.rerank(request.query, request.documents, request.top_n)
    response = schemas.RerankDocsResponse(docs=docs)
    return response
