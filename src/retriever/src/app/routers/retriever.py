import numpy as np
from fastapi import APIRouter
from utils import logger

from .. import schemas
from ..services import RetrievalService
from .examples import retrieve_docs_examples, retrieve_ids_examples

router = APIRouter()

retrieval_service: RetrievalService = RetrievalService()


@router.post(
    "/retrieve_ids",
    response_model=schemas.RetrievalIDResponse,
    openapi_extra=retrieve_ids_examples,
)
async def retrieve_docs_ids(
    request: schemas.RetrievalRequest,
) -> schemas.RetrievalIDResponse:
    logger.info(f"New request: {request.query}")
    doc_ids = await retrieval_service.retrieve_ids(
        request.query, request.top_n, return_score=False
    )
    response = schemas.RetrievalIDResponse(ids=doc_ids)
    return response


@router.post(
    "/retrieve_docs",
    response_model=schemas.RetrievalDocsResponse,
    openapi_extra=retrieve_docs_examples,
)
async def retrieve_docs(
    request: schemas.RetrievalRequest,
) -> schemas.RetrievalDocsResponse:
    doc = await retrieval_service.retrieve_docs(request.query, request.top_n)
    response = []
    for _, d in doc.replace({np.nan: None}).iterrows():
        response.append(
            schemas.RetrievalDoc(
                product_id=d.product_id,
                product_name=d.product_name,
                product_class=d.product_class,
                category_hierarchy=d["category hierarchy"],
                product_description=d.product_description,
                product_features=d.product_features,
                rating_count=d.rating_count,
                average_rating=d.average_rating,
                review_count=d.review_count,
                score=d.scores,
            )
        )
    response = schemas.RetrievalDocsResponse(docs=response)
    return response
