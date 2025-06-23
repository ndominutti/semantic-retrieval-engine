from typing import List, Optional, Tuple

from pydantic import BaseModel


class RerankRequest(BaseModel):
    query: str
    documents: dict
    top_n: int


class RerankIDResponse(BaseModel):
    ids_and_scores: List[Tuple[int, float]]


class RerankDoc(BaseModel):
    product_id: int
    product_name: str
    product_class: Optional[str]
    category_hierarchy: Optional[str]
    product_description: Optional[str]
    product_features: Optional[str]
    rating_count: Optional[float]
    average_rating: Optional[float]
    review_count: Optional[float]
    score: float


class RerankDocsResponse(BaseModel):
    docs: List[RerankDoc]
