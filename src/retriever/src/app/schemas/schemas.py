from typing import List, Optional

from pydantic import BaseModel


class RetrievalRequest(BaseModel):
    query: str
    top_n: int


class RetrievalIDResponse(BaseModel):
    ids: List[int]


class RetrievalDoc(BaseModel):
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


class RetrievalDocsResponse(BaseModel):
    docs: List[RetrievalDoc]
