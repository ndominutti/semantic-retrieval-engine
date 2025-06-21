from pydantic import BaseModel
from typing import List
import pandas as pd


class RetrievalRequest(BaseModel):
    query: str


class RetrievalIDResponse(BaseModel):
    ids: List[int]


class RetrievalDoc(BaseModel):
    product_id: int
    product_name: str
    product_class: str
    category_hierarchy: str
    product_description: str
    product_features: str
    rating_count: float
    average_rating: float
    review_count: float
    score: float


class RetrievalDocsResponse(BaseModel):
    docs: List[RetrievalDoc]
