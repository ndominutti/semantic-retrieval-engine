from ..core.cohere_reranker import CohereReranker
from utils import load_config, logger
from typing import List, Tuple
from ..schemas import RerankDocsResponse
import pandas as pd
from ..fallbacks import (
    async_error_handler_with_fallback,
    default_fallback_data_ids,
    default_fallback_data_docs,
)

config = load_config()
columns_to_rerank = config["columns_to_rerank"]

cohere_reranker: CohereReranker = CohereReranker()


class RerankService:

    @async_error_handler_with_fallback(
        fallback=default_fallback_data_ids, retries=3, delay=3
    )
    async def rerank_get_ids(
        self, query: str, documents: dict, top_n: int
    ) -> List[Tuple[int, float]]:
        documents_list = [
            " ".join(str(doc.get(col, "")) for col in columns_to_rerank)
            for doc in documents["docs"]
        ]
        logger.debug(documents_list)
        res = await cohere_reranker.rerank(query, documents_list, top_n)
        return [(r.index, r.relevance_score) for r in res.results]

    @async_error_handler_with_fallback(
        fallback=default_fallback_data_docs, retries=3, delay=3
    )
    async def rerank(
        self, query: str, documents: dict, top_n: int
    ) -> RerankDocsResponse:
        results = await self.rerank_get_ids(query, documents, top_n)
        ranked_ids = [x[0] for x in results]
        return [documents["docs"][i] for i in ranked_ids]
