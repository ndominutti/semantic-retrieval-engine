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
    """Service in charge of reranking documents given a query"""

    @async_error_handler_with_fallback(
        fallback=default_fallback_data_ids, retries=3, delay=3
    )
    async def rerank_get_ids(
        self, query: str, documents: dict, top_n: int
    ) -> List[Tuple[int, float]]:
        """
        Ranks the provided documents based on their relevance to the input query and returns the top N document
        indices with their relevance scores.

        Args:
            query (str): The search query string used to rank the documents.
            documents (dict): A dictionary containing a list of documents under the key "docs". Each document is
            expected to be a dictionary of column values.
            top_n (int): The number of top-ranked document indices to return.

        Returns:
            List[Tuple[int, float]]: A list of tuples, each containing the index of a top-ranked document and its
            corresponding relevance score.
        """
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
        """Re-ranks a set of documents based on their relevance to a given query and returns the top N ranked documents.

        Args:
            query (str): The search query used to rank the documents.
            documents (dict): A dictionary containing the documents to be ranked. Expected to have a "docs" key with a
            list of documents.
            top_n (int): The number of top-ranked documents to return.

        Returns:
            RerankDocsResponse: A list of the top N documents ranked by relevance to the query.
        """
        results = await self.rerank_get_ids(query, documents, top_n)
        ranked_ids = [x[0] for x in results]
        return [documents["docs"][i] for i in ranked_ids]
