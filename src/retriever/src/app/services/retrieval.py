import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from batch_embedings.utils import load_data_from_csv
from utils import load_config, logger

from ..core.dense_retriever import DenseRetriever
from ..core.lexical_retrievers import TFIDFRetriever
from ..core.scorer import score_mixture
from ..exceptions import WrongRetrievalMethod
from ..fallbacks import (
    async_error_handler_with_fallback,
    default_fallback_data_docs,
    default_fallback_data_ids,
)
from ..utils import load_tfidf_artifacts

config = load_config()
LEXICAL_METHOD = config["retrievers"]["lexical"]["method"]
DATA = load_data_from_csv(os.getenv("DATA_PATH"))

if LEXICAL_METHOD == "tfidf":
    lexical_retriever = TFIDFRetriever()
    vectorizer, tfidf_matrix = load_tfidf_artifacts()
    scoring_kwargs = {"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix}
elif LEXICAL_METHOD == "bm25":
    pass
else:
    WrongRetrievalMethod(
        "Selected method for lexical retrieval is not supported {LEXICAL_METHOD}. Must be tfidf or bm25"
    )

dense_retriever: DenseRetriever = DenseRetriever()


class RetrievalService:
    """Service in charge of retrieving documents given a query"""

    @async_error_handler_with_fallback(
        fallback=default_fallback_data_ids, retries=3, delay=3
    )
    async def retrieve_ids(
        self, query, top_n, return_score=False
    ) -> Union[List[int], Tuple[List[int], None]]:
        """Retrieves the top N document IDs relevant to the given query using a mixture of lexical and
        dense retrieval scores.

        Args:
            query (str): The input query string to search for relevant documents.
            top_n (int): The number of top document IDs to retrieve.
            return_score (bool, optional): If True, also returns the corresponding scores for the retrieved
            IDs. Defaults to False.

            Returns:
            Union[List[int], Tuple[List[int], None]]:
                - If return_score is False: A list of the top N retrieved document IDs.
                - If return_score is True: A tuple containing the list of top N retrieved document IDs and
                their corresponding scores.

        """
        lexical_score = lexical_retriever.score(query=query, **scoring_kwargs)
        logger.debug(lexical_score)
        dense_score, _ = await dense_retriever.score(query, return_unsorted=True)
        logger.debug(lexical_score)
        return score_mixture(lexical_score, np.array(dense_score), top_n, return_score)

    @async_error_handler_with_fallback(
        fallback=default_fallback_data_docs, retries=3, delay=3
    )
    async def retrieve_docs(self, query: str, top_n: int) -> pd.DataFrame:
        """Retrieves the top N documents most relevant to the given query.

        Args:
            query (str): The search query string used to retrieve relevant documents.
            top_n (int): The number of top documents to retrieve based on relevance.

        Resturns:
            pd.DataFrame: A DataFrame containing the merged product data and their relevance
            scores, sorted in descending order of scores.

        """
        # For this sample code will use the csv as a base, in production this may be a query against the products DB
        ids, scores = await self.retrieve_ids(query, top_n, return_score=True)
        scores_df = pd.DataFrame({"product_id": ids, "scores": scores})
        return DATA.merge(scores_df, how="inner", on="product_id").sort_values(
            by="scores", ascending=False
        )
