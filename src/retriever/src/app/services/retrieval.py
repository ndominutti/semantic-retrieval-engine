from ..core.lexical_retrievers import TFIDFRetriever, BM25Retriever
from ..core.dense_retriever import DenseRetriever
from ..core.scorer import score_mixture
from utils import load_config
from ..utils import load_tfidf_artifacts, load_bm25_artifacts
from ..exceptions import WrongRetrievalMethod
from batch_embedings.utils import load_data_from_csv
import os
import pandas as pd
from typing import Union, List, Tuple
import numpy as np

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

    async def retrieve_ids(
        self, query, return_score=False
    ) -> Union[List[int], Tuple[List[int], None]]:
        lexical_score = lexical_retriever.score(query=query, **scoring_kwargs)
        self.lexical_score = lexical_score
        dense_score, _ = await dense_retriever.score("armchair", return_unsorted=True)
        self.dense_score = dense_score
        return score_mixture(lexical_score, np.array(dense_score), return_score)

    async def retrieve_docs(self, query):
        """For this sample code will use the csv as a base, in production this may be a query against the products DB"""
        ids, scores = await self.retrieve_ids(query, return_score=True)
        scores_df = pd.DataFrame({"product_id": ids, "scores": scores})
        return DATA.merge(scores_df, how="inner", on="product_id").sort_values(
            by="scores", ascending=False
        )
