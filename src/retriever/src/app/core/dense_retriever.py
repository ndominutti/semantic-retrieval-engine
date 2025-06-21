import pandas as pd
from .base_model import RetrievalBase
from typing import List, Tuple
import cohere
import os
import asyncio
from utils import load_config
from ..utils import faiss_connection
import numpy as np

config = load_config()
cohere_model = config["retrievers"]["dense"]["model"]
max_tokens = config["retrievers"]["dense"]["max_tokens"]
output_dim = config["retrievers"]["dense"]["output_dim"]


COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(COHERE_API_KEY)


class DenseRetriever(RetrievalBase):

    def __init__(self):
        self.index = faiss_connection.load_index()

    async def get_embeddings_async(self, texts: List[str]):
        return await asyncio.to_thread(
            co.embed,
            texts=texts,
            input_type="search_document",
            model=cohere_model,
            max_tokens=max_tokens,
            output_dimension=output_dim,
            embedding_types=["float"],
        )

    async def score(
        self, query: str, top_n: int = None, return_unsorted=False
    ) -> Tuple[List[float], List[int]]:
        """Bring scores and idx sorted for ALL the products in the index

        Args:
            query (str): _description_
            top_n (int): _description_

        Returns:
            Tuple[List[float], List[int]]:
        """
        query_embedding = await self.get_embeddings_async([query])
        query_embedding = np.array(query_embedding.embeddings.float_, dtype=np.float32)[
            0
        ]
        norm_query_embedding = query_embedding / np.linalg.norm(query_embedding)
        distances, idxs = self.index.search(
            norm_query_embedding[None, :], k=self.index.ntotal if not top_n else top_n
        )
        if return_unsorted:
            return self._unsorted_scores(distances, idxs)
        return distances[0], idxs[0]

    def _unsorted_scores(
        self, distances: List[float], idxs: List[int]
    ) -> Tuple[List[float], List[int]]:
        dense_scores_unsorted = np.zeros(self.index.ntotal, dtype=np.float32)
        for score, idx in zip(distances[0], idxs[0]):
            dense_scores_unsorted[idx] = score
        return dense_scores_unsorted.tolist(), [*range(self.index.ntotal)]

    async def retrieve(self, query: str, top_n: int) -> List[int]:
        _, idxs = await self.score(query, top_n)
        return idxs
