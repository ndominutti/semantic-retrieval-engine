import pandas as pd
from .base_model import RetrievalBase
from typing import List, Tuple, Union
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
        """Initializes the DenseRetriever with the specified similarity method."""
        self.index = faiss_connection.load_index()

    async def get_embeddings_async(self, texts: List[str]):
        """Run async embeddings request.

        Args:
            texts (List[str]): texts to be embedded
        """
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
        self, query: str, top_n: Union[int, None] = None, return_unsorted=False
    ) -> Tuple[List[float], List[int]]:
        """Computes similarity scores between a query and all products in the index.

        Args:
            query (str): The input query string to be embedded and compared.
            top_n (Optional[int], optional): The number of top results to return.
            If None, returns scores for all products in the index. Defaults to None.
            return_unsorted (bool, optional): If True, returns scores and indices in their
            original (unsorted) order. Defaults to False.

        Returns:
            Tuple[List[float], List[int]]: A tuple containing a list of similarity scores and
            a list of corresponding product indices.

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
        """Generates a list of dense scores aligned with the full index, filling in zeros
        for non-selected indices.

        Args:
            distances (List[float]): A list containing the similarity or distance scores
            for selected items.
            idxs (List[int]): A list containing the indices of the selected items in the index.

        Returns:
            Tuple[List[float], List[int]]:
                - A list of scores
                - A list of indices

        """
        dense_scores_unsorted = np.zeros(self.index.ntotal, dtype=np.float32)
        for score, idx in zip(distances[0], idxs[0]):
            dense_scores_unsorted[idx] = score
        return dense_scores_unsorted.tolist(), [*range(self.index.ntotal)]

    async def retrieve(self, query: str, top_n: int) -> List[int]:
        _, idxs = await self.score(query, top_n)
        return idxs
