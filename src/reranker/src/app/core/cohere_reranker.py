import pandas as pd
from typing import List, Tuple, Union
import cohere
import os
import asyncio
from utils import load_config
import numpy as np

config = load_config()
cohere_model = config["model"]


COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(COHERE_API_KEY)


class CohereReranker:
    """Class that provides an asynchronous interface for reranking documents using the Cohere API."""

    async def rerank(self, query: str, documents: List[str], top_n: int) -> List[dict]:
        """Re-ranks a list of documents based on their relevance to a given query using the Cohere model.

        Args:
            query (str): The search query to rank documents against.
            documents (List[str]): A list of documents to be re-ranked.
            top_n (int): The number of top-ranked documents to return.

        Returns:
            List[dict]: A list of dictionaries containing the top_n documents and their relevance scores, sorted by relevance in descending order.
        """
        return await asyncio.to_thread(
            co.rerank, model=cohere_model, query=query, documents=documents, top_n=top_n
        )
