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

    async def rerank(self, query: str, documents: List[str], top_n: int):
        return await asyncio.to_thread(
            co.rerank, model=cohere_model, query=query, documents=documents, top_n=top_n
        )
