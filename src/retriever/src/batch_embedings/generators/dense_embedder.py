import pandas as pd
from .base_embedder import BaseDenseEmbedder
from typing import List
import cohere
import os
import asyncio
from utils import load_config

config = load_config()
cohere_model = config["retrievers"]["dense"]["model"]
max_tokens = config["retrievers"]["dense"]["max_tokens"]
output_dim = config["retrievers"]["dense"]["output_dim"]

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(COHERE_API_KEY)


class CohereEmbedder(BaseDenseEmbedder):

    def __init__(self, embedding_dim=output_dim):
        self.embedding_dim = embedding_dim

    async def get_embeddings_async(self, texts: List[str]):
        return await asyncio.to_thread(
            co.embed,
            texts=texts,
            input_type="search_document",
            model=cohere_model,
            max_tokens=max_tokens,
            output_dimension=self.embedding_dim,
            embedding_types=["float"],
        )

    async def embed(
        self, products_df: pd.DataFrame, cols_to_embed: List[str]
    ) -> List[List[float]]:
        missing = [col for col in cols_to_embed if col not in products_df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")
        combined_text = (
            products_df[cols_to_embed].fillna("").agg(" ".join, axis=1).tolist()
        )
        response = await self.get_embeddings_async(combined_text)
        return response.embeddings.float
