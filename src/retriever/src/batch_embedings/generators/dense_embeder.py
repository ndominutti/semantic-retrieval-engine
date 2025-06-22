import pandas as pd
from .base_embeder import BaseDenseEmbeder
from typing import List
import cohere
import os
import asyncio
from ...utils import load_config, logger
from exceptions import MissingColumnsError

config = load_config()
cohere_model = config["retrievers"]["dense"]["model"]
max_tokens = config["retrievers"]["dense"]["max_tokens"]
output_dim = config["retrievers"]["dense"]["output_dim"]

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(COHERE_API_KEY)


class CohereEmbeder(BaseDenseEmbeder):

    def __init__(self, embedding_dim=output_dim):
        """Initializes the DenseEmbeder with the specified embedding dimension.
        Args:
            embedding_dim (int, optional): The dimension of the embedding vectors. Defaults to output_dim.
        """
        self.embedding_dim = embedding_dim

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
            output_dimension=self.embedding_dim,
            embedding_types=["float"],
        )

    async def embed(
        self, products_df: pd.DataFrame, cols_to_embed: List[str]
    ) -> List[List[float]]:
        """Runs row-wise async embeddings on the concatenation of the cols_to_embed in the product_df

        Args:
            products_df (pd.DataFrame): dataframe to be processed. Must contains the columns included
            in cols_to_embed.
            cols_to_embed (List[str]): columns to concatenate prior embedding

        Raises:
            MissingColumnsError: if any of the columns in cols_to_embed is not in product_df

        Returns:
            List[List[float]]: embeddings for each one of the rows
        """
        missing = [col for col in cols_to_embed if col not in products_df.columns]
        if missing:
            raise MissingColumnsError(f"Missing columns in DataFrame: {missing}")
        combined_text = (
            products_df[cols_to_embed].fillna("").agg(" ".join, axis=1).tolist()
        )
        response = await self.get_embeddings_async(combined_text)
        logger.debug(f"Cohere Embedder response: {response}")
        return response.embeddings.float
