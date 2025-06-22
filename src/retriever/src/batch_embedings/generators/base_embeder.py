from abc import abstractmethod
import pandas as pd
from typing import List
from scipy.sparse._csr import spmatrix
import faiss
import numpy as np


class BaseLexicalEmbeder:
    """Base class for lexical embedders"""

    @abstractmethod
    def fit_transform(
        self, products_df: pd.DataFrame, cols_to_embed: List[str]
    ) -> spmatrix:
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass


class BaseDenseEmbeder:

    def __init__(self, embedding_dim):
        """Initializes the base embedder with the specified embedding dimension.

        Args:
            embedding_dim (int): The dimensionality of the embedding vectors.
        """
        self.embedding_dim = embedding_dim

    @abstractmethod
    async def embed(
        self, products_df: pd.DataFrame, cols_to_embed: List[str]
    ) -> List[List[float]]:
        pass

    def index_and_save(self, embeddings: List[List[float]], save_dir: str) -> None:
        """Normalizes the given embeddings, creates a FAISS index using inner product similarity,
        adds the embeddings to the index, and saves the index to the specified directory.

        Args:
            embeddings (List[List[float]]): A 2D list or array of embedding vectors to be indexed.
            save_dir (str): The directory path where the FAISS index file ('products_index.faiss') will be saved.

        Returns:
            None
        """
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(np.array(norm_embeddings).astype("float32"))
        faiss.write_index(index, save_dir + "products_index.faiss")
