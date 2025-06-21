from abc import abstractmethod
import pandas as pd
from typing import List
from scipy.sparse._csr import spmatrix
import faiss
import numpy as np


class BaseLexicalEmbedder:

    @abstractmethod
    def fit_transform(
        self, products_df: pd.DataFrame, cols_to_embed: List[str]
    ) -> spmatrix:
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass


class BaseDenseEmbedder:

    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    @abstractmethod
    async def embed(
        self, products_df: pd.DataFrame, cols_to_embed: List[str]
    ) -> List[List[float]]:
        pass

    def index_and_save(self, embeddings: List[List[float]], save_dir: str) -> None:
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(np.array(norm_embeddings).astype("float32"))
        faiss.write_index(index, save_dir + "products_index.faiss")
