from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
from scipy.sparse._csr import spmatrix
from .base_embeder import BaseLexicalEmbeder
from typing import List
from exceptions import MissingColumnsError


class TDIDFLexicalEmbeder(BaseLexicalEmbeder):
    def __init__(self, save_dir: str):
        self.vectorizer = TfidfVectorizer()
        self.save_dir = save_dir

    def fit_transform(
        self, products_df: pd.DataFrame, cols_to_embed: List[str]
    ) -> spmatrix:
        missing = [col for col in cols_to_embed if col not in products_df.columns]
        if missing:
            raise MissingColumnsError(f"Missing columns in DataFrame: {missing}")
        combined_text = products_df[cols_to_embed].fillna("").agg(" ".join, axis=1)
        tfidf_matrix = self.vectorizer.fit_transform(combined_text.values.astype("U"))
        return tfidf_matrix

    def save(self, vectors: spmatrix) -> None:
        # saves locally for this PoC, would save in a repository as an S3 for production level
        joblib.dump(self.vectorizer, f"{self.save_dir}/tfidf_vectorizer.joblib")
        joblib.dump(vectors, f"{self.save_dir}/tfidf_matrix.joblib")


class BM25LexicalEmbeder:
    """To be implemented"""

    pass
