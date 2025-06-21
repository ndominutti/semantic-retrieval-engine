from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
from scipy.sparse._csr import spmatrix
from .base_embedder import BaseLexicalEmbedder
from typing import List


class TDIDFLexicalEmbedder(BaseLexicalEmbedder):
    def __init__(self, save_dir: str):
        self.vectorizer = TfidfVectorizer()
        self.save_dir = save_dir

    def fit_transform(
        self, products_df: pd.DataFrame, cols_to_embed: List[str]
    ) -> spmatrix:
        # Validate columns
        missing = [col for col in cols_to_embed if col not in products_df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")
        combined_text = products_df[cols_to_embed].fillna("").agg(" ".join, axis=1)
        tfidf_matrix = self.vectorizer.fit_transform(combined_text.values.astype("U"))
        return tfidf_matrix

    def save(self, vectors: spmatrix) -> None:
        # saves locally for this PoC, would save in a repository as an S3 for production level
        joblib.dump(self.vectorizer, f"{self.save_dir}/tfidf_vectorizer.joblib")
        joblib.dump(vectors, f"{self.save_dir}/tfidf_matrix.joblib")


# class BM25LexicalEmbedder:
#     def __init__(self, save_dir: str):
#         from rank_bm25 import BM25Okapi

#         self.vectorizer = BM25Okapi
#         self.save_dir = save_dir

#     def fit(self, documents: list[str]):
#         tokenized_corpus = [
#             doc.lower().split() for doc in documents
#         ]  # should be improved for production level
#         bm25 = self.vectorizer(tokenized_corpus)
#         return bm25

#     def save(self, bm25):
#         joblib.dump(bm25, f"{self.save_dir}/bm25.joblib")
