from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
from scipy.sparse._csr import spmatrix
from .base_embeder import BaseLexicalEmbeder
from typing import List
from exceptions import MissingColumnsError


class TDIDFLexicalEmbeder(BaseLexicalEmbeder):
    def __init__(self, save_dir: str):
        """Initializes the LexicalEmbeder with a TF-IDF vectorizer and sets the directory for saving embeddings.

        Args:
            save_dir (str): Path to the directory where embeddings or related files will be saved.
        """
        self.vectorizer = TfidfVectorizer()
        self.save_dir = save_dir

    def fit_transform(
        self, products_df: pd.DataFrame, cols_to_embed: List[str]
    ) -> spmatrix:
        """Fits the internal vectorizer on the specified columns of the input DataFrame and transforms the text
        data into a sparse matrix of embeddings.

        Args:
            products_df (pd.DataFrame): DataFrame containing product data with columns to be embedded.
            cols_to_embed (List[str]): List of column names in the DataFrame whose text content will be combined
            and embedded.

        Returns:
            spmatrix: Sparse matrix representation of the embedded text data, as produced by the fitted vectorizer.
        """
        missing = [col for col in cols_to_embed if col not in products_df.columns]
        if missing:
            raise MissingColumnsError(f"Missing columns in DataFrame: {missing}")
        combined_text = products_df[cols_to_embed].fillna("").agg(" ".join, axis=1)
        tfidf_matrix = self.vectorizer.fit_transform(combined_text.values.astype("U"))
        return tfidf_matrix

    def save(self, vectors: spmatrix) -> None:
        """Save the vectorizer and tfidf matrix to storage.

        Args:
            vectors (spmatrix): tfidf matrix
        """
        # saves locally for this PoC, would save in a repository as an S3 for production level
        joblib.dump(self.vectorizer, f"{self.save_dir}/tfidf_vectorizer.joblib")
        joblib.dump(vectors, f"{self.save_dir}/tfidf_matrix.joblib")


class BM25LexicalEmbeder:
    """To be implemented"""

    pass
