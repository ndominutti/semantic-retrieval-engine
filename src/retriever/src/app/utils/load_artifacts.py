import os
from typing import Tuple

import joblib
from scipy.sparse._csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

ARTIFACTS_SAVE_PATH = os.getenv("ARTIFACTS_SAVE_PATH")


def load_tfidf_artifacts() -> Tuple[TfidfVectorizer, csr_matrix]:
    """Load TFIDF artifacts

    Returns:
        Tuple[TfidfVectorizer, csr_matrix]: tfidf vectorizer and tfidf matrix
    """
    tfidf_matrix = joblib.load(ARTIFACTS_SAVE_PATH + "tfidf_matrix.joblib")
    vectorizer = joblib.load(ARTIFACTS_SAVE_PATH + "tfidf_vectorizer.joblib")
    return vectorizer, tfidf_matrix


def load_bm25_artifacts():
    pass
