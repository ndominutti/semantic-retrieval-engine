import joblib
import os

ARTIFACTS_SAVE_PATH = os.getenv("ARTIFACTS_SAVE_PATH")


def load_tfidf_artifacts():
    tfidf_matrix = joblib.load(ARTIFACTS_SAVE_PATH + "tfidf_matrix.joblib")
    vectorizer = joblib.load(ARTIFACTS_SAVE_PATH + "tfidf_vectorizer.joblib")
    return vectorizer, tfidf_matrix


def load_bm25_artifacts():
    pass
