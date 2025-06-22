from .base_model import RetrievalBase
from ..exceptions import WrongRetrievalMethod, WrongSimilarityMethod
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TFIDFRetriever(RetrievalBase):

    def __init__(self, similarity_method="cosine"):
        if similarity_method == "cosine":
            self.similarity_algorithm = cosine_similarity
        else:
            raise WrongSimilarityMethod(
                f"Similarity method {similarity_method} is not supported for LexicalRetriever. Must be cosine"
            )

    def score(self, query, vectorizer, tfidf_matrix) -> np.ndarray:
        query_vector = vectorizer.transform([query])
        similarity_scores = self.similarity_algorithm(
            query_vector, tfidf_matrix
        ).flatten()
        return similarity_scores

    def retrieve(self, query, vectorizer, tfidf_matrix, top_n) -> np.ndarray:
        similarity_scores = self.score(query, vectorizer, tfidf_matrix)
        return similarity_scores.argsort()[-top_n:][::-1]


class BM25Retriever(RetrievalBase):
    """To be implemented"""

    pass
