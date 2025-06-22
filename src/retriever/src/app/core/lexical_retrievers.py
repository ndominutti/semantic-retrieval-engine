from .base_model import RetrievalBase
from ..exceptions import WrongRetrievalMethod, WrongSimilarityMethod
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse._csr import spmatrix


class TFIDFRetriever(RetrievalBase):
    def __init__(self, similarity_method="cosine"):
        """Initializes the LexicalRetriever with the specified similarity method.

        Args:
            similarity_method (str, optional): The similarity method to use. Currently,
            only "cosine" is supported. Defaults to "cosine".
        """
        if similarity_method == "cosine":
            self.similarity_algorithm = cosine_similarity
        else:
            raise WrongSimilarityMethod(
                f"Similarity method {similarity_method} is not supported for LexicalRetriever. Must be cosine"
            )

    def score(
        self, query: str, vectorizer: TfidfVectorizer, tfidf_matrix: spmatrix
    ) -> np.ndarray:
        """Calculates similarity scores between a query and a TF-IDF matrix using the
        specified similarity algorithm.

        Args:
            query (str): The input query string to be vectorized and compared.
            vectorizer (TfidfVectorizer): The vectorizer used to transform the
            query into a vector representation.
            tfidf_matrix (spmatrix): The TF-IDF matrix representing
            the corpus documents.

        Returns:
            np.ndarray: An array of similarity scores between the query and each document in the
            TF-IDF matrix.
        """
        query_vector = vectorizer.transform([query])
        similarity_scores = self.similarity_algorithm(
            query_vector, tfidf_matrix
        ).flatten()
        return similarity_scores

    def retrieve(
        self,
        query: str,
        vectorizer: TfidfVectorizer,
        tfidf_matrix: spmatrix,
        top_n: int,
    ) -> np.ndarray:
        """Retrieves the indices of the top_n most similar documents to the given query based on
        TF-IDF similarity scores.

        Args:
            query (str): The input query string to search for similar documents.
            vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer used to transform the query.
            tfidf_matrix (spmatrix): The TF-IDF matrix representing the corpus of documents.
            top_n (int): The number of top similar documents to retrieve.

        Returns:
            np.ndarray: An array of indices corresponding to the top_n most similar documents, sorted
            by descending similarity.
        """
        similarity_scores = self.score(query, vectorizer, tfidf_matrix)
        return similarity_scores.argsort()[-top_n:][::-1]


class BM25Retriever(RetrievalBase):
    """To be implemented"""

    pass
