from .base_model import RetrievalBase
from ..exceptions import WrongRetrievalMethod, WrongSimilarityMethod
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever(RetrievalBase):

    def __init__(self, similarity_method="cosine"):
        if similarity_method == "cosine":
            self.similarity_algorithm = cosine_similarity
        else:
            raise WrongSimilarityMethod(
                f"Similarity method {similarity_method} is not supported for LexicalRetriever. Must be cosine"
            )

    def score(self, query, vectorizer, tfidf_matrix):
        query_vector = vectorizer.transform([query])
        similarity_scores = self.similarity_algorithm(
            query_vector, tfidf_matrix
        ).flatten()
        return similarity_scores

    def retrieve(self, query, vectorizer, tfidf_matrix, top_n):
        similarity_scores = self.score(query, vectorizer, tfidf_matrix)
        return similarity_scores.argsort()[-top_n:][::-1]


class BM25Retriever(RetrievalBase):

    def score(self, query, bm25_tokenizer, bm25_model):
        return bm25_model.get_scores(bm25_tokenizer(query))

    def retrieve(self, query, bm25_tokenizer, bm25_model, top_n):
        scores = self.score(query, bm25_tokenizer, bm25_model)
        return scores.argsort()[-top_n:][
            ::-1
        ]  # sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
