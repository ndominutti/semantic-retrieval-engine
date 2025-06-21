from abc import ABC, abstractmethod


class RetrievalBase(ABC):
    @abstractmethod
    def score(self, query, *args, **kwargs):
        pass

    @abstractmethod
    def retrieve(self, query, *args, **kwargs):
        pass
