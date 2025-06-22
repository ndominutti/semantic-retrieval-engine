from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple, List


class RetrievalBase(ABC):
    @abstractmethod
    def score(
        self, query, *args, **kwargs
    ) -> Union[np.ndarray, Tuple[List[float], List[int]]]:
        pass

    @abstractmethod
    def retrieve(self, query, *args, **kwargs) -> Union[np.ndarray, List[int]]:
        pass
