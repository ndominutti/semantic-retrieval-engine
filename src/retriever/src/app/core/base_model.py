from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np


class RetrievalBase(ABC):
    """Base class for retrievers"""

    @abstractmethod
    def score(
        self, query, *args, **kwargs
    ) -> Union[np.ndarray, Tuple[List[float], List[int]]]:
        pass

    @abstractmethod
    def retrieve(self, query, *args, **kwargs) -> Union[np.ndarray, List[int]]:
        pass
