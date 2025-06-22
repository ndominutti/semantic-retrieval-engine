import yaml
from utils import load_config
import logging
from typing import Union, List, Tuple

config = load_config()
logging.warning(config)
LEXICAL_ALPHA = config["scorer"]["lexical_score_mixture_alpha"]


def score_mixture(
    lexical_scores, dense_scores, top_n, return_score=False
) -> Union[List[int], Tuple[List[int], None]]:
    """Combines lexical and dense scores using a weighted sum and returns the indices of the top-scoring items.

    Args:
        lexical_scores (np.ndarray): Array of scores from a lexical model.
        dense_scores (np.ndarray): Array of scores from a dense model.
        top_n (int): Number of top items to return.
        return_score (bool, optional): If True, also returns the corresponding scores. Defaults to False.

    Returns:
        Union[List[int], Tuple[List[int], np.ndarray]]:
            If return_score is False, returns a list of indices of the top-scoring items.
            If return_score is True, returns a tuple containing the list of indices and their corresponding scores.
    """
    scores = LEXICAL_ALPHA * lexical_scores + (1 - LEXICAL_ALPHA) * dense_scores
    top_ids = scores.argsort()[-top_n:][::-1]
    if return_score:
        return top_ids.tolist(), scores[top_ids]
    return top_ids.tolist()
