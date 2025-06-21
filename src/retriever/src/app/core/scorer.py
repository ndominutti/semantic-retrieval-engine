import yaml
from utils import load_config
import logging
from typing import Union, List, Tuple

config = load_config()
logging.warning(config)
LEXICAL_ALPHA = config["scorer"]["lexical_score_mixture_alpha"]
TOP_N = config["retrievers"]["top_n"]


def score_mixture(
    lexical_scores, dense_scores, return_score=False
) -> Union[List[int], Tuple[List[int], None]]:
    # dense score is a distance, HOW TO TURN IT COMPARABLE WITH COSINE SIMILARITY?
    scores = dense_scores  # LEXICAL_ALPHA * lexical_scores + (1 - LEXICAL_ALPHA) * dense_scores
    top_ids = scores.argsort()[-TOP_N:][::-1]
    if return_score:
        return top_ids.tolist(), scores[top_ids]
    return top_ids.tolist()
