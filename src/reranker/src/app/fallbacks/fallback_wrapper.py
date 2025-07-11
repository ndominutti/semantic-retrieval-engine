import asyncio
import functools
from typing import List, Tuple

from utils import logger


def async_error_handler_with_fallback(fallback=None, retries=0, delay=0):
    """
    Async decorator to wrap async methods with error handling and fallback.

    Args:
        fallback: Async function or sync function to call as fallback if the wrapped function fails.
                  If None, returns None on failure.
        retries: Number of retry attempts before fallback.
        delay: Delay in seconds between retries.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logger.warning(f"Error in {func.__name__}: {e}, attempt {attempts}")
                    if attempts > retries:
                        if fallback:
                            logger.warning(f"Using fallback for {func.__name__}")
                            if asyncio.iscoroutinefunction(fallback):
                                return await fallback(*args, **kwargs)
                            else:
                                return fallback(*args, **kwargs)
                        else:
                            logger.warning(
                                f"No fallback defined for {func.__name__}, returning None"
                            )
                            return None
                    if delay:
                        await asyncio.sleep(delay)

        return wrapper

    return decorator


def default_fallback_data_ids(*args, **kwargs) -> List[Tuple[int, float]]:
    """Placeholder function to handle errors.
    A good approach might be to recommend on item's popularity for fallback

    Returns:
        RerankIDResponse: a fallback response with items ids
    """
    return [(0, 0.0), (1, 0.0), (2, 0.0)]


def default_fallback_data_docs(*args, **kwargs) -> List[dict]:
    """Placeholder function to handle errors.
    A good approach might be to recommend on item's popularity for fallback

    Returns:
        RerankIDResponse: a fallback response with items ids
    """
    placeholder_response = [
        {
            "product_id": [0],
            "product_name": ["armchair"],
            "product_class": ["chairs"],
            "category_hierarchy": ["chairs"],
            "product_description": ["an armchair"],
            "product_features": [""],
            "rating_count": [10],
            "average_rating": [4.5],
            "review_count": [5],
            "score": [0],
        }
    ]
    return placeholder_response
