import os
import sys

from loguru import logger

ENV = os.getenv("ENV", "")
LOG_LEVEL = "DEBUG" if ENV == "dev" else "INFO"

logger.remove()
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)
