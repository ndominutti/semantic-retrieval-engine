import os

import faiss

ARTIFACTS_SAVE_PATH = os.getenv("ARTIFACTS_SAVE_PATH")


def load_index():
    return faiss.read_index(ARTIFACTS_SAVE_PATH + "products_index.faiss")
