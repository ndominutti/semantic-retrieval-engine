from .generators.dense_embedder import CohereEmbedder
from .generators.lexical_embedder import TDIDFLexicalEmbedder
from .utils import load_data_from_csv
from utils import load_config
import os
import logging
from tqdm import tqdm

ARTIFACTS_SAVE_PATH = os.getenv("ARTIFACTS_SAVE_PATH")
DATA_PATH = os.getenv("DATA_PATH")
items = load_data_from_csv(DATA_PATH)
COLUMNS_TO_EMBED = [
    col.strip() for col in os.getenv("COLUMNS_TO_EMBED").split(",") if col.strip()
]
config = load_config()
cohere_max_batch = config["retrievers"]["dense"]["max_processing_batch"]
import joblib

INTERMEDIATE_SAVE_PATH = "dense_embeddings_partial.joblib"


async def embedd_products() -> None:
    logging.info("Running lexical embedding...")
    lexical_embedder = TDIDFLexicalEmbedder(ARTIFACTS_SAVE_PATH)
    vectors = lexical_embedder.fit_transform(items, COLUMNS_TO_EMBED)
    lexical_embedder.save(vectors)

    logging.info("Running dense embedding...")
    embedder = CohereEmbedder()
    dense_embeddings = []
    import time

    # need to batch to follow Cohere quotas
    for batch_start in tqdm(range(0, items.shape[0], cohere_max_batch)):
        batch_end = batch_start + cohere_max_batch
        batch_items = items[batch_start:batch_end]
        embeddings = await embedder.embed(
            products_df=batch_items,
            cols_to_embed=COLUMNS_TO_EMBED,
        )
        dense_embeddings.extend(embeddings)
        # save in case the api fails
        joblib.dump(dense_embeddings, INTERMEDIATE_SAVE_PATH)
    embedder.index_and_save(dense_embeddings, ARTIFACTS_SAVE_PATH)
    # remove saving if indexing was successful
    os.remove(INTERMEDIATE_SAVE_PATH)
