import os

import joblib
from tqdm import tqdm

from ..utils import load_config, logger
from .generators.dense_embeder import CohereEmbeder
from .generators.lexical_embeder import TDIDFLexicalEmbeder
from .utils import load_data_from_csv

ARTIFACTS_SAVE_PATH = os.getenv("ARTIFACTS_SAVE_PATH", "")
DATA_PATH = os.getenv("DATA_PATH", "")
items = load_data_from_csv(DATA_PATH)

config = load_config()
cohere_max_batch = config["retrievers"]["dense"]["max_processing_batch"]
COLUMNS_TO_EMBED = config["retrievers"]["columns_to_embed"]

INTERMEDIATE_SAVE_PATH = "dense_embeddings_partial.joblib"


async def embedd_products() -> None:
    """Bulk embed product data using both lexical (TF-IDF) and dense (Cohere) methods,
    saving intermediate and final artifacts.

    This function performs the following steps:
    1. Computes and saves lexical embeddings for specified columns using a TF-IDF-based embeder.
    2. Computes dense embeddings in batches using the Cohere API to respect quota limits, saving intermediate results
    to disk after each batch.
    3. Indexes and saves the dense embeddings if all batches complete successfully, and removes the intermediate save
    file.
    4. Logs progress and success messages throughout the process.
    """
    logger.info("Running lexical embedding...")
    lexical_embedder = TDIDFLexicalEmbeder(ARTIFACTS_SAVE_PATH)
    vectors = lexical_embedder.fit_transform(items, COLUMNS_TO_EMBED)
    lexical_embedder.save(vectors)

    logger.info("Running dense embedding...")
    embedder = CohereEmbeder()
    dense_embeddings = []

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
    logger.info("Success...")
