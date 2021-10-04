"""
+-------------+
|    MODEL    |
+-------------+
"""
from timeit import default_timer as timer
from typing import Tuple

import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer

from src.encode_dialogs import get_dialogs
from utils.basic_utilities import *
from utils.embedding_utilities import GloVeEmbedding


def get_tokenizer(dialogs: List, logger: logging.Logger, replies: List) -> Tokenizer:
    """
    Initialize tokenizer.
    """
    logger.info("Initializing tokenizer...")
    start = timer()
    tokenizer = Tokenizer()
    logger.debug("Fitting the tokenizer.")
    tokenizer.fit_on_texts(dialogs + replies)
    logger.debug(f"Tokenizer initialzed in {timer() - start} seconds.")
    return tokenizer


def get_embedding_matrix(embedder: GloVeEmbedding, logger_: logging.Logger,
                         word2idx: Dict) -> Tuple[np.ndarray, int, int]:
    """
    Creates a weight matrix.
    """
    logger = logger_.getChild("get_embedding_matrix")
    logger.info("Creating weights for Embedding Layer.")
    start = timer()
    vocab_length = len(word2idx) + 1
    embed_vector_len = embedder.dimension
    embed_matrix = np.zeros((vocab_length, embed_vector_len))
    for word, index in word2idx.items():
        embed_matrix[index, :] = embedder.get(word)
    logger.debug(f"Time taken to create weight {timer() - start} seconds.")
    return embed_matrix, embed_vector_len, vocab_length


def driver(logger_main_: logging.Logger) -> None:
    """
    Driver.
    """
    logger = logger_main_.getChild("driver")
    dialogs, replies = get_dialogs(logger)
    logger.debug(f"Length of 'DIALOGS': {len(dialogs)}")
    logger.debug(f"Length of 'REPLIES' :{len(replies)}")

    tokenizer = get_tokenizer(dialogs, logger, replies)
    word2idx = tokenizer.word_index
    idx2word = {v: k for k, v in word2idx.items()}

    logger.info("Initializing embedding...")
    embedder = GloVeEmbedding()

    embed_matrix, embed_vector_len, vocab_length = get_embedding_matrix(embedder, logger, word2idx)

    logger.info("Creating a Embedding Layer.")
    max_len = 150
    embedding_layer = Embedding(input_dim=vocab_length, output_dim=embed_vector_len,
                                input_length=max_len, weights=[embed_matrix],
                                trainable=False)

    logger.info("Creating Model...")
    start = timer()
    logger.debug(f"Time taken to create a model is {timer() - start} seconds.")

    return None


if __name__ == '__main__':
    logger_main = logging.Logger("MODEL")
    logging.basicConfig(**get_config(logging.DEBUG,
                                     file_logging=False,
                                     # filename="batch-generation",
                                     stop_stream_logging=False))
    logger_main.critical(__doc__)
    driver(logger_main)
    logger_main.critical(end_line.__doc__)
