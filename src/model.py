"""
+-------------+
|    MODEL    |
+-------------+
"""
from os.path import isfile
from pickle import load, dump
from timeit import default_timer as timer
from typing import Tuple

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

from encode_dialogs import get_encoded_dialogs
from read_dataset import read_data_and_create_dialog
from utils.basic_utilities import *
from utils.embedding_utilities import GloVeEmbedding

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input, LSTM, Activation, Dropout, Dense
from tensorflow.keras import Model

DIALOG_FROM_PCSD_PICKLE = join(DATA_DIR, "dialog.from.processed.pkl")
DIALOG_TO_PCSD_PICKLE = join(DATA_DIR, "dialog.to.processed.pkl")


def __get_dialogs(logger_: logging.Logger) -> Tuple[List, List]:
    """
    """
    logger = logger_.getChild("__get_dialogs")
    logger.info("Attempting to read pickles of processed dialogs and replies.")
    start = timer()
    try:
        assert not isfile(DIALOG_FROM_PCSD_PICKLE)
        assert not isfile(DIALOG_TO_PCSD_PICKLE)
        logger.debug("File not found. Therefore processing dialogs and their replies.")
        read_data_and_create_dialog(logger)
        dialogs, replies = get_encoded_dialogs(logger)
        with open(DIALOG_FROM_PCSD_PICKLE, 'wb') as f:
            dump(dialogs, f)
        with open(DIALOG_TO_PCSD_PICKLE, 'wb') as f:
            dump(replies, f)
    except AssertionError:
        logger.warning("Files already present.")
        logger.debug("File found. Now reading them.")
        with open(DIALOG_FROM_PCSD_PICKLE, 'rb') as f:
            dialogs = load(f)
        with open(DIALOG_TO_PCSD_PICKLE, 'rb') as f:
            replies = load(f)
        logger.info("Files read successfully.")
    logger.debug(f"Dialogs found in {timer() - start} seconds.")
    return dialogs, replies


def get_tokenizer(dialogs: List, logger: logging.Logger, replies: List) -> Tokenizer:
    """

    """
    logger.info("Initializing tokenizer...")
    start = timer()
    tokenizer = Tokenizer()
    logger.debug("Fitting the tokenizer.")
    tokenizer.fit_on_texts(dialogs + replies)
    logger.debug(f"Tokenizer initialzed in {timer() - start} seconds.")
    return tokenizer


def driver(logger_main_: logging.Logger) -> None:
    logger = logger_main_.getChild("driver")
    dialogs, replies = __get_dialogs(logger)
    logger.debug(f"Length of 'DIALOGS': {len(dialogs)}")
    logger.debug(f"Length of 'REPLIES' :{len(replies)}")

    tokenizer = get_tokenizer(dialogs, logger, replies)
    word2idx = tokenizer.word_index
    idx2word = {v: k for k, v in word2idx.items()}

    logger.info("Initializing embedding...")
    embedder = GloVeEmbedding()
    # word_to_vec_map = embedder.embeddings

    logger.info("Creating weights for Embedding Layer.")
    start = timer()
    vocab_length = len(word2idx) + 1
    embed_vector_len = embedder.dimension
    embed_matrix = np.zeros((vocab_length, embed_vector_len))
    for word, index in word2idx.items():
        embed_matrix[index, :] = embedder.get(word)
    logger.debug(f"Time taken to create weight {timer() - start} seconds.")

    logger.info("Creating a ")
    max_len = 150
    embedding_layer = Embedding(input_dim=vocab_length, output_dim=embed_vector_len,
                                input_length=max_len, weights=[embed_matrix],
                                trainable=False)

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
