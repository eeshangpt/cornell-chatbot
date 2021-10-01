"""
+-------------+
|    MODEL    |
+-------------+
"""
from os.path import isfile
from pickle import load, dump
from timeit import default_timer as timer
from typing import Tuple

from tensorflow.keras.preprocessing.text import Tokenizer

from encode_dialogs import get_encoded_dialogs
from read_dataset import read_data_and_create_dialog
from utils.basic_utilities import *
from utils.embedding_utilities import GloVeEmbedding

# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import Embedding

DIALOG_FROM_PCSD_PICKLE = join(DATA_DIR, "dialog.from.processed.pkl")
DIALOG_TO_PCSD_PICKLE = join(DATA_DIR, "dialog.to.processed.pkl")


def driver(logger_main_: logging.Logger) -> None:
    logger = logger_main_.getChild("driver")
    dialogs, replies = __get_dialogs(logger)
    logger.debug(f"Length of 'DIALOGS': {len(dialogs)}")
    logger.debug(f"Length of 'REPLIES' :{len(replies)}")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dialogs + replies)
    word2idx = tokenizer.word_index
    idx2word = {v: k for k, v in word2idx.items()}
    embedder = GloVeEmbedding()
    return None


def __get_dialogs(logger_: logging.Logger) -> Tuple[List, List]:
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


if __name__ == '__main__':
    logger_main = logging.Logger("MODEL")
    logging.basicConfig(**get_config(logging.DEBUG,
                                     file_logging=False,
                                     # filename="batch-generation",
                                     stop_stream_logging=False))
    logger_main.critical(__doc__)
    driver(logger_main)
    logger_main.critical(end_line.__doc__)
