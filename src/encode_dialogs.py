"""
+----------------------+
|    ENCODE DIALOGS    |
+----------------------+
"""
from string import punctuation
from typing import Any

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

from read_dataset import read_data_and_create_dialog
from utils.basic_utilities import *
from utils.embedding_utilities import GloVeEmbedding

nltk.download('punkt')


def __tokenize(line: str) -> List:
    """
    Tokenize first into sentences and then a sentence into words.
    """
    return [i for i in map(word_tokenize, sent_tokenize(line))]


def __cleaning_punctuations(word: str) -> str:
    """
    Removes punctuations from anywhere in a word.
    """
    word_ = word
    for char in word:
        if char in punctuation:
            word_ = word_[:word.index(char)] + word_[word.index(char) + 1:]
    return word_


def clean_and_tokenize(line: str) -> map:
    """
    Cleaning and tokenizing driver.
    """
    # return map(lambda sent: [__cleaning_punctuations(word) for word in sent if len(word) > 0], __tokenize(line))
    return map(lambda sent: [word for word in sent if len(word) > 0], __tokenize(line))


def main(logger_main_: logging.Logger) -> None:
    logger = logger_main_.getChild("main")
    logger.info("Starting Encoding ")

    logger.info("Initializing Embedding Objects.")
    embedding = GloVeEmbedding(embedding_dir=EMBEDDING_DIR, default_dim_index=2)

    dialogs_file = open(join(DATA_DIR, "dialog.from"))
    replies_file = open(join(DATA_DIR, "dialog.to"))
    for itr, dialogs in enumerate(zip(dialogs_file, replies_file)):
        # logger.debug(f"\n~~>{dialogs[0].strip()}\n~~>{dialogs[1].strip()}\n")

        # line_tokens = list(clean_and_tokenize(dialogs[0]))
        # reply_line_tokens = list(clean_and_tokenize(dialogs[1]))
        line = get_encoded_dialog(embedding, list(clean_and_tokenize(dialogs[0])))
        reply_line = get_encoded_dialog(embedding, list(clean_and_tokenize(dialogs[1])))
        # logger.debug(f"\n~~> {line}\n~~> {reply_line}\n")

        # if (itr + 1) % 1 == 0:
        #     break

    dialogs_file.close()
    replies_file.close()

    return None


def get_encoded_dialog(embedding: GloVeEmbedding, tokens: List):
    """
    Encode the dialog.
    """
    return ["<START>"] + [embedding.get(token) for sent in tokens for token in sent] + ["<END>"]


if __name__ == '__main__':
    logger_main = logging.getLogger("ENCODE_DIALOGS")
    logging.basicConfig(**get_config(logging.DEBUG,
                                     # file_logging=True, filename="encode-dialog",
                                     stop_stream_logging=False))
    logger_main.critical(__doc__)
    read_data_and_create_dialog(logger_main)
    main(logger_main)
    logger_main.critical(end_line.__doc__)
