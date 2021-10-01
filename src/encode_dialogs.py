"""
+----------------------+
|    ENCODE DIALOGS    |
+----------------------+
"""
import re
from string import punctuation
from typing import Tuple

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

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


def decontracted(phrase: str) -> str:
    """
    Expands words which are contracted.
    """
    phrase = re.sub(r" won\'t ", "will not", phrase)
    phrase = re.sub(r" can\'t ", "can not", phrase)

    phrase = re.sub(r"n\'t ", "not ", phrase)
    phrase = re.sub(r" \'re ", " are ", phrase)
    phrase = re.sub(r" \'s ", " is ", phrase)
    phrase = re.sub(r" \'d ", " would ", phrase)
    phrase = re.sub(r" \'ll ", " will ", phrase)
    phrase = re.sub(r" \'t ", " not ", phrase)
    phrase = re.sub(r" \'ve ", " have ", phrase)
    phrase = re.sub(r" \'m ", " am ", phrase)
    return phrase


def get_encoded_dialog(embedding: GloVeEmbedding, tokens: List):
    """
    Encode the dialog.
    """
    return ["<START>"] + [embedding.get(token) for sent in tokens for token in sent] + ["<END>"]


def get_cleaned_dialog(tokens: List) -> str:
    """
    Encode the dialog.
    """
    return decontracted(" ".join(["<START>"] + [token for sent in tokens for token in sent] + ["<END>"]))


def get_encoded_dialogs(logger_main_: logging.Logger) -> Tuple[List, List]:
    logger = logger_main_.getChild("get_encoded_dialogs")
    logger.info("Starting Encoding ")

    # logger.info("Initializing Embedding Objects.")
    # embedding = GloVeEmbedding(embedding_dir=EMBEDDING_DIR, default_dim_index=2)

    logger.debug("Reading files.")
    dialogs_file = open(join(DATA_DIR, "dialog.from"))
    replies_file = open(join(DATA_DIR, "dialog.to"))

    logger.info("Start of encoding...")
    logger.debug("Encoding first dialogs...")
    # dialogs = [get_encoded_dialog(embedding, list(clean_and_tokenize(dialog)))
    #            for dialog in dialogs_file]
    dialogs = [get_cleaned_dialog(list(clean_and_tokenize(dialog)))
               for dialog in dialogs_file]
    logger.debug("Encoding first dialogs completed.")
    logger.debug("Encoding reply dialogs...")
    # replies = [get_encoded_dialog(embedding, list(clean_and_tokenize(dialog)))
    #            for dialog in replies_file]
    replies = [get_cleaned_dialog(list(clean_and_tokenize(dialog)))
               for dialog in replies_file]
    logger.debug("Encoding reply dialogs completed.")
    logger.info("Encoding completed.")

    logger.debug("Closing files.")
    dialogs_file.close()
    replies_file.close()

    return dialogs, replies
