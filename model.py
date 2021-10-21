"""
+-------------+
|    MODEL    |
+-------------+
"""
from timeit import default_timer as timer
from typing import Tuple

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from encode_dialogs import get_dialogs
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
    dialogs = dialogs[:2000]
    replies = replies[:2000]
    logger.debug(f"Length of 'DIALOGS': {len(dialogs)}")
    logger.debug(f"Length of 'REPLIES' :{len(replies)}")

    tokenizer = get_tokenizer(dialogs, logger, replies)
    word2idx = tokenizer.word_index
    idx2word = {v: k for k, v in word2idx.items()}

    logger.debug("Encoding the dialogs and thier replies...")
    encoded_dialogs = tokenizer.texts_to_sequences(dialogs)
    encoded_replies = tokenizer.texts_to_sequences(replies)
    max_len_dialog = np.max([len(_) for _ in encoded_replies + encoded_dialogs])

    logger.info("Initializing embedding...")
    embedder = GloVeEmbedding()

    embed_matrix, embed_vector_len, vocab_length = get_embedding_matrix(embedder, logger, word2idx)

    logger.debug(f"Padding to equalize to length of {max_len_dialog}...")
    encoder_dialogs_padded = np.array(pad_sequences(encoded_dialogs, maxlen=max_len_dialog, padding='post'))
    encoder_replies_padded = np.array(pad_sequences(encoded_replies, maxlen=max_len_dialog, padding='post'))
    onehot_replies = to_categorical(encoder_replies_padded, vocab_length)
    decoder_ops = np.array(onehot_replies)
    logger.debug(f"Shape of padded dialogs = {encoder_dialogs_padded.shape}")
    logger.debug(f"Shape of padded replies = {encoder_replies_padded.shape}")
    logger.debug(f"Shape of padded replies = {decoder_ops.shape}")

    logger.info("Creating a Embedding Layer.")
    max_len = 150
    embedding_layer = Embedding(input_dim=vocab_length, output_dim=embed_vector_len,
                                input_length=max_len, weights=[embed_matrix],
                                trainable=False)

    logger.info("Creating Model...")
    start = timer()
    encoder_input = Input(shape=(None,))
    encoder_embedding = embedding_layer(encoder_input)
    encoder_output, state_h, state_c = LSTM(200, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(None,))
    decoder_embedding = embedding_layer(decoder_input)
    decoder_lstm = LSTM(200, return_state=True, return_sequences=True)
    decoder_replies, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_length, activation=softmax)
    output = decoder_dense(decoder_replies)

    model = Model([encoder_input, decoder_input], output)
    model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')
    logger.debug(f"Time taken to create a model is {timer() - start} seconds.")

    model.summary()
    model.fit([encoder_dialogs_padded, encoder_replies_padded], decoder_ops, batch_size=50, epochs=10)
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
