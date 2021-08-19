"""
+---------------------------+
|    EMBEDDING UTILITIES    |
+---------------------------+
"""
import pickle
from os.path import isfile

import numpy as np

from utils.basic_utilities import *


def main(logger_: logging.Logger) -> None:
    logger = logger_.getChild("main")
    available_dimensions = [25, 50, 100, 200]
    dimension = available_dimensions[2]  # TODO: make it either dynamics or user choice.
    logger.info(f"Choosing {dimension} dimensions.")
    embedding_file_path = join(EMBEDDING_DIR, f"glove.twitter.27B.{dimension}d.txt")
    logger.debug(f"File found @ {embedding_file_path}")
    pickle_file_path = join(EMBEDDING_DIR, f"glove_embedding_{dimension}d.pkl")
    try:
        assert not isfile(pickle_file_path)
        logger.info("Getting the embedding file.")
        with open(embedding_file_path, 'r') as f:
            embeddings = {i[0]: i[1] for i in
                          map(lambda line: [line.strip().split()[0], np.array(list(map(float,
                                                                                       line.strip().split()[1:])))],
                              f.readlines())}
        logger.info("File read.")
        logger.debug(f"Writing the embedding object as pickle for future use.")
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(embeddings, f)
    except AssertionError:
        logger.debug("Embedding serialized object already present.")
        logger.info("Reading the object file.")
        with open(pickle_file_path, "rb") as f:
            embeddings = pickle.load(f)
    logger.debug(f"Embedding for {len(embeddings)} words found.")

    return None


if __name__ == '__main__':
    logger_main = logging.getLogger("ENCODE_DIALOGS")
    logging.basicConfig(**get_config(logging.DEBUG,
                                     file_logging=False,
                                     # filename="encode-dialog",
                                     stop_stream_logging=False))
    logger_main.critical(__doc__)
    main(logger_main)
    logger_main.critical(end_line.__doc__)
