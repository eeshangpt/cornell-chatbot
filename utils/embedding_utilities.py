"""
+---------------------------+
|    EMBEDDING UTILITIES    |
+---------------------------+
"""
import pickle
from os.path import isfile

import numpy as np

from .basic_utilities import *


class GloVeEmbedding(object):
    """
    Embedding Class
    """

    def __init__(self, default_dim_index: int = 2, embedding_dir: str = EMBEDDING_DIR):
        """
        Constructor.
        """
        self.logger = logging.getLogger("GloVeEmbedding")
        self.available_dimensions = [25, 50, 100, 200]
        # TODO: either make it dynamics or select one based on strong evidence.
        self.dimension = self.available_dimensions[default_dim_index]
        self.embedding_file_name = f"glove.twitter.27B.{self.dimension}d.txt"
        self.embedding_dir = embedding_dir
        self.embeddings = None
        self.initialize_embedding_dictionary()

    def __get_embedding_dictionary(self, embedding_file_path: str):
        """
        Either Reading the text file and creating a embedding dictionary or reading a pickle of embedding dictionary.
        """
        logger = self.logger.getChild("__get_embedding_dictionary")
        pickle_file_path = join(self.embedding_dir, f"glove_embedding_{self.dimension}d.pkl")
        try:
            assert not isfile(pickle_file_path)
            logger.info("Getting the embedding file.")
            with open(embedding_file_path, 'r') as f:
                self.embeddings = {i[0]: i[1]
                                   for i in map(lambda line: [line.strip().split()[0],
                                                              np.array(list(map(float, line.strip().split()[1:])))],
                                                f.readlines())}
            logger.info("File read.")
            logger.debug(f"Writing the embedding object as pickle for future use.")
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
        except AssertionError:
            logger.debug("Embedding serialized object already present.")
            logger.info("Reading the object file.")
            with open(pickle_file_path, "rb") as f:
                self.embeddings = pickle.load(f)

    def initialize_embedding_dictionary(self):
        """
        Initializing the embedding dictionary.
        """
        logger = self.logger.getChild("initialize_embedding_dictionary")
        logger.info(f"Choosing {self.dimension} dimensions.")
        embedding_file_path = join(self.embedding_dir, self.embedding_file_name)
        logger.debug(f"File found @ {embedding_file_path}")
        self.__get_embedding_dictionary(embedding_file_path)
        logger.debug(f"Embedding for {len(self.embeddings)} words found.")

    def get(self, word: str) -> np.ndarray:
        if word.lower() in self.embeddings:
            return self.embeddings[word.lower()]
        return np.array([0] * self.dimension)

# if __name__ == '__main__':
#     logger_main = logging.getLogger("ENCODE_UTILITIES")
#     logging.basicConfig(**get_config(logging.DEBUG, file_logging=False,
#                                      filename="encode-dialog",
#                                      stop_stream_logging=False))
#     logger_main.critical(__doc__)
#     logger_main.critical(end_line.__doc__)#
