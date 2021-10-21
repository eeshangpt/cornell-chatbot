"""
+------------------------+
|    BATCH GENERATION    |
+------------------------+
"""
import pickle
from os.path import isfile
from typing import Tuple

from encode_dialogs import get_encoded_dialogs
from utils.basic_utilities import *

BATCH_SIZE = 64


def get_encoded_dialog_to_process(logger_: logging.Logger, batch_size: int = BATCH_SIZE) -> Tuple[List, List]:
    """
    Fetches dialogs from the files and pickles it. If the pickle is already present, returns the object.
    """
    logger = logger_.getChild("__get_encoded_dialog_to_process")
    pickled_encoded_dialog_file = join(DATA_DIR, "all_encoded_dialog.pkl")
    logger.debug(f"Finding file @ {pickled_encoded_dialog_file}")
    try:
        assert not isfile(pickled_encoded_dialog_file)
        logger.warning("File not found.")
        logger.info("Creating object")
        encoded_dialogs, encoded_replies = get_encoded_dialogs(logger)
        logger.info("Pickling file for future use.")
        with open(pickled_encoded_dialog_file, 'wb') as f:
            pickle.dump((encoded_dialogs, encoded_replies), f)
    except AssertionError:
        logger.info("File found. Reading object.")
        with open(pickled_encoded_dialog_file, 'rb') as f:
            encoded_dialogs, encoded_replies = pickle.load(f)

    dialogs, replies = [], []
    for itr, (d, r) in enumerate(zip(encoded_dialogs, encoded_replies)):
        if itr % batch_size == 0:
            if itr != 0:
                logger.debug(f"Yielding batch of sizes: ({len(dialogs)}, {len(replies)})...")
                yield dialogs, replies
            dialogs, replies = [], []
        dialogs.append(d)
        replies.append(r)

# def driver(logger_main_: logging.Logger) -> None:
#     """
#     Drives the current logic.
#     """
#     logger = logger_main_.getChild("driver")
#     read_data_and_create_dialog(logger)
#
#     dataset = get_encoded_dialog_to_process(logger)
#     for i, j in dataset:
#         logger.debug(f"dialog_batch_size: {len(i)}, reply_batch_size: {len(j)}")
#     return None
# if __name__ == '__main__':
#     logger_main = logging.getLogger("BATCH_GENERATION")
#     logging.basicConfig(**get_config(logging.DEBUG,
#                                      file_logging=False,
#                                      # filename="batch-generation",
#                                      stop_stream_logging=False))
#     logger_main.critical(__doc__)
#     driver(logger_main)
#     logger_main.critical(end_line.__doc__)
