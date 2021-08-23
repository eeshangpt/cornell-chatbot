"""
+------------------------+
|    BATCH GENERATION    |
+------------------------+
"""
import pickle
from os.path import isfile

from encode_dialogs import get_encoded_dialogs
from read_dataset import read_data_and_create_dialog
from utils.basic_utilities import *


def __get_encoded_dialog_to_process(logger_: logging.Logger) -> List:
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
        encoded_dialogs = get_encoded_dialogs(logger)

        logger.info("Pickling file for future use.")
        with open(pickled_encoded_dialog_file, 'wb') as f:
            pickle.dump(encoded_dialogs, f)
    except AssertionError:
        logger.info("File found. Reading object.")
        with open(pickled_encoded_dialog_file, 'rb') as f:
            encoded_dialogs = pickle.load(f)
    return encoded_dialogs


def driver(logger_main_: logging.Logger) -> None:
    """
    Drives the current logic.
    """
    logger = logger_main_.getChild("main")
    read_data_and_create_dialog(logger)

    encoded_dialogs = __get_encoded_dialog_to_process(logger)
    logger.debug(f"{len(encoded_dialogs)} dialogs found.")

    return None


if __name__ == '__main__':
    logger_main = logging.getLogger("BATCH_GENERATION")
    logging.basicConfig(**get_config(logging.DEBUG,
                                     file_logging=False,
                                     # filename="batch-generation",
                                     stop_stream_logging=False))
    logger_main.critical(__doc__)
    driver(logger_main)
    logger_main.critical(end_line.__doc__)
