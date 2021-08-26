import logging

from encode_dialogs import get_encoded_dialogs
from read_dataset import read_data_and_create_dialog
from utils.basic_utilities import *


def driver(logger_main_: logging.Logger):
    logger = logger_main_.getChild("driver")
    read_data_and_create_dialog(logger)
    dataset = get_encoded_dialogs(logger)


if __name__ == '__main__':
    logger_main = logging.Logger("MODEL")
    logging.basicConfig(**get_config(logging.DEBUG,
                                     file_logging=False,
                                     # filename="batch-generation",
                                     stop_stream_logging=False))
