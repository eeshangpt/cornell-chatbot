"""
+------------------------+
|    BATCH GENERATION    |
+------------------------+
"""

from encode_dialogs import get_encoded_dialogs
from read_dataset import read_data_and_create_dialog
from utils.basic_utilities import *

if __name__ == '__main__':
    logger_main = logging.getLogger("BATCH_GENERATION")
    logging.basicConfig(**get_config(logging.DEBUG,
                                     file_logging=False,
                                     # filename="batch-generation",
                                     stop_stream_logging=False))
    logger_main.critical(__doc__)
    read_data_and_create_dialog(logger_main)
    get_encoded_dialogs(logger_main)
    logger_main.critical(end_line.__doc__)
