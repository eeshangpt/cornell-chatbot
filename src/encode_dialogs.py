from read_dataset import read_data_and_create_dialog
from utils.basic_utilities import *


def main(logger_main: logging.Logger):
    logger = logger_main.getChild("main")
    logger.info("Starting encoding...")

    return None


if __name__ == '__main__':
    logger_main = logging.getLogger("ENCODE_DIALOGS")
    logging.basicConfig(**get_config(logging.DEBUG, file_logging=False))
    read_data_and_create_dialog(logger_main)
    main(logger_main)
    print(end_line.__doc__)
