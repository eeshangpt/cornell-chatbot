import pickle
from os import walk
from os.path import isfile
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm

from utils.basic_utilities import *

ENCODING = 'iso-8859-1'  # Encoding for the Cornell Dataset.
SEPERATOR = " +++$+++ "  # Seperator as specified in readme.


def __read_file_lines(file_path: str, logger: logging.Logger) -> List:
    """
    Read line from the file.
    """
    logger_ = logger.getChild("__read_file_lines")
    logger_.debug(f"Reading file {file_path} and fetching each line.")
    with open(file_path, 'r', encoding=ENCODING) as f:
        lines = f.readlines()
    logger_.debug(f"File read, returning the list of lines")
    logger_.debug(f"Length of files read = {len(lines)}")
    return lines


def __modifiy_conversations(file_content: Dict, logger: logging.Logger) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Modify conversations.
    """
    logger_ = logger.getChild("__modifiy_conversations")
    logger_.debug(f"Found {len(file_content)} data frames.")
    logger_.debug(f"Starting modification of conversation dataframe...")
    file_content['movie_conversations.txt'][3] = file_content['movie_conversations.txt'][3].apply(
        lambda x: x[1:-1].split(',')).apply(lambda x: [_.strip()[1:-1] for _ in x])
    logger_.debug("Selecting the conversation Series.")
    file_content['movie_conversations.txt'] = file_content['movie_conversations.txt'][3]
    logger_.debug("Modification 1 completed.")

    logger_.debug("Starting modiifcation of lines dataframe.")
    file_content['movie_lines.txt'] = file_content['movie_lines.txt'][[0, 4]]
    logger_.debug("Modification 2 completed.")

    logger_.debug(f"Shape of conversation dataframe = {file_content['movie_conversations.txt'].shape}")
    logger_.debug(f"Shape of lines dataframe = {file_content['movie_lines.txt'].shape}")
    return file_content['movie_conversations.txt'], file_content['movie_lines.txt']


def __get_dialog_and_reply(conversations: pd.Series, lines: pd.DataFrame, logger: logging.Logger) -> Tuple[List, List]:
    """
    Get the dialog and its reply.
    """
    logger_ = logger.getChild("__get_dialog_and_reply")
    dialog, reply_dialog = [], []
    logger_.debug(f"Starting separating the dialog and its reply.")
    for conversation in tqdm(conversations):
        for i in range(1, len(conversation)):
            dialog.append(lines[lines[0] == conversation[i - 1]][4].values[0])
            reply_dialog.append(lines[lines[0] == conversation[i]][4].values[0])
    logger_.debug(f"Length of dialog = {len(dialog)}")
    logger_.debug(f"Length of reply_dialog = {len(reply_dialog)}")
    return dialog, reply_dialog


def __create_dialogs_replies(conversations: pd.Series, lines: pd.DataFrame, output_pickle_file: str,
                             logger: logging.Logger) -> Tuple[List, List]:
    """
    Creates dialog and reply.
    """
    logger_ = logger.getChild("__create_dialogs_replies")
    try:
        logger_.info("Attempting to read the pickle of dialogs and thier replies.")
        assert isfile(output_pickle_file)
        with open(output_pickle_file, 'rb') as f:
            dialog, reply_dialog = pickle.load(f)
        logger_.info("Pickle file found and objects restored.")
    except AssertionError:
        logger_.warning("Pickle file not found.")
        logger_.info("Creating the files.")
        dialog, reply_dialog = __get_dialog_and_reply(conversations, lines, logger_)
        logger_.info("Dialog and Reply Dialog objects created.")
        logger_.info("Pickling started.")
        with open(output_pickle_file, 'wb') as f:
            pickle.dump((dialog, reply_dialog), f)
    return dialog, reply_dialog


def __write_dialog_files(dialog: List, dialog_file_path: str, reply_dialog: List, reply_file_path: str,
                         logger: logging.Logger) -> None:
    """
    Creating files for dialog and its reply
    """
    logger_ = logger.getChild("__write_dialog_files")
    logger_.debug("Opening files.")
    dialog_file = open(dialog_file_path, 'w')
    reply_file = open(reply_file_path, 'w')
    logger_.debug("Staring the writing process...")
    for d, r in tqdm(zip(dialog, reply_dialog)):
        if d is not None and r is not None:
            dialog_file.write(d)
            reply_file.write(r)
        # else:
        #     logger.warning("Either `from` and `to` is None.")

    logger.info("File writing completed.")
    dialog_file.close()
    reply_file.close()


def read_data_and_create_dialog(logger_main: logging.Logger):
    """
    Driver.
    """
    logger = logger_main.getChild("read_data_and_create_dialog")
    logger.info("Reading the directory.")
    root, dirs, files = list(walk(DATA_SET_DIR))[0]
    logger.info("Separating the useful file.")
    data_files = {file: join(DATA_SET_DIR, file) for file in files if "movie" in file}
    logger.info("Creating DataFrame and selecting the conversations.")
    conversations, lines = __modifiy_conversations({file_name: pd.DataFrame([line.strip().split(SEPERATOR)
                                                                             for line in
                                                                             __read_file_lines(file_path, logger)])
                                                    for file_name, file_path in data_files.items()}, logger)

    output_pickle_file = join(DATA_DIR, "Dialog_Reply.pkl")
    dialog, reply_dialog = __create_dialogs_replies(conversations, lines, output_pickle_file, logger)

    logger.debug(f"Total number of Dialogs = {len(dialog)}")
    logger.debug(f"Total number of Replies = {len(reply_dialog)}")
    assert len(dialog) == len(reply_dialog)

    dialog_file_path = join(DATA_DIR, "dialog.from")
    reply_file_path = join(DATA_DIR, "dialog.to")
    try:
        assert (isfile(dialog_file_path) and isfile(reply_file_path))
        logger.warning("Files already present.")
    except AssertionError as e:
        logger.error(e)
        logger.info("Creating dialog and reply file.")
        __write_dialog_files(dialog, dialog_file_path, reply_dialog, reply_file_path, logger)

    logger.critical("DATA IS READ AND DIALOGS ARE CREATED.")
