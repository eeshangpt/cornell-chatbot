import logging
from datetime import datetime


def get_unique_file_name():
    """
    Method returns TimeStamp as a string.
    """
    a = datetime.now()
    return str(a.year) + "_" + str(a.month) + "_" + str(a.day) + "_" + str(a.hour) + "_" + str(a.minute)


def end_line():
    """
+--------------------------+
|    THAT'S ALL FOLKS!!    |
+--------------------------+
    """
    pass


def get_handlers(file_logging=True, filename='', stop_stream_logs=False):
    """
    """

    handlers = []
    if not stop_stream_logs:
        handlers.append(logging.StreamHandler())
    if file_logging and (filename != ''):
        handlers.append(logging.FileHandler("D:\\Projects\\CombinedClassification\\logs\\{}.log".format("_".join([
            get_unique_file_name(), filename]))))
    return handlers


def get_config(level=logging.DEBUG, file_logging=True, filename='', stop_stream_logging=False):
    """
    """
    if file_logging:
        if filename != '':
            config = {
                'level': level,
                'format': '[%(asctime)-5s] [%(name)-10s] [%(levelname)-8s]: %(message)s',
                'handlers': get_handlers(file_logging, filename, stop_stream_logging)
            }
        else:
            config = {
                'level': level,
                'format': '[%(asctime)-5s] [%(name)-10s] [%(levelname)-8s]: %(message)s',
                'handlers': get_handlers(file_logging, 'generic', stop_stream_logging)
            }
    else:
        config = {
            'level': level,
            'format': '[%(asctime)-5s] [%(name)-10s] [%(levelname)-8s]: %(message)s',
            'handlers': get_handlers(file_logging)
        }
    return config
