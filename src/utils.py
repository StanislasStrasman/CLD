import os
import sys
import logging
from logging import getLogger
from typing import Tuple
from datetime import datetime
import shutil


####### Create Logger ########    

def create_logger_bis() -> Tuple[logging.Logger, str]:
    logger = getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir: str = f'./results/{timestamp}_train'
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, 'train.log')

        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            fmt='[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger, log_dir


def create_logger(result_folder: str) -> None:

    def get_result_folder() -> str:
        return result_folder
    
    def set_result_folder(folder: str) -> None:
        global result_folder
        result_folder = folder

    log_desc: str = 'train'
    log_filename: str = 'run_log'

    log_filepath: str = get_result_folder().format(desc='_' + log_desc)
    set_result_folder(log_filepath)

    if not os.path.exists(log_filepath):
        os.makedirs(log_filepath)

    filename: str = log_filepath + '/' + log_filename

    file_mode: str = 'a' if os.path.isfile(filename) else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


