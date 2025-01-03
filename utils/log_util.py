import glob
import logging
import os
import shutil
import time

from tensorboardX import SummaryWriter

from utils.io_util import folder_create


def _setup_logger(logger_name: str, log_file: str, level=logging.INFO):
    logger_setter = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s, %(message)s', datefmt='%Y-%m-%d, %H:%M:%S')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger_setter.setLevel(level)
    logger_setter.addHandler(file_handler)
    logger_setter.addHandler(stream_handler)


def copy_pys(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def init_a_logger(name: str, folder=None, time_id: bool = True):
    if time_id:
        t = '-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.log'
    else:
        t = '.log'
    if folder:
        assert os.path.isdir(folder)
        log_file = os.path.join(folder, name + t)
    else:
        log_file = name + t
    _setup_logger(name, log_file)
    logger = logging.getLogger(name)
    return logger


def init_a_SummaryWriter(board_folder: str, folder=None, time_id: bool = True) -> SummaryWriter:
    if time_id:
        t = '-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        t = ''
    if folder:
        assert os.path.isdir(folder)
        board_folder = os.path.join(folder, board_folder + t)
    else:
        board_folder = board_folder + t
    writer = SummaryWriter(board_folder)
    return writer


class Logger:
    _logger = None
    one_run_folder = None

    @staticmethod
    def get_instance():
        if Logger._logger is None:
            raise Exception('Not initialized')
        return Logger._logger

    def __init__(self, name: str, folder=None, time_id: bool = True):
        if Logger._logger is None:
            t = '-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) if time_id else ''
            one_run_folder = folder_create(os.path.join(folder, name + t)) if folder else folder_create(name + t)
            Logger._logger = init_a_logger(name, one_run_folder, time_id)
            copy_pys(one_run_folder, scripts_to_save=glob.glob('*.py'))
            Logger.one_run_folder = one_run_folder

    @staticmethod
    def info(s: str):
        Logger._logger.info(s)

    @staticmethod
    def get_folder():
        if Logger._logger is None:
            raise Exception('Not initialized')
        return Logger.one_run_folder
