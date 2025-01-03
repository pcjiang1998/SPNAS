import os
import shutil


def folder_create(path):
    if os.path.isfile(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)
    return path
