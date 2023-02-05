import os
import shutil


def dir_func(path: str, rmtree: bool = True, mkdir: bool = True):
    if rmtree:
        shutil.rmtree(path, ignore_errors=True)
    if mkdir:
        if not os.path.exists(path):
            os.makedirs(path)


def classifier():
    pass