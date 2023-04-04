from os.path import isabs, join
import sys
import subprocess

try:
    from numba import jit
    print('numba available, importing jit')
except Exception:
    print('cannot import numba, creating dummy jit definition')

    def jit(function):
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)
        return wrapper


def compose_path(file_path, reference_path):
    if reference_path and not isabs(file_path):
        file_path = join(reference_path, file_path)
    return file_path
