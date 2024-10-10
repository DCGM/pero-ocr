from os.path import isabs, join
import sys
import logging
import subprocess

try:
    subprocess.check_output(
        '{} -c "import numba"'.format(sys.executable), shell=True
    )
    logging.info('numba available, importing jit')
    from numba import jit
except Exception:
    logging.warning('cannot import numba, creating dummy jit definition')

    def jit(*_, **__):
        def decorator(function):
            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)
            return wrapper
        return decorator


def compose_path(file_path, reference_path):
    if reference_path and not isabs(file_path):
        file_path = join(reference_path, file_path)
    return file_path
