from os.path import isabs, join
import sys
import logging
import subprocess
import json

logger = logging.getLogger(__name__)

try:
    subprocess.check_output(
        '{} -c "import numba"'.format(sys.executable), shell=True
    )
    logging.info('numba available, importing jit')
    from numba import jit
except Exception:
    logging.warning('cannot import numba, creating dummy jit definition')

    def jit(function):
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)
        return wrapper


def compose_path(file_path, reference_path):
    if reference_path and not isabs(file_path):
        file_path = join(reference_path, file_path)
    return file_path


def config_get_list(config, key, fallback=None):
    """Get list from config."""
    fallback = fallback if fallback is not None else []

    if key not in config:
        return fallback

    try:
        value = json.loads(config[key])
    except json.decoder.JSONDecodeError as e:
        logger.warning(f'Failed to parse list from config key "{key}", returning fallback {fallback}:\n{e}')
        return fallback
    else:
        return value
