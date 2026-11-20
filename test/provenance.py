"""Record which code and library versions produced committed test data."""
import json
import os
import platform
import subprocess

import numpy
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_generated_with(directory):
    """Write ``generated_with.json`` into ``directory``."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()

    info = {
        'git_commit': git('rev-parse', '--short', 'HEAD'),
        'pero_ocr_uncommitted_changes': bool(git('status', '--porcelain', '--', 'pero_ocr')),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': numpy.__version__,
    }
    path = os.path.join(directory, 'generated_with.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=1)
        f.write('\n')
    return path
