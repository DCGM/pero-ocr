#!/usr/bin/env python3
"""Recreate the frozen tiny models in this directory with the current pero-ocr code.

    python test/engines/consistency/data/regenerate.py

Only after a deliberate, announced break of the model format, or to add a new kind of stand-in model. These models
are meant to stay old: regenerating them resets the check that previously exported models still work.

The models themselves are defined in test/tiny_models.py, because the engine tests build fresh copies of the same
models in every test session; this script only writes them here and records generated_with.json.
"""
import json
import os
import platform
import subprocess
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(DATA_DIR, '..', '..', '..', '..'))
sys.path.insert(0, REPO_DIR)  # this checkout's pero_ocr and test packages, not installed ones

import numpy  # noqa: E402
import torch  # noqa: E402

from test import tiny_models  # noqa: E402


def write_generated_with(directory):
    """Record which code and library versions produced the data in ``directory``, as generated_with.json."""
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


def main():
    for name, path in tiny_models.write_all(DATA_DIR).items():
        print('wrote', name, os.path.relpath(path, REPO_DIR))
    print('wrote', os.path.relpath(write_generated_with(DATA_DIR), REPO_DIR))


if __name__ == '__main__':
    main()
