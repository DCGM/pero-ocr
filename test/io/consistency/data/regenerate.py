#!/usr/bin/env python3
"""Recreate the golden IO files in this directory with the current pero-ocr code.

    python test/io/consistency/data/regenerate.py

Run it when the output format changes on purpose, then review `git diff` of this directory and commit it together
with the code change. Which files exist and which exporter call produces each of them is GOLDEN_IO_FILES in
../golden_io.py; this script writes all of them, removes files no longer listed, and records generated_with.json.
"""
import json
import logging
import os
import platform
import shutil
import subprocess
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(DATA_DIR, '..', '..', '..', '..'))
sys.path.insert(0, REPO_DIR)  # this checkout's pero_ocr and test packages, not installed ones

import numpy  # noqa: E402
import torch  # noqa: E402

from test.io.consistency.golden_io import GOLDEN_IO_FILES, produce  # noqa: E402

FORMAT_DIRS = ['page_xml', 'alto', 'logits']


def write_golden_io_files(directory):
    """Write every file of GOLDEN_IO_FILES into ``directory``; files no longer listed are removed."""
    for format_dir in FORMAT_DIRS:
        shutil.rmtree(os.path.join(directory, format_dir), ignore_errors=True)
    written = []
    for name in GOLDEN_IO_FILES:
        path = os.path.join(directory, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        content = produce(name)
        if isinstance(content, str):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(path, 'wb') as f:
                f.write(content)
        written.append(path)
    return written


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
    # the edge-case page has a line without logits on purpose; the warnings pero-ocr logs for it are expected
    logging.getLogger('pero_ocr').setLevel(logging.ERROR)
    for path in write_golden_io_files(DATA_DIR):
        print('wrote', os.path.relpath(path, REPO_DIR))
    print('wrote', os.path.relpath(write_generated_with(DATA_DIR), REPO_DIR))


if __name__ == '__main__':
    main()
