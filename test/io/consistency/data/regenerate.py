#!/usr/bin/env python3
"""Recreate the golden IO files in this directory with the current pero-ocr code.

    python test/io/consistency/data/regenerate.py

Run it when the output format changes on purpose, then review `git diff` of this directory and commit it together
with the code change. The list of files and how each is produced is GOLDEN_IO_FILES in ../golden_io.py.
"""
import logging
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(DATA_DIR, '..', '..', '..', '..'))
sys.path.insert(0, REPO_DIR)  # this checkout's pero_ocr and test packages, not installed ones

from test.io.consistency.golden_io import write_all  # noqa: E402
from test.provenance import write_generated_with  # noqa: E402


def main():
    # the edge-case page has a line without logits on purpose; the warnings pero-ocr logs for it are expected
    logging.getLogger('pero_ocr').setLevel(logging.ERROR)
    for path in write_all(DATA_DIR):
        print('wrote', os.path.relpath(path, REPO_DIR))
    print('wrote', os.path.relpath(write_generated_with(DATA_DIR), REPO_DIR))


if __name__ == '__main__':
    main()
