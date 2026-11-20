#!/usr/bin/env python3
"""Recreate the frozen tiny models in this directory with the current pero-ocr code.

    python test/engines/consistency/data/regenerate.py

Only after a deliberate, announced break of the model format, or to add a new kind of stand-in model. These models
are meant to stay old: regenerating them resets the check that previously exported models still work. The models are
defined in test/tiny_models.py.
"""
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(DATA_DIR, '..', '..', '..', '..'))
sys.path.insert(0, REPO_DIR)  # this checkout's pero_ocr and test packages, not installed ones

from test import tiny_models  # noqa: E402
from test.provenance import write_generated_with  # noqa: E402


def main():
    for name, path in tiny_models.write_all(DATA_DIR).items():
        print('wrote', name, os.path.relpath(path, REPO_DIR))
    print('wrote', os.path.relpath(write_generated_with(DATA_DIR), REPO_DIR))


if __name__ == '__main__':
    main()
