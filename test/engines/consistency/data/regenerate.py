#!/usr/bin/env python3
"""Recreate frozen tiny models in this directory with the current pero-ocr code.

    python test/engines/consistency/data/regenerate.py            # all models
    python test/engines/consistency/data/regenerate.py lm         # only the named ones, e.g. a newly added stand-in

Regenerate all models only after a deliberate, announced break of the model format. These models are meant to stay
old: regenerating a model resets the check that previously exported models of that kind still work. To add a new kind
of stand-in, name only that one, so the existing frozen models stay untouched. Each model directory gets its own
generated_with.json, because models can be frozen at different times.

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
    names = sys.argv[1:] or list(tiny_models.WRITERS)
    unknown = sorted(set(names) - set(tiny_models.WRITERS))
    if unknown:
        sys.exit(f'unknown model(s) {unknown}, choose from {sorted(tiny_models.WRITERS)}')
    for name, path in tiny_models.write_all(DATA_DIR, names).items():
        print('wrote', name, os.path.relpath(path, REPO_DIR))
        model_dir = os.path.join(DATA_DIR, name)
        print('wrote', os.path.relpath(write_generated_with(model_dir), REPO_DIR))


if __name__ == '__main__':
    main()
