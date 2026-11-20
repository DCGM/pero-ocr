#!/usr/bin/env python3
"""Recreate the frozen engine data in this directory with the current pero-ocr code.

    python test/engines/consistency/data/regenerate.py            # everything
    python test/engines/consistency/data/regenerate.py lm         # only the named ones, e.g. a newly added stand-in

Names: the tiny models (ctc, transformer, parsenet, lm) and cropper (frozen line crops).

Regenerate only after a deliberate, announced break. The data is meant to stay old: regenerating a model resets the
check that previously exported models of that kind still work, and regenerating the crops resets the check that the
cropper still cuts the crops the models were trained on. To add a new kind of data, name only that one, so the existing
frozen data stays untouched. Each directory gets its own generated_with.json, because data can be frozen at different
times.

The models themselves are defined in test/tiny_models.py, because the engine tests build fresh copies of the same
models in every test session. The synthetic page and the line cases for the cropper are defined here, because only
this script uses them; the check reads them back from data/cropper.
"""
import json
import os
import platform
import subprocess
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(DATA_DIR, '..', '..', '..', '..'))
sys.path.insert(0, REPO_DIR)  # this checkout's pero_ocr and test packages, not installed ones

import cv2  # noqa: E402
import numpy  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402

from test import tiny_models  # noqa: E402
from test.engines.consistency import frozen_crops  # noqa: E402

CROPPER = 'cropper'
NAMES = list(tiny_models.WRITERS) + [CROPPER]


def write_generated_with(directory, extra=None):
    """Record which code and library versions produced the data in ``directory``, as generated_with.json."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()

    info = {
        'git_commit': git('rev-parse', '--short', 'HEAD'),
        'pero_ocr_uncommitted_changes': bool(git('status', '--porcelain', '--', 'pero_ocr')),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': numpy.__version__,
        **(extra or {}),
    }
    path = os.path.join(directory, 'generated_with.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=1)
        f.write('\n')
    return path


# --- frozen line crops --------------------------------------------------------------------------------------------

PAGE_HEIGHT, PAGE_WIDTH = 480, 640


def synthetic_page():
    """A page with sharp, antialiased text-like strokes over a slowly changing background. The strokes make any change
    of the interpolation visible (opencv 5 changes every frozen crop), and the page compresses well as PNG."""
    yy, xx = np.mgrid[0:PAGE_HEIGHT, 0:PAGE_WIDTH].astype(np.float64)
    page = np.stack([170 + 0.08 * xx, 185 - 0.05 * yy, 160 + 0.04 * (xx + yy)], axis=2).astype(np.uint8)
    rng = np.random.default_rng(20261120)
    words = ['pero', 'ocr', 'line', 'crop', 'text', 'Abg', 'qyp', '1987']
    for y in range(30, PAGE_HEIGHT, 34):
        x = int(rng.integers(5, 30))
        while x < PAGE_WIDTH - 60:
            word = words[int(rng.integers(len(words)))]
            colour = tuple(int(c) for c in rng.integers(0, 70, 3))
            cv2.putText(page, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
            x += 22 * len(word) + int(rng.integers(8, 30))
    return page


def straight(p0, angle_deg, length):
    a = np.deg2rad(angle_deg)
    return [list(map(float, p0)), [float(round(p0[0] + length * np.cos(a))), float(round(p0[1] + length * np.sin(a)))]]


def cropper_cases():
    """Reasonable lines that the current cropper handles, with the settings used in pero-ocr:
    production LineCropper (LINE_HEIGHT 48, INTERP 2, LINE_SCALE 1.15), the ALTO word alignment (32, 2, 1.0) and the
    baseline refiner (32, 0, 1.0). Shapes the cropper cannot handle yet (S-curves with INTERP 2, circles) are known
    limits in test/engines/test_line_cropper.py and do not belong here."""
    production = dict(line_height=48, poly=2, scale=1.15)
    t = np.linspace(0, 1, 9)
    arc = np.stack([120 + 400 * t, 250 + 120 * (t - .5) ** 2], axis=1).round().tolist()
    wave = np.stack([120 + 400 * t, 380 + 6 * np.sin(2 * np.pi * t)], axis=1).round().tolist()
    tilted = np.stack([60 + 450 * t, 120 + 450 * t * np.tan(np.deg2rad(3))], axis=1).round().tolist()
    return [
        dict(name='horizontal', baseline=straight((40, 64), 0, 520), heights=[22, 7], **production),
        dict(name='tilted_3deg_multipoint', baseline=tilted, heights=[20, 8], **production),
        dict(name='tilted_30deg', baseline=straight((100, 140), 30, 300), heights=[20, 8], **production),
        dict(name='vertical', baseline=straight((600, 60), 90, 360), heights=[18, 6], **production),
        dict(name='upside_down', baseline=straight((560, 300), 180, 420), heights=[20, 8], **production),
        dict(name='gentle_arc', baseline=arc, heights=[20, 8], **production),
        dict(name='short', baseline=straight((300, 200), 0, 40), heights=[20, 8], **production),
        dict(name='partly_outside_the_page', baseline=straight((-60, 440), 0, 300), heights=[20, 8], **production),
        dict(name='alto_alignment', baseline=straight((40, 166), 0, 500), heights=[20, 8],
             line_height=32, poly=2, scale=1.0, forward_mapping=True),
        dict(name='baseline_refiner_wave', baseline=wave, heights=[20, 8], line_height=32, poly=0, scale=1.0),
        dict(name='blend_in', baseline=straight((200, 300), 2, 160), heights=[20, 8],
             line_height=48, poly=2, scale=1.0, blend_in=True),
    ]


def write_cropper(directory):
    os.makedirs(directory, exist_ok=True)
    page = synthetic_page()
    cv2.imwrite(os.path.join(directory, frozen_crops.PAGE_FILE), page, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    page = cv2.imread(os.path.join(directory, frozen_crops.PAGE_FILE), cv2.IMREAD_COLOR)  # exactly what the check reads
    cases = cropper_cases()
    with open(os.path.join(directory, frozen_crops.CASES_FILE), 'w') as f:
        json.dump(cases, f, indent=1)
        f.write('\n')
    crops = {}
    for case in cases:
        crops.update(frozen_crops.crop_case(page, case))
    np.savez_compressed(os.path.join(directory, frozen_crops.CROPS_FILE), **crops)
    return directory


def main():
    names = sys.argv[1:] or NAMES
    unknown = sorted(set(names) - set(NAMES))
    if unknown:
        sys.exit(f'unknown name(s) {unknown}, choose from {sorted(NAMES)}')
    models = [name for name in names if name != CROPPER]
    # write_all() treats an empty list as "all models", so only call it when models were asked for
    for name, path in (tiny_models.write_all(DATA_DIR, models) if models else {}).items():
        print('wrote', name, os.path.relpath(path, REPO_DIR))
        model_dir = os.path.join(DATA_DIR, name)
        print('wrote', os.path.relpath(write_generated_with(model_dir), REPO_DIR))
    if CROPPER in names:
        directory = write_cropper(os.path.join(DATA_DIR, CROPPER))
        print('wrote', CROPPER, os.path.relpath(directory, REPO_DIR))
        extra = {'opencv': cv2.__version__, 'scipy': scipy.__version__}
        print('wrote', os.path.relpath(write_generated_with(directory, extra), REPO_DIR))


if __name__ == '__main__':
    main()
