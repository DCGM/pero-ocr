#!/usr/bin/env python3
"""Recreate the frozen engine data in this directory with the current pero-ocr code.

    python test/engines/consistency/data/regenerate.py                      # everything
    python test/engines/consistency/data/regenerate.py lm                   # only the named engines
    python test/engines/consistency/data/regenerate.py --outputs-only ctc   # keep model and input, recompute output

Names: ctc, transformer, parsenet, lm, region_sorter (tiny models) and cropper (the line cropper, which has no model).

For every name, data/<name>/ holds the frozen model, the frozen input (input.npz, input.json), the output the engine
gave for it (output.npz) and generated_with.json. The checks in test/engines/consistency run the current code with the
frozen model on the frozen input and compare with output.npz (see frozen_engines.py).

Regenerate only after a deliberate, announced change, and name only the engines it concerns, so the rest stays old:

- the engine output changes on purpose (new logits format, a fixed bug, a new cropper): --outputs-only recomputes
  output.npz from the frozen model and input, so the check that old models still load keeps working;
- the model format breaks on purpose (models have to be re-exported): regenerate the engine completely;
- a new engine or stand-in: name only that one.

The models are defined in test/tiny_models.py, because the engine tests build fresh copies of them in every session.
The inputs are defined here, because only this script uses them; the checks read them back from data/<name>.
"""
import argparse
import json
import os
import platform
import subprocess
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(DATA_DIR, '..', '..', '..', '..'))
sys.path.insert(0, REPO_DIR)  # this checkout's pero_ocr and test packages, not installed ones

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402

from test import tiny_models  # noqa: E402
from test.engines.consistency import frozen_engines  # noqa: E402
from test.synthetic import line_image, page_image  # noqa: E402


def write_generated_with(directory):
    """Record which code and library versions produced the data in ``directory``, as generated_with.json."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()

    info = {
        'git_commit': git('rev-parse', '--short', 'HEAD'),
        'pero_ocr_uncommitted_changes': bool(git('status', '--porcelain', '--', 'pero_ocr')),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'opencv': cv2.__version__,
        'scipy': scipy.__version__,
    }
    path = os.path.join(directory, 'generated_with.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=1)
        f.write('\n')
    return path


# --- frozen inputs ------------------------------------------------------------------------------------------------

def line_inputs():
    """Line crops with 0, 1, 4 and 13 bright blocks: an empty line, short ones and one long enough for another batch."""
    return {f'line/{i}': line_image(n) for i, n in enumerate((0, 1, 4, 13))}


def ctc_input():
    return line_inputs(), {}


def transformer_input():
    return line_inputs(), {'beam_sizes': [1, 3]}


def parsenet_input():
    pages = {'three_lines': page_image(3), 'five_narrow_lines': page_image(5, line_height=24, line_gap=40)}
    params = {'engine': {'downsample': 4, 'detection_threshold': 0.2, 'max_mp': 5}, 'pages': list(pages)}
    return {f'page/{name}': page for name, page in pages.items()}, params


def ctc_frames(*specs):
    """CTC log-probabilities over tiny_models.CHARACTERS + blank, e.g. ctc_frames({'a': .98}, {'blank': .98})."""
    characters = tiny_models.CHARACTERS
    n = len(characters) + 1
    rows = []
    for spec in specs:
        frame = np.full(n, 1e-4)
        for symbol, p in spec.items():
            frame[n - 1 if symbol == 'blank' else characters.index(symbol)] = p
        rows.append(np.log(frame / frame.sum()))
    return np.stack(rows)


def lm_input():
    a, blank = {'a': .98}, {'blank': .98}
    frames = {
        'c_or_b_after_a': ctc_frames(a, blank, {'c': .6, 'b': .4}, blank),  # the LM prefers b after a
        'clear_c_after_a': ctc_frames(a, blank, {'c': .98}, blank),  # visually clear: the LM must not override it
        'c_or_b_at_start': ctc_frames({'c': .6, 'b': .4}, blank),  # nothing before: the visual model decides
    }
    params = {'characters': tiny_models.CHARACTERS, 'prefixes': ['', 'a', 'ab', 'ba'], 'frames': list(frames),
              'lm_scales': [1.0, 0.4], 'beam_size': 8}
    return {f'frames/{name}': f for name, f in frames.items()}, params


def region_sorter_input():
    """Region boxes [left, top, right, bottom] on a 1000 x 800 page, deliberately not in reading order. The pages stay
    within MAX_BBOX_COUNT: regions beyond it are appended in a random order (a known bug, see
    test/engines/test_region_sorter.py), which a frozen output cannot capture."""
    rng = np.random.default_rng(20261120)
    two_columns = np.array([[420, 500, 760, 700], [40, 300, 380, 480], [420, 50, 760, 450], [40, 50, 380, 250]])
    column = np.array([[40, 20 + 80 * i, 380, 90 + 80 * i] for i in range(12)])
    pages = {
        'two_columns': two_columns,
        'twelve_regions_in_a_column': column[rng.permutation(len(column))],  # more than 8: the path search merges
        'one_region': np.array([[40, 50, 760, 950]]),
    }
    inputs = {}
    for name, boxes in pages.items():
        inputs[f'boxes/{name}'] = boxes.astype(np.int64)
        inputs[f'image/{name}'] = np.zeros((1000, 800, 3), dtype=np.uint8)
    params = {'config': {'model_path': 'model-cuda.pt', 'max_bbox_count': 32, 'image_width': 64, 'image_height': 64,
                         'use_cpu': 'yes'},
              'pages': list(pages)}
    return inputs, params


PAGE_HEIGHT, PAGE_WIDTH = 480, 640


def synthetic_text_page():
    """A page with sharp, antialiased text-like strokes over a slowly changing background. The strokes make any change
    of the interpolation visible (opencv 5 changes every frozen crop), and the page compresses well."""
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


def cropper_input():
    """Reasonable lines that the current cropper handles, with the settings used in pero-ocr:
    production LineCropper (LINE_HEIGHT 48, INTERP 2, LINE_SCALE 1.15), the ALTO word alignment (32, 2, 1.0) and the
    baseline refiner (32, 0, 1.0). Shapes the cropper cannot handle yet (S-curves with INTERP 2, circles) are known
    limits in test/engines/test_line_cropper.py and do not belong here."""
    production = dict(line_height=48, poly=2, scale=1.15)
    t = np.linspace(0, 1, 9)
    arc = np.stack([120 + 400 * t, 250 + 120 * (t - .5) ** 2], axis=1).round().tolist()
    wave = np.stack([120 + 400 * t, 380 + 6 * np.sin(2 * np.pi * t)], axis=1).round().tolist()
    tilted = np.stack([60 + 450 * t, 120 + 450 * t * np.tan(np.deg2rad(3))], axis=1).round().tolist()
    cases = [
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
    return {'page': synthetic_text_page()}, {'cases': cases}


INPUTS = {'ctc': ctc_input, 'transformer': transformer_input, 'parsenet': parsenet_input, 'lm': lm_input,
          'region_sorter': region_sorter_input, 'cropper': cropper_input}
assert set(INPUTS) == set(frozen_engines.RUNNERS)


# --- writing ------------------------------------------------------------------------------------------------------

def regenerate(name, outputs_only):
    directory = os.path.join(DATA_DIR, name)
    if outputs_only:
        directory, inputs, params, _ = frozen_engines.load(name)
    else:
        if name in tiny_models.WRITERS:
            tiny_models.WRITERS[name](DATA_DIR)
        inputs, params = INPUTS[name]()
        frozen_engines.save_input(directory, inputs, params)
        _, inputs, params, _ = frozen_engines.load(name, with_output=False)  # run on exactly what the check reads
    frozen_engines.save_output(directory, frozen_engines.RUNNERS[name](directory, inputs, params))
    write_generated_with(directory)
    what = 'output' if outputs_only else 'model, input and output' if name in tiny_models.WRITERS else 'input and output'
    print(f'wrote {name}: {what} in {os.path.relpath(directory, REPO_DIR)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('names', nargs='*', help=f'default: all of {sorted(INPUTS)}')
    parser.add_argument('--outputs-only', action='store_true',
                        help='keep the frozen model and input, recompute only output.npz')
    args = parser.parse_args()
    names = args.names or list(INPUTS)
    unknown = sorted(set(names) - set(INPUTS))
    if unknown:
        sys.exit(f'unknown name(s) {unknown}, choose from {sorted(INPUTS)}')
    for name in names:
        regenerate(name, args.outputs_only)


if __name__ == '__main__':
    main()
