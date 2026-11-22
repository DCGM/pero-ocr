#!/usr/bin/env python3
"""Recreate the frozen engine data in this directory with the current pero-ocr code.

    python test/engines/consistency/data/regenerate.py                      # everything
    python test/engines/consistency/data/regenerate.py lm                   # only the named engines
    python test/engines/consistency/data/regenerate.py --outputs-only ctc   # keep model and input, recompute output

Names: ctc, transformer, layout_cnn, lm, region_sorter_transformer (tiny models); smart_sorter, line_postprocessing,
layout_postprocessing and cropper (no model).

For every name, data/<name>/ holds the frozen model, the frozen input (input.npz, input.json), the output the engine
gave for it (output.npz) and generated_with.json. The checks in test/engines/consistency run the current code with the
frozen model on the frozen input and compare with output.npz (see frozen_engines.py).

It refuses to run outside the basis environment (requirements/lock.txt, issue #94) unless --any-environment is given:
data made with other library versions would not be what the live workers produce (shapely 2.1, for example, changes
the layout polygons), and models exported with a newer torch do not load in an older one.

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
import shapely  # noqa: E402
import torch  # noqa: E402

from test import tiny_models  # noqa: E402
from test.basis import require_basis  # noqa: E402
from test.engines.consistency import frozen_engines  # noqa: E402
from test.synthetic import layout_page, line_image, mark_reading_order, text_line  # noqa: E402


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
        'shapely': shapely.__version__,
    }
    path = os.path.join(directory, 'generated_with.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=1)
        f.write('\n')
    return path


# --- frozen inputs ------------------------------------------------------------------------------------------------

def line_inputs():
    """Line crops for the tiny OCR models (one block per character, its colour gives the character and the doubt):
    an empty line, white blocks, the whole alphabet, a line long enough for another batch, and a 'c' read with some
    doubt and with so much that 'b' wins."""
    lines = [line_image(0), line_image(3), text_line('bad cab'), text_line('abcd abcd abcd'),
             text_line('ac', [0.0, 0.45]), text_line('ac', [0.0, 0.6])]
    return {f'line/{i}': line for i, line in enumerate(lines)}


def ctc_input():
    return line_inputs(), {}


def transformer_input():
    return line_inputs(), {'beam_sizes': [1, 3]}


def tilted_text(page, left, baseline_y, length, height, angle_deg):
    """A white text block whose bottom edge runs from (left, baseline_y) at angle_deg, drawn into page."""
    a = np.deg2rad(angle_deg)
    along, up = np.array([np.cos(a), np.sin(a)]), np.array([np.sin(a), -np.cos(a)])
    p0 = np.array([left, baseline_y], dtype=np.float64)
    corners = np.stack([p0, p0 + length * along, p0 + length * along + height * up, p0 + height * up])
    cv2.fillPoly(page, [np.round(corners).astype(np.int32)], (255, 255, 255))


def layout_cnn_input():
    """Pages for the ParseNet stand-in (white text blocks, yellow line-end marks, red separators; see
    test/tiny_models.py), each run under two LAYOUT_CNN configs: the one the production engines use, and an
    experimental one (DETECTION_THRESHOLD 0.4, LINE_END_WEIGHT 0.6).
    The pages reach every step the production configs run: line detection with line ends, paragraph clustering with
    separators, region outlines, the adaptive second pass for large text, lines cut to their regions, and non-text
    regions kept."""
    column = [(100, 80 + 60 * i, 700, 112 + 60 * i) for i in range(6)]
    tilted = np.zeros((600, 800, 3), dtype=np.uint8)
    for i in range(4):
        tilted_text(tilted, 120, 160 + 70 * i, 520, 30, -3.0)
    pages = {
        'two_columns': layout_page(text=[(80, 100 + 50 * i, 380, 130 + 50 * i) for i in range(5)] +
                                   [(420, 100 + 50 * i, 720, 130 + 50 * i) for i in range(4)]),
        'title_and_paragraph': layout_page(text=[(150, 60, 650, 120)] +
                                           [(100, 150 + 50 * i, 700, 180 + 50 * i) for i in range(5)]),
        'separated_paragraphs': layout_page(text=column, separators=[(100, 252, 700, 260)]),
        'line_end': layout_page(text=[(100, 100, 700, 136), (100, 170, 700, 206)],
                                line_ends=[(390, 170, 410, 206)]),
        'large_text': layout_page(text=[(100, 60 + 160 * i, 700, 160 + 160 * i) for i in range(3)]),
        'varying_heights': layout_page(text=[(100, 100, 700, 124), (100, 150, 700, 186), (100, 212, 700, 242),
                                             (100, 268, 520, 296)]),
        'tilted_3deg': tilted,
        'with_an_image_region': layout_page(text=[(100, 100 + 50 * i, 420, 130 + 50 * i) for i in range(4)]),
        'empty': layout_page(),
    }
    production = dict(DETECT_LINES='yes', DETECT_REGIONS='yes', MERGE_LINES='no', ADJUST_HEIGHTS='no',
                      MAX_MEGAPIXELS=5, DOWNSAMPLE=3, DETECTION_THRESHOLD=0.3, LINE_END_WEIGHT=1.0,
                      VERTICAL_LINE_CONNECTION_RANGE=3, SMOOTH_LINE_PREDICTIONS='no')
    params = {
        'configs': {'layout_2022_04_13': production,
                    'experimental_2024_12_08': {**production, 'ADAPTIVE_DOWNSAMPLE': 1, 'DETECTION_THRESHOLD': 0.4,
                                                'LINE_END_WEIGHT': 0.6}},
        'pages': list(pages),
        'non_text': {'with_an_image_region': [
            {'id': 'image_1', 'category': 'image', 'polygon': [[480, 100], [720, 100], [720, 380], [480, 380]]}]},
    }
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


def region_sorter_transformer_input():
    """Region boxes [left, top, right, bottom] on a 1000 x 800 page, deliberately not in reading order, and one page
    whose image carries reading-order marks that contradict the geometry (the tiny model reads them as a learned model
    reads the page). The pages stay
    within MAX_BBOX_COUNT: regions beyond it are appended in a random order (a known bug, see
    test/engines/test_region_sorter_transformer.py), which a frozen output cannot capture."""
    rng = np.random.default_rng(20261120)
    two_columns = np.array([[420, 500, 760, 700], [40, 300, 380, 480], [420, 50, 760, 450], [40, 50, 380, 250]])
    column = np.array([[40, 20 + 80 * i, 380, 90 + 80 * i] for i in range(12)])
    pages = {
        'two_columns': two_columns,
        'twelve_regions_in_a_column': column[rng.permutation(len(column))],  # more than 8: the path search merges
        'one_region': np.array([[40, 50, 760, 950]]),
        'marked_against_the_geometry': two_columns,
    }
    marked = {'marked_against_the_geometry': [two_columns[0], two_columns[2]]}  # right column first, bottom to top
    inputs = {}
    for name, boxes in pages.items():
        inputs[f'boxes/{name}'] = boxes.astype(np.int64)
        inputs[f'image/{name}'] = mark_reading_order(np.zeros((1000, 800, 3), dtype=np.uint8), marked.get(name, []))
    params = {'config': {'model_path': 'model-cuda.pt', 'max_bbox_count': 32, 'image_width': 64, 'image_height': 64,
                         'use_cpu': 'yes'},
              'pages': list(pages)}
    return inputs, params


def smart_sorter_region(box, n_lines=None, angle=0.0):
    """Region polygon and lines (baseline, heights, line polygon) for a box [left, top, right, bottom]: one line per
    40 px unless n_lines is given, everything rotated by angle degrees around the page origin."""
    l, t, r, b = box
    polygon = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float64)
    ys = list(range(t + 30, b, 40)) if n_lines is None else list(np.linspace(t + 30, b - 10, n_lines))[:n_lines]
    lines = []
    for y in ys:
        baseline = np.array([[l + 5.0, y], [(l + r) / 2, y], [r - 5.0, y]])
        line_polygon = np.array([[l + 5.0, y - 20], [r - 5.0, y - 20], [r - 5.0, y + 5], [l + 5.0, y + 5]])
        lines.append((baseline, np.array([20.0, 5.0]), line_polygon))
    if angle:
        a = np.deg2rad(angle)
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
        polygon = polygon @ rotation
        lines = [(baseline @ rotation, heights, line_polygon @ rotation) for baseline, heights, line_polygon in lines]
    return polygon, lines


def smart_sorter_input():
    """Region layouts on a 1000 x 800 page, given in a scrambled order: the reading orders pero-ocr meets in
    production (title over columns, columns whose gaps do not line up, a tilted page, a newspaper-like page, a layout
    that no straight cut divides, which takes the fallback ordering) and the small overlaps that
    FakeIntersectionParameter (default 0.1) ignores."""
    title, left_top, left_bottom = (40, 40, 760, 120), (40, 160, 380, 500), (40, 540, 380, 940)
    right_top, right_bottom = (420, 160, 760, 700), (420, 740, 760, 940)
    columns = [right_bottom, left_bottom, title, right_top, left_top]
    pages = {
        'title_and_two_columns': [smart_sorter_region(box) for box in columns],
        'overlap_below_the_threshold': [smart_sorter_region(box) for box in
                                        [(420, 600, 760, 940), (40, 540, 380, 940), (420, 160, 760, 560),
                                         (40, 160, 380, 500)]],
        'tilted_4deg': [smart_sorter_region(box, angle=4.0) for box in columns],
        'newspaper': [smart_sorter_region(box, **kwargs) for box, kwargs in [
            ((560, 300, 760, 640), {}), ((40, 160, 760, 260), dict(n_lines=2)), ((300, 300, 520, 520), dict(n_lines=0)),
            ((40, 40, 760, 130), dict(n_lines=1)), ((40, 300, 260, 940), {}), ((300, 560, 520, 940), {}),
            ((560, 680, 760, 940), {}), ((40, 960, 760, 990), dict(n_lines=1))]],
        'pinwheel': [smart_sorter_region(box) for box in
                     [(200, 600, 760, 760), (600, 40, 760, 600), (40, 40, 600, 200), (40, 200, 200, 760),
                      (240, 240, 560, 560)]],
        'one_region': [smart_sorter_region((40, 40, 760, 940))],
    }
    inputs = {}
    for name, regions in pages.items():
        for i, (polygon, lines) in enumerate(regions):
            inputs[f'polygon/{name}/{i}'] = polygon
            for j, (baseline, heights, line_polygon) in enumerate(lines):
                inputs[f'baseline/{name}/{i}/{j}'] = baseline
                inputs[f'heights/{name}/{i}/{j}'] = heights
                inputs[f'line_polygon/{name}/{i}/{j}'] = line_polygon
    return inputs, {'config': {}, 'page_size': [1000, 800], 'pages': list(pages)}


def postprocessing_region(box, lines, angle=0.0, curve=0.0):
    """Region polygon and lines for a box [left, top, right, bottom] and lines (x0, x1, y), the baselines with 7 points
    bent by ``curve`` px in the middle, everything rotated by angle degrees around the box centre."""
    l, t, r, b = box
    polygon = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float64)
    out = []
    for x0, x1, y in lines:
        xs = np.linspace(x0, x1, 7)
        baseline = np.stack([xs, y + curve * (1 - ((xs - (x0 + x1) / 2) / ((x1 - x0) / 2)) ** 2)], axis=1)
        line_polygon = np.concatenate([baseline + [0, -20], (baseline + [0, 6])[::-1]])
        out.append((baseline, np.array([20.0, 6.0]), line_polygon))
    if angle:
        a = np.deg2rad(angle)
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
        centre = np.array([(l + r) / 2, (t + b) / 2])
        polygon = (polygon - centre) @ rotation + centre
        out = [((bl - centre) @ rotation + centre, h, (lp - centre) @ rotation + centre) for bl, h, lp in out]
    return polygon, out


def _region_inputs(pages):
    inputs = {}
    for name, regions in pages.items():
        for i, (polygon, lines) in enumerate(regions):
            inputs[f'polygon/{name}/{i}'] = polygon
            for j, (baseline, heights, line_polygon) in enumerate(lines):
                inputs[f'baseline/{name}/{i}/{j}'] = baseline
                inputs[f'heights/{name}/{i}/{j}'] = heights
                inputs[f'line_polygon/{name}/{i}/{j}'] = line_polygon
    return inputs


def line_postprocessing_input():
    """Lines shorter than their regions, under the three things LINE_POSTPROCESSING does: stretching to the region
    border, stretching by pixels, resampling to ten points. Tilted regions only with a single line: with more,
    stretching to the border is a known bug (test/engines/test_line_postprocessing.py)."""
    column = [(120, 560, 140 + 50 * i) for i in range(4)]
    pages = {
        'straight_column': [postprocessing_region((80, 100, 640, 330), column)],
        'single_tilted_line': [postprocessing_region((80, 100, 640, 200), [(150, 520, 160)], angle=6.0)],
        'curved_lines': [postprocessing_region((80, 100, 640, 330), column[:3], curve=8.0)],
        'two_regions': [postprocessing_region((40, 60, 360, 300), [(80, 300, 120 + 50 * i) for i in range(3)]),
                        postprocessing_region((420, 60, 760, 300), [(470, 700, 120 + 50 * i) for i in range(3)])],
    }
    off = dict(STRETCH_LINES=0, RESAMPLE_LINES='no', HEIGHTS_FROM_REGIONS='no')
    configs = {'stretch_to_border': {**off, 'STRETCH_LINES': 'max'}, 'stretch_by_20px': {**off, 'STRETCH_LINES': 20},
               'resample': {**off, 'RESAMPLE_LINES': 'yes'}}
    return _region_inputs(pages), {'configs': configs, 'page_size': [600, 800], 'pages': list(pages)}


def layout_postprocessing_input():
    """Regions whose outlines do not fit their lines, retraced around them: a column, the same tilted by 8 degrees,
    lines far apart, and two regions. Regions without lines are a known bug (test_layout_postprocessing.py)."""
    column = [(120, 560, 140 + 50 * i) for i in range(4)]
    pages = {
        'column': [postprocessing_region((40, 40, 760, 560), column)],
        'tilted_column': [postprocessing_region((40, 40, 760, 560), column, angle=8.0)],
        'lines_far_apart': [postprocessing_region((40, 40, 760, 560), [(100, 300, 120), (450, 700, 480)])],
        'two_regions': [postprocessing_region((20, 20, 390, 580), [(60, 340, 100 + 60 * i) for i in range(4)]),
                        postprocessing_region((410, 20, 780, 580), [(450, 740, 300 + 60 * i) for i in range(3)])],
    }
    return _region_inputs(pages), {'configs': {'retrace': {'RETRACE_REGIONS': 'yes'}}, 'page_size': [600, 800],
                                   'pages': list(pages)}


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


INPUTS = {'ctc': ctc_input, 'transformer': transformer_input, 'layout_cnn': layout_cnn_input, 'lm': lm_input,
          'region_sorter_transformer': region_sorter_transformer_input, 'smart_sorter': smart_sorter_input,
          'line_postprocessing': line_postprocessing_input, 'layout_postprocessing': layout_postprocessing_input,
          'cropper': cropper_input}
assert set(INPUTS) == set(frozen_engines.RUNNERS)
MODELS = {  # the frozen models, written into data/<name>/ (the smart sorter and the cropper have none)
    'ctc': tiny_models.write_ctc,
    'transformer': tiny_models.write_transformer,
    'layout_cnn': tiny_models.write_parsenet,
    'lm': tiny_models.write_lm,
    'region_sorter_transformer': tiny_models.write_region_sorter,
}


# --- writing ------------------------------------------------------------------------------------------------------

def regenerate(name, outputs_only):
    directory = os.path.join(DATA_DIR, name)
    if outputs_only:
        directory, inputs, params, _ = frozen_engines.load(name)
    else:
        if name in MODELS:
            MODELS[name](directory)
        inputs, params = INPUTS[name]()
        frozen_engines.save_input(directory, inputs, params)
        _, inputs, params, _ = frozen_engines.load(name, with_output=False)  # run on exactly what the check reads
    frozen_engines.save_output(directory, frozen_engines.RUNNERS[name](directory, inputs, params))
    write_generated_with(directory)
    what = 'output' if outputs_only else 'model, input and output' if name in MODELS else 'input and output'
    print(f'wrote {name}: {what} in {os.path.relpath(directory, REPO_DIR)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('names', nargs='*', help=f'default: all of {sorted(INPUTS)}')
    parser.add_argument('--outputs-only', action='store_true',
                        help='keep the frozen model and input, recompute only output.npz')
    parser.add_argument('--any-environment', action='store_true',
                        help='regenerate even when the installed versions differ from requirements/lock.in')
    args = parser.parse_args()
    names = args.names or list(INPUTS)
    unknown = sorted(set(names) - set(INPUTS))
    if unknown:
        sys.exit(f'unknown name(s) {unknown}, choose from {sorted(INPUTS)}')
    require_basis(args.any_environment)
    for name in names:
        regenerate(name, args.outputs_only)


if __name__ == '__main__':
    main()
