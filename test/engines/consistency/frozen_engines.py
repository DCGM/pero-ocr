"""Frozen engine data: what the engine consistency checks and data/regenerate.py share.

Every engine has a directory data/<name>/ that was written once by data/regenerate.py and then committed:

- the frozen model, exported by the pero-ocr, torch and brnolm of that time (the smart sorter and the cropper have
  none),
- input.npz, plus input.json for parameters: the frozen input,
- output.npz: what the engine returned for that input at that time,
- generated_with.json: the commit and library versions.

A check runs the current engine code with the frozen model on the frozen input (RUNNERS[name]) and compares the result
with output.npz (check()). So it fails both when an old model no longer loads or behaves differently, and when the
engine code or a library changes the output. regenerate.py runs exactly the same RUNNERS to write output.npz, so the
check and the script cannot drift apart.

Text, integers and crops (uint8) must match exactly. Float outputs (logits, baselines, log-probabilities) must match
up to FLOAT_TOLERANCE, which ignores the last bits of float arithmetic that can differ between torch builds but not any
change of behaviour.
"""
import configparser
import json
import os

import numpy as np
import torch

from pero_ocr.core.crop_engine import EngineLineCropper
from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.decoding.decoders import BLANK_SYMBOL, CTCPrefixLogRawNumpyDecoder
from pero_ocr.decoding.decoding_itf import construct_lm
from pero_ocr.decoding.lm_wrapper import LMWrapper
from pero_ocr.layout_engines.cnn_layout_engine import LayoutEngine
from pero_ocr.layout_engines.smart_sorter import SmartRegionSorter
from pero_ocr.layout_engines.transformer_sorter import TransformerRegionSorter
from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR
from pero_ocr.ocr_engine.transformer_ocr_engine import TransformerEngineLineOCR

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
INPUT_FILE, PARAMS_FILE, OUTPUT_FILE = 'input.npz', 'input.json', 'output.npz'
FLOAT_TOLERANCE = dict(rtol=1e-5, atol=1e-5)
CPU = torch.device('cpu')


# --- the engines, run on frozen input -----------------------------------------------------------------------------

def _ocr_outputs(prefix, result):
    transcriptions, logits, logit_coords = result
    out = {f'{prefix}transcriptions': np.array(transcriptions, dtype=str)}
    for i, (line_logits, coords) in enumerate(zip(logits, logit_coords)):
        out[f'{prefix}logits/{i}'] = line_logits.toarray()
        out[f'{prefix}logit_coords/{i}'] = np.array(coords, dtype=np.int64)
    return out


def _lines(inputs):
    return [inputs[f'line/{i}'] for i in range(len([k for k in inputs if k.startswith('line/')]))]


def run_ctc(directory, inputs, params):
    engine = PytorchEngineLineOCR(os.path.join(directory, 'ocr_engine.json'), CPU)
    return _ocr_outputs('', engine.process_lines(_lines(inputs)))


def run_transformer(directory, inputs, params):
    out = {}
    for beam_size in params['beam_sizes']:
        engine = TransformerEngineLineOCR(os.path.join(directory, 'ocr_engine.json'), CPU, beam_size=beam_size)
        out.update(_ocr_outputs(f'beam_{beam_size}/', engine.process_lines(_lines(inputs))))
    return out


def run_parsenet(directory, inputs, params):
    engine = LayoutEngine(os.path.join(directory, 'parsenet.pt'), CPU, **params['engine'])
    out = {}
    for name in params['pages']:
        regions, baselines, heights, *_ = engine.detect(inputs[f'page/{name}'])
        out[f'{name}/region_count'] = np.array(len(regions))
        for i, region in enumerate(regions):
            out[f'{name}/region/{i}'] = np.asarray(region, dtype=np.float64)
        for i, (baseline, height) in enumerate(zip(baselines, heights)):
            out[f'{name}/baseline/{i}'] = np.asarray(baseline, dtype=np.float64)
            out[f'{name}/heights/{i}'] = np.asarray(height, dtype=np.float64)
    return out


def run_lm(directory, inputs, params):
    characters = params['characters']
    lm = LMWrapper(construct_lm(os.path.join(directory, 'lm.pt')), characters, CPU)
    out = {}
    for prefix in params['prefixes']:
        h = lm.initial_h(1)
        for char in prefix:
            h = lm.advance_h0(np.array([characters.index(char)]), h)
        out[f'log_probs/{prefix or "<start>"}'] = lm.log_probs(h)
    for name in params['frames']:
        for lm_scale in params['lm_scales']:
            decoder = CTCPrefixLogRawNumpyDecoder(characters + [BLANK_SYMBOL], params['beam_size'], lm, lm_scale)
            out[f'decoded/{name}/lm_scale_{lm_scale}'] = np.array(decoder(inputs[f'frames/{name}']).best_hyp())
    return out


def run_region_sorter(directory, inputs, params):
    """Reading order of the frozen pages: output order/<page> lists the input regions in the order the sorter put them."""
    config = configparser.ConfigParser()
    config.read_dict({'S': {key.upper(): str(value) for key, value in params['config'].items()}})
    sorter = TransformerRegionSorter(config['S'], CPU, config_path=directory)
    out = {}
    for name in params['pages']:
        boxes = inputs[f'boxes/{name}']
        image = inputs[f'image/{name}']
        layout = PageLayout(id=name, page_size=image.shape[:2])
        layout.regions = [RegionLayout(str(i), np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float64))
                          for i, (l, t, r, b) in enumerate(boxes)]
        layout = sorter.process_page(image, layout)
        out[f'order/{name}'] = np.array([int(region.id) for region in layout.regions], dtype=np.int64)
    return out


def _layout_from_arrays(inputs, name, page_size):
    """A PageLayout from the arrays polygon/<page>/<region>, and per line baseline/, heights/ and line_polygon/
    <page>/<region>/<line>. Region i gets the id str(i)."""
    layout = PageLayout(id=name, page_size=tuple(page_size))
    n_regions = len([k for k in inputs if k.startswith(f'polygon/{name}/')])
    for i in range(n_regions):
        region = RegionLayout(str(i), inputs[f'polygon/{name}/{i}'])
        n_lines = len([k for k in inputs if k.startswith(f'baseline/{name}/{i}/')])
        for j in range(n_lines):
            region.lines.append(TextLine(id=f'{i}-l{j}', baseline=inputs[f'baseline/{name}/{i}/{j}'],
                                         heights=inputs[f'heights/{name}/{i}/{j}'].tolist(),
                                         polygon=inputs[f'line_polygon/{name}/{i}/{j}']))
        layout.regions.append(region)
    return layout


def run_smart_sorter(directory, inputs, params):
    """Reading order of the frozen pages (order/<page>: the input regions in the sorted order), and the geometry of
    the returned regions in that order: the sorter rotates a tilted page to sort it and rotates it back."""
    config = configparser.ConfigParser()
    config.read_dict({'S': {key.upper(): str(value) for key, value in params['config'].items()}})
    sorter = SmartRegionSorter(config['S'])
    out = {}
    for name in params['pages']:
        page_size = params['page_size']
        layout = sorter.process_page(np.zeros(tuple(page_size) + (3,), dtype=np.uint8),
                                     _layout_from_arrays(inputs, name, page_size))
        out[f'order/{name}'] = np.array([int(region.id) for region in layout.regions], dtype=np.int64)
        for k, region in enumerate(layout.regions):
            out[f'polygon/{name}/{k}'] = np.asarray(region.polygon, dtype=np.float64)
            for j, line in enumerate(region.lines):
                out[f'baseline/{name}/{k}/{j}'] = np.asarray(line.baseline, dtype=np.float64)
                out[f'line_polygon/{name}/{k}/{j}'] = np.asarray(line.polygon, dtype=np.float64)
    return out


def run_cropper(directory, inputs, params):
    page = inputs['page']
    out = {}
    for case in params['cases']:
        cropper = EngineLineCropper(line_height=case['line_height'], poly=case['poly'], scale=case['scale'])
        baseline = np.array(case['baseline'], dtype=np.float64)
        heights = case['heights']
        name = case['name']
        if case.get('forward_mapping'):
            crop, out[f'{name}/coords'] = cropper.crop(page, baseline, heights, return_forward_mapping=True)
        elif case.get('blend_in'):
            crop, mapping, offset = cropper.crop(page, baseline, heights, return_mapping=True)
            out[f'{name}/blended'] = cropper.blend_in(np.zeros_like(page), crop, mapping, offset)
        else:
            crop = cropper.crop(page, baseline, heights)
        out[f'{name}/crop'] = crop
    return out


RUNNERS = {'ctc': run_ctc, 'transformer': run_transformer, 'parsenet': run_parsenet, 'lm': run_lm,
           'region_sorter': run_region_sorter, 'smart_sorter': run_smart_sorter, 'cropper': run_cropper}


# --- reading, writing, comparing ----------------------------------------------------------------------------------

def save_input(directory, inputs, params):
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(os.path.join(directory, INPUT_FILE), **inputs)
    with open(os.path.join(directory, PARAMS_FILE), 'w') as f:
        json.dump(params, f, indent=1)
        f.write('\n')


def save_output(directory, outputs):
    np.savez_compressed(os.path.join(directory, OUTPUT_FILE), **outputs)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load(name, with_output=True):
    """Returns (directory, input arrays, parameters, frozen output or None)."""
    directory = os.path.join(DATA_DIR, name)
    with open(os.path.join(directory, PARAMS_FILE)) as f:
        params = json.load(f)
    output = _load_npz(os.path.join(directory, OUTPUT_FILE)) if with_output else None
    return directory, _load_npz(os.path.join(directory, INPUT_FILE)), params, output


def run(name):
    """Run the current code on the frozen data of ``name``; returns (current output, frozen output)."""
    directory, inputs, params, frozen = load(name)
    return RUNNERS[name](directory, inputs, params), frozen


def check(name):
    """Assert that the current code reproduces the frozen output of ``name``."""
    current, frozen = run(name)
    regenerate = f'run test/engines/consistency/data/regenerate.py {name} only after a deliberate change'
    assert set(current) == set(frozen), (
        f'{name}: outputs {sorted(set(current) ^ set(frozen))} appeared or disappeared; {regenerate}')
    problems = []
    for key in sorted(frozen):
        value, expected = np.asarray(current[key]), frozen[key]
        if value.shape != expected.shape:
            problems.append(f'{key}: shape {value.shape}, frozen {expected.shape}')
        elif expected.dtype.kind == 'f':
            if not np.allclose(value, expected, **FLOAT_TOLERANCE):
                problems.append(f'{key}: values differ by up to {np.abs(value - expected).max():.3g}')
        elif expected.dtype.kind == 'U':
            if not np.array_equal(value, expected):
                problems.append(f'{key}: {value.tolist()!r}, frozen {expected.tolist()!r}')
        elif not np.array_equal(value, expected):
            differing = value != expected
            problems.append(f'{key}: {differing.mean():.2%} of the values differ, by up to '
                            f'{np.abs(value.astype(np.int64) - expected).max()}')
    assert not problems, f'{name}: the output changed ({regenerate}):\n  ' + '\n  '.join(problems)
