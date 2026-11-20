"""Frozen line crops: what test_line_cropper.py and data/regenerate.py share.

data/cropper/ holds a synthetic page (page.png), the lines cut out of it with the cropper settings used in pero-ocr
(cases.json), and the crops an earlier version produced (crops.npz). Both the check and the regenerate script run every
case through crop_case(), so they cannot drift apart."""
import json
import os

import cv2
import numpy as np

from pero_ocr.core.crop_engine import EngineLineCropper

CROPPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cropper')
PAGE_FILE = 'page.png'
CASES_FILE = 'cases.json'
CROPS_FILE = 'crops.npz'


def crop_case(page: np.ndarray, case: dict) -> dict:
    """Run one case through the cropper. Returns the arrays that are frozen for it, keyed '<case name>/<what>'."""
    cropper = EngineLineCropper(line_height=case['line_height'], poly=case['poly'], scale=case['scale'])
    baseline = np.array(case['baseline'], dtype=np.float64)
    heights = case['heights']
    out = {}
    if case.get('forward_mapping'):
        crop, coords = cropper.crop(page, baseline, heights, return_forward_mapping=True)
        out['coords'] = coords
    elif case.get('blend_in'):
        crop, mapping, offset = cropper.crop(page, baseline, heights, return_mapping=True)
        out['blended'] = cropper.blend_in(np.zeros_like(page), crop, mapping, offset)
    else:
        crop = cropper.crop(page, baseline, heights)
    out['crop'] = crop
    return {f"{case['name']}/{key}": value for key, value in out.items()}


def load_frozen(directory: str = CROPPER_DIR):
    page = cv2.imread(os.path.join(directory, PAGE_FILE), cv2.IMREAD_COLOR)
    with open(os.path.join(directory, CASES_FILE)) as f:
        cases = json.load(f)
    with np.load(os.path.join(directory, CROPS_FILE)) as data:
        crops = {key: data[key] for key in data.files}
    return page, cases, crops
