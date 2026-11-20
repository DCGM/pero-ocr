import numpy as np
import pytest

from pero_ocr.core.crop_engine import EngineLineCropper


@pytest.fixture
def page():
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    img[100:140, 50:550] = 255
    return img


@pytest.mark.parametrize('line_height', [32, 40, 48])
def test_crop_has_requested_height(page, line_height):
    cropper = EngineLineCropper(line_height=line_height, poly=2, scale=1.25)
    crop = cropper.crop(page, np.array([[50.0, 135.0], [550.0, 135.0]]), [30.0, 8.0])
    assert crop.shape[0] == line_height
    assert crop.shape[2] == 3
    assert crop.dtype == np.uint8


def test_crop_contains_the_line(page):
    cropper = EngineLineCropper(line_height=40, poly=0, scale=1)
    crop = cropper.crop(page, np.array([[50.0, 135.0], [550.0, 135.0]]), [30.0, 8.0])
    assert crop.mean() > 100
    assert crop.shape[1] > 300


def test_crop_width_follows_baseline_length(page):
    cropper = EngineLineCropper(line_height=40, poly=0, scale=1)
    short = cropper.crop(page, np.array([[50.0, 135.0], [250.0, 135.0]]), [30.0, 8.0])
    long = cropper.crop(page, np.array([[50.0, 135.0], [550.0, 135.0]]), [30.0, 8.0])
    assert long.shape[1] > short.shape[1]


def test_crop_mapping(page):
    cropper = EngineLineCropper(line_height=40, poly=0, scale=1)
    crop, mapping, offset = cropper.crop(page, np.array([[50.0, 135.0], [550.0, 135.0]]), [30.0, 8.0],
                                         return_mapping=True)
    assert mapping.ndim == 3 and len(offset) == 2
    crop, forward = cropper.crop(page, np.array([[50.0, 135.0], [550.0, 135.0]]), [30.0, 8.0],
                                 return_forward_mapping=True)
    assert forward.shape[:2] == crop.shape[:2]


def test_line_partly_outside_the_page(page):
    cropper = EngineLineCropper(line_height=40, poly=0, scale=1)
    crop = cropper.crop(page, np.array([[-20.0, 10.0], [300.0, 10.0]]), [30.0, 8.0])
    assert crop.shape[0] == 40
