import os

import numpy as np
import pytest

from pero_ocr.layout_engines.cnn_layout_engine import LayoutEngine
from test import tiny_models
from test.conftest import FROZEN_MODELS
from test.synthetic import page_image

LINE_TOPS = [100, 200, 300]  # page_image(3) defaults: top=100, line_height=40, line_gap=60


@pytest.fixture(scope='module')
def engine(parsenet_path, cpu):
    return LayoutEngine(parsenet_path, cpu, downsample=4, detection_threshold=0.2, max_mp=5)


def test_detects_every_line(engine):
    polygons, baselines, heights, textlines = engine.detect(page_image(3))
    assert len(baselines) == len(heights) == len(textlines) == 3
    assert len(polygons) >= 1


def test_baselines_lie_at_the_bottom_of_the_lines(engine):
    _, baselines, _, _ = engine.detect(page_image(3))
    for top, baseline in zip(LINE_TOPS, sorted(baselines, key=lambda b: b[0, 1])):
        assert np.all(np.abs(baseline[:, 1] - (top + 40)) <= 8)
        assert baseline[:, 0].min() < 120 and baseline[:, 0].max() > 680


def test_heights_scale_with_downsampling(engine):
    _, _, heights, _ = engine.detect(page_image(3))
    for h in heights:
        np.testing.assert_allclose(h, [4 * tiny_models.PARSENET_ASCENDER, 4 * tiny_models.PARSENET_DESCENDER])


def test_textlines_enclose_baselines(engine):
    _, baselines, _, textlines = engine.detect(page_image(2))
    for baseline, textline in zip(baselines, textlines):
        assert textline[:, 1].min() < baseline[:, 1].min()
        assert textline[:, 1].max() > baseline[:, 1].max()


def test_empty_page(engine):
    assert engine.detect(np.zeros((600, 800, 3), dtype=np.uint8)) == ([], [], [], [])


def test_rotated_detection_returns_page_coordinates(engine):
    page = page_image(3)
    rotated = np.ascontiguousarray(np.rot90(page, k=-1))  # lines become vertical
    _, baselines, _, _ = engine.detect(rotated, rot=1)
    assert len(baselines) == 3
    for baseline in baselines:
        assert np.ptp(baseline[:, 0]) < 10  # vertical in the page
        assert 0 <= baseline[:, 0].min() and baseline[:, 0].max() <= rotated.shape[1]


def test_frozen_model_still_works(engine, cpu):
    """A model exported by an older pero-ocr/torch (test/engines/data) loads and gives the same output."""
    frozen = LayoutEngine(os.path.join(FROZEN_MODELS, 'parsenet', 'parsenet.pt'), cpu, downsample=4,
                          detection_threshold=0.2, max_mp=5)
    page = page_image(3)
    frozen_baselines, fresh_baselines = frozen.detect(page)[1], engine.detect(page)[1]
    assert len(frozen_baselines) == len(fresh_baselines) == 3
    for a, b in zip(frozen_baselines, fresh_baselines):
        np.testing.assert_allclose(a, b)
