"""The ParseNet layout engine (LayoutEngine in pero_ocr.layout_engines.cnn_layout_engine), inside the LAYOUT_CNN
engine, with the tiny ParseNet: lines and their heights, line ends, separators between paragraphs, columns,
rotated detection and the adaptive downsampling."""
import numpy as np
import pytest

from pero_ocr.layout_engines.cnn_layout_engine import LayoutEngine
from test import tiny_models
from test.synthetic import layout_page, page_image

LINE_TOPS = [100, 200, 300]  # page_image(3) defaults: top=100, line_height=40, line_gap=60
COLUMN = [(100, 80 + 60 * i, 700, 112 + 60 * i) for i in range(6)]  # six 32 px lines, 28 px apart
SEPARATOR_ABOVE_LINE_4 = (100, 252, 700, 260)  # along the top of the fourth line
PRODUCTION = dict(downsample=3, detection_threshold=0.3, max_mp=5, line_end_weight=1.0,
                  vertical_line_connection_range=3, smooth_line_predictions=False)


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
        np.testing.assert_allclose(h, [40, 40 * tiny_models.PARSENET_DESCENDER_RATIO])  # 40 px tall lines


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


def make(parsenet_path, cpu, **options):
    return LayoutEngine(parsenet_path, cpu, **{'downsample': 4, 'detection_threshold': 0.2, 'max_mp': 5, **options})


# --- what the ParseNet maps steer -----------------------------------------------------------------------------------

def test_heights_follow_the_size_of_the_text(engine):
    _, _, heights, _ = engine.detect(layout_page(text=[(100, 100, 700, 124), (100, 300, 700, 348)]))
    np.testing.assert_allclose(heights, [[24, 24 * 0.3], [48, 48 * 0.3]], rtol=1e-6)


@pytest.mark.parametrize('options, n_lines', [
    ({}, 2),
    ({'line_end_weight': 0.0}, 1),  # line ends ignored
    (PRODUCTION, 2),
    ({**PRODUCTION, 'detection_threshold': 0.4, 'line_end_weight': 0.6}, 2),  # the 2024 experimental engine
])
def test_line_end_cuts_a_line(parsenet_path, cpu, options, n_lines):
    page = layout_page(text=[(100, 100, 700, 140)], line_ends=[(392, 100, 408, 140)])
    _, baselines, _, _ = make(parsenet_path, cpu, **options).detect(page)
    assert len(baselines) == n_lines


@pytest.mark.parametrize('options', [{}, PRODUCTION], ids=['default', 'production'])
def test_separator_splits_a_column_into_regions(parsenet_path, cpu, options):
    engine = make(parsenet_path, cpu, **options)
    regions, baselines, _, _ = engine.detect(layout_page(text=COLUMN))
    assert (len(regions), len(baselines)) == (1, 6)
    regions, baselines, _, _ = engine.detect(layout_page(text=COLUMN, separators=[SEPARATOR_ABOVE_LINE_4]))
    assert (len(regions), len(baselines)) == (2, 6)
    tops = sorted(region[:, 1].min() for region in regions)
    assert tops[0] < 80 and 240 < tops[1] < 264


def test_columns_are_separate_regions(engine):
    left = [(80, 100 + 50 * i, 380, 130 + 50 * i) for i in range(4)]
    right = [(420, 100 + 50 * i, 720, 130 + 50 * i) for i in range(4)]
    regions, _, _, _ = engine.detect(layout_page(text=left + right))
    assert sorted(region[:, 0].min() > 400 for region in regions) == [False, True]


def test_large_text_is_detected_again_at_a_coarser_downsample(parsenet_path, cpu):
    """With ADAPTIVE_DOWNSAMPLE (the default), lines more than 15 map pixels tall send the page through the network
    again, downsampled so that they are about 12 pixels tall (at most by 8)."""
    engine = make(parsenet_path, cpu)
    _, _, heights, _ = engine.detect(layout_page(text=[(100, 60 + 160 * i, 700, 160 + 160 * i) for i in range(3)]))
    assert engine.parsenet.last_downsample == 8
    assert all(abs(h[0] - 100) <= 8 for h in heights)


def test_the_adapted_downsample_carries_over_to_the_next_page(parsenet_path, cpu):
    """The engine starts every page at the downsample it adapted to on the previous one, so the same page can come out
    differently depending on the page before it (#96, Weird behaviour). Pinned as current behaviour."""
    page = layout_page(text=[(100, 100, 700, 140)])
    _, fresh_baselines, fresh_heights, _ = make(parsenet_path, cpu).detect(page)
    engine = make(parsenet_path, cpu)
    engine.detect(layout_page(text=[(100, 60 + 160 * i, 700, 160 + 160 * i) for i in range(3)]))
    _, baselines, heights, _ = engine.detect(page)
    np.testing.assert_allclose(fresh_heights, [[40, 12]], rtol=1e-6)
    np.testing.assert_allclose(heights, [[48, 48 * 0.3]], rtol=1e-6)
    assert fresh_baselines[0][0, 1] != baselines[0][0, 1]


@pytest.mark.xfail(reason='LayoutEngine.parse sorts lines left to right with random noise (np.random.rand) to break ties, '
                          'and the regions are numbered in that order, so lines that start at the same x make the '
                          'region order and ids random (#100)', strict=True)
def test_region_order_does_not_depend_on_the_random_state(parsenet_path, cpu):
    page = layout_page(text=COLUMN, separators=[SEPARATOR_ABOVE_LINE_4])  # every line starts at x = 100
    state = np.random.get_state()
    try:
        first_region_tops = set()
        for seed in range(6):
            np.random.seed(seed)
            regions, _, _, _ = make(parsenet_path, cpu, **PRODUCTION).detect(page)
            first_region_tops.add(int(regions[0][:, 1].min()))
    finally:
        np.random.set_state(state)
    assert len(first_region_tops) == 1
