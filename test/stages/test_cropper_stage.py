"""The [LINE_CROPPER] stage (line_cropper_factory and LineCropper in pero_ocr.document_ocr.page_parser): a config
section and a PageLayout in, the lines of the page cropped. There is only one cropper engine and no METHOD; the
engine itself (EngineLineCropper: the geometry of the crops) is tested in test/engines/test_line_cropper.py."""
import numpy as np
import pytest

from pero_ocr.core.crop_engine import EngineLineCropper
from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr.page_parser import LineCropper, line_cropper_factory
from test.configs import section

PAGE_SHAPE = (600, 800)
HEIGHTS = [20, 8]
PAGE = np.random.default_rng(20261121).integers(0, 256, PAGE_SHAPE + (3,), dtype=np.uint8)


def make(**options):
    return line_cropper_factory(section(**{'LINE_HEIGHT': 40, 'INTERP': 2, 'LINE_SCALE': 1.0, **options}))


def page_with_lines(*categories):
    layout = PageLayout(id='p', page_size=PAGE_SHAPE)
    region = RegionLayout('r1', np.array([[0, 0], [800, 0], [800, 600], [0, 600]]))
    for i, category in enumerate(categories):
        y = 100.0 + 80 * i
        region.lines.append(TextLine(id=f'r1-l{i}', baseline=np.array([[100.0, y], [500.0, y]]), heights=HEIGHTS,
                                     category=category))
    layout.regions = [region]
    return layout


def crops(layout):
    return [line.crop for line in layout.lines_iterator()]


def test_factory_builds_a_line_cropper():
    assert isinstance(make(), LineCropper)


@pytest.mark.parametrize('options, engine', [
    ({}, dict(line_height=40, poly=2, scale=1.0)),
    (dict(LINE_HEIGHT=48, INTERP=0, LINE_SCALE=1.15), dict(line_height=48, poly=0, scale=1.15)),
])
def test_options_reach_the_engine(options, engine):
    """LINE_HEIGHT, INTERP and LINE_SCALE are the engine's line_height, poly and scale; every line gets the engine's
    crop."""
    layout = make(**options).process_page(PAGE, page_with_lines('text', 'text'))
    for line in layout.lines_iterator():
        np.testing.assert_array_equal(line.crop, EngineLineCropper(**engine).crop(PAGE, line.baseline, line.heights))


def test_categories():
    """Only lines of the CATEGORIES are cropped; lines without a category pass every filter (PageLayout.lines_iterator).
    Without CATEGORIES every line is cropped."""
    assert [c is not None for c in crops(make(CATEGORIES=['title']).process_page(
        PAGE, page_with_lines('text', 'title', None)))] == [False, True, True]
    assert all(c is not None for c in crops(make().process_page(PAGE, page_with_lines('text', 'title', None))))


def test_page_without_lines():
    layout = make().process_page(PAGE, page_with_lines())
    assert layout.regions[0].lines == []


def test_a_line_the_engine_rejects_gets_a_black_square_crop(monkeypatch, capsys):
    """When the engine raises ValueError for a line, the line gets a black LINE_HEIGHT x LINE_HEIGHT crop and the other
    lines are still cropped. (The current engine catches its own errors and returns a blank crop instead, so this
    fallback is not reached in practice.)"""
    stage = make()
    original = stage.crop_engine.crop

    def crop(img, baseline, heights):
        if baseline[0, 1] == 100.0:
            raise ValueError('vertical line')
        return original(img, baseline, heights)

    monkeypatch.setattr(stage.crop_engine, 'crop', crop)
    rejected, cropped = crops(stage.process_page(PAGE, page_with_lines('text', 'text')))
    assert rejected.shape == (40, 40, 3) and rejected.max() == 0
    assert cropped.shape[0] == 40 and cropped.max() > 0
    assert 'Failed to crop line r1-l0 in page p' in capsys.readouterr().out


def test_crop_lines_crops_the_given_lines_regardless_of_category():
    lines = list(page_with_lines('text', 'title').lines_iterator())
    make(CATEGORIES=['text']).crop_lines(PAGE, lines)
    assert all(line.crop is not None and line.crop.shape[0] == 40 for line in lines)
