"""Layout stages without a model, as black boxes (config section and PageLayout in, PageLayout out):
REGION_WHOLE_PAGE (WholePageRegion), LINE_POSTPROCESSING (LinePostprocessor) and LAYOUT_POSTPROCESSING
(LayoutPostprocessor), all in pero_ocr.document_ocr.page_parser."""
import numpy as np
import pytest
import torch

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr.page_parser import layout_parser_factory
from test.configs import section

PAGE_SIZE = (600, 800)  # height, width
IMAGE = np.zeros(PAGE_SIZE + (3,), dtype=np.uint8)


def make(method, **options):
    return layout_parser_factory(section(METHOD=method, **options), torch.device('cpu'))


def line(line_id, x0, x1, y, heights=(20.0, 5.0)):
    return TextLine(id=line_id, baseline=np.array([[x0, y], [x1, y]], dtype=np.float64), heights=list(heights),
                    polygon=np.array([[x0, y - heights[0]], [x1, y - heights[0]], [x1, y + heights[1]],
                                      [x0, y + heights[1]]]))


def region(region_id, l, t, r, b, lines=()):
    result = RegionLayout(region_id, np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float64))
    result.lines = list(lines)
    return result


def page(*regions):
    layout = PageLayout(id='p', page_size=PAGE_SIZE)
    layout.regions = list(regions)
    return layout


# --- REGION_WHOLE_PAGE --------------------------------------------------------------------------------------------

def test_whole_page_region_replaces_all_regions():
    layout = make('REGION_WHOLE_PAGE').process_page(IMAGE, page(region('a', 10, 10, 50, 50), region('b', 0, 0, 5, 5)))
    whole, = layout.regions
    assert whole.id == 'r1' and whole.lines == []
    np.testing.assert_array_equal(whole.polygon, [[0, 0], [800, 0], [800, 600], [0, 600]])


# --- LINE_POSTPROCESSING ------------------------------------------------------------------------------------------

LINE_OFF = dict(STRETCH_LINES=0, RESAMPLE_LINES='no', HEIGHTS_FROM_REGIONS='no')


def postprocess(layout, **options):
    return make('LINE_POSTPROCESSING', **{**LINE_OFF, **options}).process_page(IMAGE, layout)


def test_stretch_lines_to_the_region():
    layout = postprocess(page(region('r1', 50, 50, 750, 200, [line('l1', 200, 400, 100)])), STRETCH_LINES='max')
    baseline = layout.regions[0].lines[0].baseline
    assert baseline[:, 0].min() == pytest.approx(50, abs=1) and baseline[:, 0].max() == pytest.approx(750, abs=1)
    np.testing.assert_allclose(baseline[:, 1], 100, atol=1)


def test_stretch_lines_by_pixels():
    layout = postprocess(page(region('r1', 50, 50, 750, 200, [line('l1', 200, 400, 100)])), STRETCH_LINES=30)
    baseline = layout.regions[0].lines[0].baseline
    assert (baseline[0, 0], baseline[-1, 0]) == pytest.approx((170, 430), abs=1)


def test_resample_lines():
    layout = postprocess(page(region('r1', 50, 50, 750, 200, [line('l1', 200, 400, 100)])), RESAMPLE_LINES='yes')
    baseline = layout.regions[0].lines[0].baseline
    assert len(baseline) == 10
    np.testing.assert_allclose(baseline[:, 0], np.linspace(200, 400, 10))


@pytest.mark.xfail(reason="PostprocessingEngine.get_heights_from_regions scores lines with an undefined name "
                          "'baseline' (meant: line.baseline), so HEIGHTS_FROM_REGIONS = yes always raises NameError",
                   strict=True)
def test_heights_from_regions_keep_the_biggest_line():
    """For regions that are one line each (e.g. from a word or line detector): the region box gives the heights."""
    lines = [line('short', 300, 350, 150), line('long', 100, 700, 160)]
    layout = postprocess(page(region('r1', 50, 100, 750, 180, lines)), HEIGHTS_FROM_REGIONS='yes')
    kept, = layout.regions[0].lines
    assert kept.id == 'long' and tuple(kept.heights) == (60, 20)
    assert (kept.polygon[:, 1].min(), kept.polygon[:, 1].max()) == pytest.approx((100, 180), abs=1)


def test_line_postprocessing_leaves_regions_without_lines_and_empty_pages(capsys):
    empty_region = region('r1', 50, 50, 750, 200)
    assert postprocess(page(empty_region), STRETCH_LINES='max').regions == [empty_region]
    assert postprocess(page(), STRETCH_LINES='max').regions == []
    assert 'Skipping line post processing' in capsys.readouterr().out


# --- LAYOUT_POSTPROCESSING ----------------------------------------------------------------------------------------

def test_retrace_regions_around_their_lines():
    lines = [line('l1', 200, 400, 100), line('l2', 200, 450, 140)]
    layout = make('LAYOUT_POSTPROCESSING', RETRACE_REGIONS='yes').process_page(
        IMAGE, page(region('r1', 0, 0, 800, 600, lines)))
    polygon = layout.regions[0].polygon
    assert polygon[:, 0].min() >= 150 and polygon[:, 0].max() <= 500
    assert polygon[:, 1].min() >= 50 and polygon[:, 1].max() <= 190


def test_layout_postprocessing_without_retrace_changes_nothing(capsys):
    original = region('r1', 0, 0, 800, 600, [line('l1', 200, 400, 100)])
    polygon = original.polygon.copy()
    layout = make('LAYOUT_POSTPROCESSING', RETRACE_REGIONS='no').process_page(IMAGE, page(original))
    np.testing.assert_array_equal(layout.regions[0].polygon, polygon)
    assert make('LAYOUT_POSTPROCESSING', RETRACE_REGIONS='yes').process_page(IMAGE, page()).regions == []
    assert 'Skipping layout post processing' in capsys.readouterr().out
