"""LINE_POSTPROCESSING (LinePostprocessor and PostprocessingEngine) as a black box: the lines of every region are
stretched to the region border or by a number of pixels (STRETCH_LINES), refitted to 10 points (RESAMPLE_LINES), or
given heights from their region (HEIGHTS_FROM_REGIONS); line polygons are then rebuilt from baselines and heights.

Stretching works in the frame of the region's tilt, estimated from its lines, so it is tested on tilted regions too."""
import numpy as np
import pytest
import shapely.geometry as sg
import torch

from pero_ocr.document_ocr.page_parser import layout_parser_factory
from test.configs import section
from test.layouts import box_region, page, rotated, straight_line, textline

OFF = dict(STRETCH_LINES=0, RESAMPLE_LINES='no', HEIGHTS_FROM_REGIONS='no')
CENTRE = (400.0, 300.0)


def postprocess(layout, **options):
    config = section(METHOD='LINE_POSTPROCESSING', **{**OFF, **options})
    return layout_parser_factory(config, torch.device('cpu')).process_page(None, layout)


def tilted_region(angle, ys=(250, 300, 350), x0=300, x1=500):
    """A 600 x 300 region with lines from x0 to x1, all tilted by angle degrees around the page centre."""
    region = box_region('r1', 100, 150, 700, 450)
    region.polygon = rotated(region.polygon, angle, CENTRE)
    region.lines = [textline(f'l{y}', rotated([[x0, y], [x1, y]], angle, CENTRE)) for y in ys]
    return region


def distance_along(baseline, angle):
    """Positions of the baseline points along the direction of the tilt."""
    return baseline @ np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])


def assert_on_its_straight_line(baseline, original):
    direction = (original[-1] - original[0]) / np.linalg.norm(original[-1] - original[0])
    offsets = (baseline - original[0]) @ np.array([-direction[1], direction[0]])
    np.testing.assert_allclose(offsets, 0, atol=1.0)


# --- STRETCH_LINES ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('angle', [0, pytest.param(10, marks=pytest.mark.xfail(
    reason='PostprocessingEngine.stretch_baselines rotates the baselines and the region into the frame of the tilt, '
           'but then takes the x extent and the intersection from the unrotated region, so on a tilted region with '
           'two or more lines the stretched lines stick out on one side and fall short on the other', strict=True))])
def test_stretch_to_the_region_border(angle):
    region = tilted_region(angle)
    originals = [line.baseline.copy() for line in region.lines]
    postprocess(page(region), STRETCH_LINES='max')
    border = sg.Polygon(region.polygon).exterior
    for line, original in zip(region.lines, originals):
        assert_on_its_straight_line(line.baseline, original)
        for end in (line.baseline[0], line.baseline[-1]):
            assert border.distance(sg.Point(end)) < 1.0


def test_stretch_a_single_tilted_line_to_the_region_border():
    """With one line, the estimated tilt is 0 (only the longer half of the lines counts), and the line is stretched
    along itself in the page frame."""
    region = tilted_region(10, ys=(300,))
    original = region.lines[0].baseline.copy()
    postprocess(page(region), STRETCH_LINES='max')
    assert_on_its_straight_line(region.lines[0].baseline, original)
    border = sg.Polygon(region.polygon).exterior
    assert all(border.distance(sg.Point(end)) < 1.0 for end in region.lines[0].baseline[[0, -1]])


@pytest.mark.parametrize('angle', [0, 10])
def test_stretch_by_pixels(angle):
    """STRETCH_LINES = 30 adds a point 30 px before the start and after the end of every line, along the tilt."""
    region = tilted_region(angle)
    originals = [line.baseline.copy() for line in region.lines]
    postprocess(page(region), STRETCH_LINES=30)
    for line, original in zip(region.lines, originals):
        assert len(line.baseline) == len(original) + 2
        assert_on_its_straight_line(line.baseline, original)
        along, original_along = distance_along(line.baseline, angle), distance_along(original, angle)
        assert (along[0], along[-1]) == pytest.approx((original_along[0] - 30, original_along[-1] + 30), abs=1.0)


# --- RESAMPLE_LINES -----------------------------------------------------------------------------------------------

def test_resample_lines_to_ten_points():
    region = box_region('r1', 50, 50, 750, 200, [straight_line('l1', 200, 400, 100)])
    postprocess(page(region), RESAMPLE_LINES='yes')
    baseline = region.lines[0].baseline
    np.testing.assert_allclose(baseline, np.stack([np.linspace(200, 400, 10), np.full(10, 100.0)], axis=1))


def test_resample_keeps_a_curved_line_curved():
    """Baselines with more than two points are refitted with a parabola."""
    region = box_region('r1', 0, 0, 800, 600, [textline('l1', [[100, 300], [400, 250], [700, 300]])])
    postprocess(page(region), RESAMPLE_LINES='yes')
    baseline = region.lines[0].baseline
    x = baseline[:, 0]
    np.testing.assert_allclose(baseline[:, 1], 250 + 50 * ((x - 400) / 300) ** 2, atol=1e-6)


# --- line polygons ------------------------------------------------------------------------------------------------

def test_line_polygons_follow_the_new_baselines():
    region = box_region('r1', 50, 50, 750, 200, [straight_line('l1', 200, 400, 100, heights=(20.0, 5.0))])
    postprocess(page(region), STRETCH_LINES='max')
    polygon = region.lines[0].polygon
    assert (polygon[:, 0].min(), polygon[:, 0].max()) == pytest.approx((50, 750), abs=1)
    assert (polygon[:, 1].min(), polygon[:, 1].max()) == pytest.approx((80, 105), abs=1)


def test_all_off_changes_nothing():
    region = tilted_region(10)
    before = [(line.baseline.copy(), line.polygon.copy()) for line in region.lines]
    postprocess(page(region))
    for line, (baseline, polygon) in zip(region.lines, before):
        np.testing.assert_array_equal(line.baseline, baseline)
        np.testing.assert_array_equal(line.polygon, polygon)


# --- HEIGHTS_FROM_REGIONS -----------------------------------------------------------------------------------------

@pytest.mark.xfail(reason="PostprocessingEngine.get_heights_from_regions scores lines with an undefined name "
                          "'baseline' (meant: line.baseline), so HEIGHTS_FROM_REGIONS = yes always raises NameError",
                   strict=True)
def test_heights_from_regions_keep_the_biggest_line():
    """For regions that are one line each (e.g. from a word or line detector): the region box gives the heights."""
    lines = [straight_line('short', 300, 350, 150), straight_line('long', 100, 700, 160)]
    region = box_region('r1', 50, 100, 750, 180, lines)
    postprocess(page(region), HEIGHTS_FROM_REGIONS='yes')
    kept, = region.lines
    assert kept.id == 'long' and tuple(kept.heights) == (60, 20)
    assert (kept.polygon[:, 1].min(), kept.polygon[:, 1].max()) == pytest.approx((100, 180), abs=1)


# --- pages and regions --------------------------------------------------------------------------------------------

def test_every_region_is_processed_and_empty_ones_are_left_alone():
    first = box_region('r1', 50, 50, 750, 200, [straight_line('l1', 200, 400, 100)])
    empty = box_region('r2', 50, 250, 750, 300)
    second = box_region('r3', 50, 350, 750, 500, [straight_line('l2', 300, 600, 400)])
    postprocess(page(first, empty, second), STRETCH_LINES='max')
    assert [tuple(region.lines[0].baseline[[0, -1], 0]) for region in (first, second)] == [(50, 750), (50, 750)]
    assert empty.lines == []


def test_empty_page(capsys):
    assert postprocess(page(), STRETCH_LINES='max').regions == []
    assert 'Skipping line post processing' in capsys.readouterr().out
