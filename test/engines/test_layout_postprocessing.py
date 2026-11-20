"""LAYOUT_POSTPROCESSING (LayoutPostprocessor) as a black box: with RETRACE_REGIONS, every region polygon is
replaced by an outline traced around the region's line polygons (layout_helpers.retrace_region: an alpha shape over
the line polygon points, via a Delaunay triangulation and shapely, simplified by 5 px; a convex hull when the
outline falls apart)."""
import numpy as np
import pytest
import shapely.geometry as sg
import torch

from pero_ocr.document_ocr.page_parser import layout_parser_factory
from test.configs import section
from test.layouts import box_region, page, rotated, straight_line, textline


def postprocess(layout, retrace='yes'):
    config = section(METHOD='LAYOUT_POSTPROCESSING', RETRACE_REGIONS=retrace)
    return layout_parser_factory(config, torch.device('cpu')).process_page(None, layout)


def outline(region):
    return sg.Polygon(region.polygon)


def contains_its_lines(region, tolerance=5.5):
    """Every line polygon lies inside the region, up to the 5 px simplification."""
    return all(outline(region).buffer(tolerance).contains(sg.Polygon(line.polygon)) for line in region.lines)


def test_retrace_around_the_lines():
    lines = [straight_line('l1', 200, 400, 100), straight_line('l2', 200, 450, 140)]
    region = box_region('r1', 0, 0, 800, 600, lines)
    postprocess(page(region))
    assert contains_its_lines(region)
    minx, miny, maxx, maxy = outline(region).bounds
    assert (minx, miny, maxx, maxy) == pytest.approx((200, 80, 450, 145), abs=6)


def test_retrace_a_tilted_region():
    angle, centre = 8.0, (400.0, 300.0)
    lines = [textline(f'l{y}', rotated([[150, y], [650, y]], angle, centre)) for y in (200, 240, 280, 320)]
    region = box_region('r1', 0, 0, 800, 600, lines)
    postprocess(page(region))
    assert contains_its_lines(region)
    hull = sg.MultiPolygon([sg.Polygon(line.polygon) for line in lines]).convex_hull
    assert outline(region).area == pytest.approx(hull.area, rel=0.1)  # the outline is tilted with the lines


def test_retrace_lines_far_apart_gives_one_outline_around_all():
    """Lines too far apart for one alpha shape: the region becomes the convex hull of the pieces."""
    region = box_region('r1', 0, 0, 800, 600, [straight_line('l1', 100, 300, 100), straight_line('l2', 500, 700, 500)])
    postprocess(page(region))
    assert outline(region).is_valid and contains_its_lines(region)


def test_retrace_every_region():
    regions = [box_region(f'r{i}', 0, 0, 800, 600, [straight_line(f'l{i}', 100 + 300 * i, 300 + 300 * i, 200)])
               for i in range(2)]
    postprocess(page(*regions))
    assert [round(outline(region).bounds[0]) for region in regions] == [100, 400]


@pytest.mark.xfail(reason='retrace_region takes the maximum point spacing over the lines of a region, which fails '
                          'with ValueError for a region without lines (e.g. a YOLO image region or a whole-page '
                          'region), and stops the whole page', strict=True)
def test_regions_without_lines_keep_their_polygon():
    image = box_region('image', 600, 400, 750, 550, category='image')
    text = box_region('r1', 0, 0, 800, 600, [straight_line('l1', 100, 300, 100)])
    postprocess(page(text, image))
    np.testing.assert_array_equal(image.polygon, [[600, 400], [750, 400], [750, 550], [600, 550]])


def test_without_retrace_nothing_changes():
    region = box_region('r1', 0, 0, 800, 600, [straight_line('l1', 200, 400, 100)])
    polygon = region.polygon.copy()
    postprocess(page(region), retrace='no')
    np.testing.assert_array_equal(region.polygon, polygon)


def test_empty_page(capsys):
    assert postprocess(page()).regions == []
    assert 'Skipping layout post processing' in capsys.readouterr().out
