"""REGION_WHOLE_PAGE (WholePageRegion in pero_ocr.document_ocr.page_parser) as a black box: it replaces all regions
with one region over the whole page, without lines. Followed by a stage that puts lines into existing regions (e.g.
LAYOUT_CNN with DETECT_REGIONS = no), it forces one region, so that the layout engine only gives lines."""
import numpy as np
import torch

from pero_ocr.document_ocr.page_parser import layout_parser_factory
from test.configs import section
from test.layouts import box_region, page, straight_line


def whole_page(layout):
    return layout_parser_factory(section(METHOD='REGION_WHOLE_PAGE'), torch.device('cpu')).process_page(None, layout)


def test_replaces_all_regions_with_one_over_the_page():
    text = box_region('a', 10, 10, 50, 50, [straight_line('a-l0', 15, 45, 40)])
    image = box_region('b', 0, 0, 5, 5, category='image')
    whole, = whole_page(page(text, image, page_size=(600, 800))).regions
    assert whole.id == 'r1' and whole.lines == [] and whole.category is None
    np.testing.assert_array_equal(whole.polygon, [[0, 0], [800, 0], [800, 600], [0, 600]])  # x, y: width first


def test_empty_page():
    whole, = whole_page(page(page_size=(1000, 300))).regions
    np.testing.assert_array_equal(whole.polygon, [[0, 0], [300, 0], [300, 1000], [0, 1000]])
