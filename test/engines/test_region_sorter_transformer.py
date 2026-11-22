"""The transformer reading-order sorter (REGION_SORTER_TRANSFORMER, pero_ocr.layout_engines.transformer_sorter).

The engine turns the regions of a page into model inputs (boxes normalized to [0, 1], query types, the resized page
image), lets the model score which region follows which, and orders the regions along the cheapest path through those
scores. The tiny model (test/tiny_models.py) orders regions by their left edge, then by their top edge."""
import configparser
import os
import shutil

import numpy as np
import pytest
import torch

from pero_ocr.core.layout import PageLayout, RegionLayout
from pero_ocr.layout_engines.transformer_sorter import QueryType, TransformerRegionSorter, page_layout_to_model_inputs
from test.synthetic import mark_reading_order

PAGE_SIZE = (1000, 800)  # height, width
IMAGE = np.zeros(PAGE_SIZE + (3,), dtype=np.uint8)
TWO_COLUMNS = [(420, 500, 760, 700), (40, 300, 380, 480), (420, 50, 760, 450), (40, 50, 380, 250)]


def make_sorter(model_path, max_bbox_count=32, width=64, height=64, use_cpu='yes'):
    config = configparser.ConfigParser()
    config.read_string(f'[S]\nMODEL_PATH = {os.path.basename(model_path)}\nMAX_BBOX_COUNT = {max_bbox_count}\n'
                       f'IMAGE_WIDTH = {width}\nIMAGE_HEIGHT = {height}\nUSE_CPU = {use_cpu}\n')
    return TransformerRegionSorter(config['S'], torch.device('cpu'), config_path=os.path.dirname(model_path))


def page(boxes, page_size=PAGE_SIZE):
    layout = PageLayout(id='p', page_size=page_size)
    layout.regions = [RegionLayout(f'r{i}', np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float64))
                      for i, (l, t, r, b) in enumerate(boxes)]
    return layout


def ids(layout):
    return [region.id for region in layout.regions]


class RecordingModel:
    """Wraps the loaded model and records what the engine passes to it."""

    def __init__(self, model):
        self.model, self.calls = model, []

    def __call__(self, *args):
        self.calls.append(args)
        return self.model(*args)


# --- ordering -----------------------------------------------------------------------------------------------------

def test_orders_regions_as_the_model_says(region_sorter_path):
    layout = make_sorter(region_sorter_path).process_page(IMAGE, page(TWO_COLUMNS))
    assert ids(layout) == ['r3', 'r1', 'r2', 'r0']  # left column top to bottom, then the right one


def test_the_image_can_impose_an_order_against_the_geometry(region_sorter_path):
    """What a learned model does and a geometric sorter cannot: the tiny model reads reading-order marks from the page
    image (here: right column first, bottom to top), and regions without a mark follow by geometry."""
    image = mark_reading_order(IMAGE.copy(), [TWO_COLUMNS[0], TWO_COLUMNS[2]])
    layout = make_sorter(region_sorter_path).process_page(image, page(TWO_COLUMNS))
    assert ids(layout) == ['r0', 'r2', 'r3', 'r1']


@pytest.mark.parametrize('width, height', [(64, 64), (32, 96)])
def test_marks_survive_the_resizing_of_the_image(region_sorter_path, width, height):
    image = mark_reading_order(IMAGE.copy(), [TWO_COLUMNS[i] for i in (1, 0, 3, 2)])
    layout = make_sorter(region_sorter_path, width=width, height=height).process_page(image, page(TWO_COLUMNS))
    assert ids(layout) == ['r1', 'r0', 'r3', 'r2']


def test_more_than_eight_regions(region_sorter_path):
    """Above 8 regions, the path search first merges the most certain pairs, then tries all orders of the rest."""
    order = [7, 2, 10, 0, 5, 11, 3, 9, 1, 6, 8, 4]
    boxes = [(40, 20 + 80 * i, 380, 90 + 80 * i) for i in order]
    layout = make_sorter(region_sorter_path).process_page(IMAGE, page(boxes))
    assert [order[int(region_id[1:])] for region_id in ids(layout)] == list(range(12))


@pytest.mark.parametrize('n', [0, 1])
def test_fewer_than_two_regions_are_left_alone(region_sorter_path, n):
    sorter = make_sorter(region_sorter_path)
    sorter.model = RecordingModel(sorter.model)
    layout = sorter.process_page(IMAGE, page(TWO_COLUMNS[:n]))
    assert ids(layout) == [f'r{i}' for i in range(n)] and not sorter.model.calls


def test_regions_keep_their_content(region_sorter_path):
    layout = page(TWO_COLUMNS)
    before = {region.id: region for region in layout.regions}
    layout = make_sorter(region_sorter_path).process_page(IMAGE, layout)
    assert all(region is before[region.id] for region in layout.regions)


# --- model inputs -------------------------------------------------------------------------------------------------

def test_model_inputs(region_sorter_path):
    sorter = make_sorter(region_sorter_path, max_bbox_count=8)
    sorter.model = RecordingModel(sorter.model)
    boxes = [(40, 50, 380, 250), (420, -30, 900, 450)]  # the second one reaches outside the page
    sorter.process_page(IMAGE, page(boxes))
    (bboxes, query_types, image), = sorter.model.calls
    assert bboxes.shape == (1, 8, 4) and bboxes.dtype == torch.float32
    np.testing.assert_allclose(bboxes[0, :2].numpy(), [[0.05, 0.05, 0.475, 0.25], [0.525, 0.0, 1.0, 0.45]], rtol=1e-6)
    assert not bboxes[0, 2:].any()
    assert query_types.tolist() == [[QueryType.BBOX.value] * 2 + [QueryType.PADDING.value] * 6]
    assert image.dtype == torch.uint8 and image.shape[:2] == (1, 3)


def test_model_inputs_are_trimmed_to_max_bbox_count():
    boxes, query_types = page_layout_to_model_inputs(page([(40, 20 + 60 * i, 380, 70 + 60 * i) for i in range(5)]),
                                                     max_bbox_count=3)
    assert boxes.shape == (3, 4) and query_types.tolist() == [QueryType.BBOX.value] * 3


@pytest.mark.xfail(reason='TransformerRegionSorter resizes the image with (IMAGE_HEIGHT, IMAGE_WIDTH), but cv2.resize '
                          'takes (width, height); harmless for the square 1024 x 1024 production config',
                   strict=True)
def test_image_has_the_configured_size(region_sorter_path):
    sorter = make_sorter(region_sorter_path, width=96, height=48)
    sorter.model = RecordingModel(sorter.model)
    sorter.process_page(IMAGE, page(TWO_COLUMNS))
    assert tuple(sorter.model.calls[0][2].shape) == (1, 3, 48, 96)


@pytest.mark.xfail(reason='regions beyond MAX_BBOX_COUNT are appended from a set, i.e. in a random order', strict=True)
def test_regions_beyond_max_bbox_count_keep_their_order(region_sorter_path):
    """The model sees only the first MAX_BBOX_COUNT regions; the rest should follow in their original order.
    With 7 leftover regions, a random order passes by chance once in 5040 runs."""
    boxes = [(40, 20 + 60 * i, 380, 70 + 60 * i) for i in range(10)]
    layout = make_sorter(region_sorter_path, max_bbox_count=3).process_page(IMAGE, page(boxes))
    assert ids(layout) == [f'r{i}' for i in range(10)]


# --- model files --------------------------------------------------------------------------------------------------

def test_on_cpu_the_cpu_model_file_is_loaded(region_sorter_path, tmp_path):
    """On CPU, 'cuda' in the file name is replaced by 'cpu': the config names model-cuda.pt, model-cpu.pt is loaded."""
    shutil.copyfile(region_sorter_path.replace('cuda', 'cpu'), tmp_path / 'model-cpu.pt')
    sorter = make_sorter(str(tmp_path / 'model-cuda.pt'))
    assert sorter.model_path == str(tmp_path / 'model-cpu.pt')
    assert ids(sorter.process_page(IMAGE, page(TWO_COLUMNS))) == ['r3', 'r1', 'r2', 'r0']


@pytest.mark.xfail(reason='on CPU the sorter only looks for model-cpu.pt, but a production package ships '
                          'only model-cuda.pt, which torch can load on CPU too',
                   strict=True)
def test_cuda_only_package_runs_on_cpu(region_sorter_path, tmp_path):
    shutil.copyfile(region_sorter_path, tmp_path / 'model-cuda.pt')
    sorter = make_sorter(str(tmp_path / 'model-cuda.pt'))
    assert ids(sorter.process_page(IMAGE, page(TWO_COLUMNS))) == ['r3', 'r1', 'r2', 'r0']
