"""The YOLO layout engine (LAYOUT_YOLO: LayoutExtractorYolo in pero_ocr.document_ocr.page_parser, LayoutEngineYolo in
pero_ocr.layout_engines.cnn_layout_engine): non-text regions such as images and stamps as boxes, optionally with one
line per box.

ultralytics.YOLO is replaced by test/fake_yolo.FakeYOLO, which returns fixed detections (see there for why); the tests
cover pero-ocr's side: reading the config, calling the network, and turning detections into regions."""
import configparser
import os
import sys

import numpy as np
import pytest
import torch

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr.page_parser import LayoutExtractorYolo, layout_parser_factory

PAGE_SIZE = (1000, 800)  # height, width
IMAGE = np.zeros(PAGE_SIZE + (3,), dtype=np.uint8)


def make_extractor(extra='', model_path='yolo.pt', config_path='', threshold=0.3, device=torch.device('cpu')):
    config = configparser.ConfigParser()
    config.read_string(f'[S]\nMETHOD = LAYOUT_YOLO\nMODEL_PATH = {model_path}\nDETECTION_THRESHOLD = {threshold}\n'
                       f'USE_CPU = yes\n{extra}')
    return layout_parser_factory(config['S'], device, config_path=config_path)


def new_page(regions=()):
    layout = PageLayout(id='p', page_size=PAGE_SIZE)
    layout.regions = list(regions)
    return layout


def box_region(region_id, l, t, r, b, category=None):
    return RegionLayout(region_id, np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float64), category=category)


def summary(layout):
    return [(region.id, region.category) for region in layout.regions]


# --- detections become regions ------------------------------------------------------------------------------------

def test_detections_become_regions(fake_yolo):
    layout = make_extractor().process_page(IMAGE, new_page())
    assert summary(layout) == [('r000', 'image'), ('r001', 'stamp'), ('r002', 'page_number')]
    image = layout.regions[0]
    # a closed rectangle: the first corner is repeated at the end
    np.testing.assert_array_equal(image.polygon, [[100, 120], [100, 400], [500, 400], [500, 120], [100, 120]])
    assert image.detection_confidence == pytest.approx(0.91) and image.lines == []


def test_detection_threshold_is_passed_to_the_network(fake_yolo):
    layout = make_extractor(threshold=0.05).process_page(IMAGE, new_page())
    assert len(layout.regions) == 4  # the weak stamp is in now
    assert fake_yolo.instances[0].calls[0]['conf'] == 0.05


def test_categories_filter(fake_yolo):
    layout = make_extractor('CATEGORIES = ["image", "page_number"]\n').process_page(IMAGE, new_page())
    # ids count all detections, the filtered stamp leaves a gap
    assert summary(layout) == [('r000', 'image'), ('r002', 'page_number')]


def test_line_categories_get_one_line_across_the_box(fake_yolo):
    layout = make_extractor('LINE_CATEGORIES = ["page_number"]\n').process_page(IMAGE, new_page())
    assert [len(region.lines) for region in layout.regions] == [0, 0, 1]
    line = layout.regions[2].lines[0]
    assert line.id == 'r002-l000' and line.category == 'page_number'
    np.testing.assert_array_equal(line.baseline, [[380, 915.5], [420, 915.5]])  # through the middle of the box
    np.testing.assert_array_equal(line.heights, [15, 15])  # 15.5 up and down, floored
    np.testing.assert_array_equal(line.polygon, layout.regions[2].polygon)


# --- existing regions ---------------------------------------------------------------------------------------------

def test_text_regions_are_kept_and_new_ids_continue_after_them(fake_yolo):
    text = box_region('r004', 50, 450, 750, 850, category='text')
    text.lines = [TextLine(id='r004-l001', baseline=np.array([[60.0, 500.0], [740.0, 500.0]]), heights=[20.0, 5.0])]
    uncategorized = box_region('r001', 50, 50, 90, 90)  # no category counts as text
    layout = make_extractor().process_page(IMAGE, new_page([text, uncategorized]))
    assert summary(layout) == [('r004', 'text'), ('r001', None), ('r005', 'image'), ('r006', 'stamp'),
                               ('r007', 'page_number')]
    assert layout.regions[0] is text and len(text.lines) == 1


def test_earlier_non_text_regions_are_replaced(fake_yolo):
    """Running the stage again (or after another non-text detector) replaces all non-text regions."""
    layout = make_extractor().process_page(IMAGE, new_page([box_region('r000', 10, 10, 20, 20, category='logo')]))
    assert summary(layout) == [('r000', 'image'), ('r001', 'stamp'), ('r002', 'page_number')]


@pytest.mark.xfail(reason='LayoutExtractorYolo.get_start_id expects every text region id to start with r<number> and '
                          'crashes on others, e.g. from PAGE XML written by other tools', strict=True)
def test_text_region_ids_of_other_tools(fake_yolo):
    layout = new_page([box_region('TextRegion_1', 50, 450, 750, 850, category='text')])
    layout = make_extractor().process_page(IMAGE, layout)
    assert [region.category for region in layout.regions] == ['text', 'image', 'stamp', 'page_number']


# --- config and the network call ----------------------------------------------------------------------------------

def test_model_path_is_relative_to_the_config_and_runs_on_cpu(fake_yolo, tmp_path):
    make_extractor(model_path='layout/yolo.pt', config_path=str(tmp_path), device=torch.device('cuda'))
    net, = fake_yolo.instances
    assert net.model_path == os.path.join(str(tmp_path), 'layout', 'yolo.pt')
    assert net.device == torch.device('cpu')  # USE_CPU overrides the device given


@pytest.mark.parametrize('extra, imgsz', [
    ('', None),
    ('IMAGE_SIZE = 640\n', 640),
    ('IMAGE_SIZE = [640, 480]\n', (640, 480)),
])
def test_image_size(fake_yolo, extra, imgsz):
    make_extractor(extra).process_page(IMAGE, new_page())
    call, = fake_yolo.instances[0].calls
    assert call.get('imgsz') == imgsz and call['verbose'] is False and call['image'] is IMAGE


def test_invalid_image_size(fake_yolo):
    with pytest.raises(ValueError, match='Invalid image size'):
        make_extractor('IMAGE_SIZE = [640, 480, 3]\n')


def test_without_ultralytics(monkeypatch):
    monkeypatch.setitem(sys.modules, 'ultralytics', None)  # import ultralytics raises ImportError
    with pytest.raises(ImportError, match='pip install ultralytics'):
        make_extractor()


def test_is_built_by_the_factory(fake_yolo):
    assert isinstance(make_extractor(), LayoutExtractorYolo)


# --- the real ultralytics -----------------------------------------------------------------------------------------

def test_real_ultralytics_matches_the_fake(monkeypatch, tmp_path):
    """The fake's interface is what the installed ultralytics really does: an untrained YOLOv8n built from its yaml
    (no download, random weights, so arbitrary boxes) goes through LAYOUT_YOLO. Skipped without ultralytics."""
    monkeypatch.setenv('YOLO_CONFIG_DIR', str(tmp_path))  # ultralytics writes its settings there
    pytest.importorskip('ultralytics')
    torch.manual_seed(0)
    extractor = make_extractor('IMAGE_SIZE = 64\nLINE_CATEGORIES = ["0"]\n', model_path='yolov8n.yaml', threshold=0.0)
    layout = extractor.process_page(IMAGE, new_page())
    assert layout.regions
    for region in layout.regions:
        assert region.polygon.shape == (5, 2) and 0.0 <= region.detection_confidence <= 1.0
        assert region.category.isdigit() and len(region.lines) == (region.category == '0')
