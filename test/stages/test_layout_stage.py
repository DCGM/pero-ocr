"""The [LAYOUT_PARSER] stage (layout_parser_factory in pero_ocr.document_ocr.page_parser): a config section in, the
layout engine its METHOD names out, which turns a page image and a PageLayout into a PageLayout.

The stage has no logic of its own yet: every method reads its own options, devices and model paths, and decides itself
what happens to the regions already on the page. What each method does is tested with its engine in test/engines; here
only which engine a section builds, and that every engine keeps the contract the page parser relies on.

Not covered: REGION_SIMPLE_THRESHOLD, LINES_SIMPLE_THRESHOLD and LINE_FILTER, which may be dropped (#101, #102)."""
import numpy as np
import pytest

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr import page_parser as pp
from pero_ocr.layout_engines.naive_sorter import NaiveRegionSorter
from pero_ocr.layout_engines.smart_sorter import SmartRegionSorter
from pero_ocr.layout_engines.transformer_sorter import TransformerRegionSorter
from test.configs import section
from test.synthetic import page_image

PAGE = page_image(3)
CNN_OFF = dict(DETECT_STRAIGHT_LINES_IN_REGIONS='no', MERGE_LINES='no', ADJUST_HEIGHTS='no', ADJUST_BASELINES='no',
               MULTI_ORIENTATION='no')

METHODS = {
    'REGION_WHOLE_PAGE': pp.WholePageRegion,
    'LAYOUT_CNN': pp.LayoutExtractor,
    'LAYOUT_YOLO': pp.LayoutExtractorYolo,
    'LINE_POSTPROCESSING': pp.LinePostprocessor,
    'LAYOUT_POSTPROCESSING': pp.LayoutPostprocessor,
    'REGION_SORTER_NAIVE': NaiveRegionSorter,
    'REGION_SORTER_SMART': SmartRegionSorter,
    'REGION_SORTER_TRANSFORMER': TransformerRegionSorter,
}


def options(method, parsenet_path, region_sorter_path):
    return {
        'LAYOUT_CNN': dict(MODEL_PATH=parsenet_path, USE_CPU='yes', DOWNSAMPLE=4, DETECTION_THRESHOLD=0.2,
                           MAX_MEGAPIXELS=5, DETECT_REGIONS='yes', DETECT_LINES='yes', **CNN_OFF),
        'LAYOUT_YOLO': dict(MODEL_PATH='yolo.pt', DETECTION_THRESHOLD=0.3, USE_CPU='yes'),
        'LINE_POSTPROCESSING': dict(STRETCH_LINES=0, RESAMPLE_LINES='no', HEIGHTS_FROM_REGIONS='no'),
        'LAYOUT_POSTPROCESSING': dict(RETRACE_REGIONS='no'),
        'REGION_SORTER_TRANSFORMER': dict(MODEL_PATH=region_sorter_path, MAX_BBOX_COUNT=32, IMAGE_WIDTH=64,
                                          IMAGE_HEIGHT=64, USE_CPU='yes'),
    }.get(method, {})


def make(method, parsenet_path, region_sorter_path, cpu, config_path='', **overrides):
    config = {'METHOD': method, **options(method, parsenet_path, region_sorter_path), **overrides}
    return pp.layout_parser_factory(section(**config), cpu, config_path=config_path)


def page_with_a_region():
    layout = PageLayout(id='page', page_size=PAGE.shape[:2])
    region = RegionLayout('r000', np.array([[90, 90], [710, 90], [710, 370], [90, 370]]), category='text')
    for i, y in enumerate([140, 240, 340]):
        region.lines.append(TextLine(id=f'r000-l{i:03d}', baseline=np.array([[100.0, y], [700.0, y]]), heights=[40, 10],
                                     polygon=np.array([[100, y - 40], [700, y - 40], [700, y + 10], [100, y + 10]]),
                                     category='text'))
    layout.regions = [region]
    return layout


@pytest.mark.parametrize('method', METHODS)
def test_method_selects_the_engine(method, parsenet_path, region_sorter_path, fake_yolo, cpu):
    assert isinstance(make(method, parsenet_path, region_sorter_path, cpu), METHODS[method])


def test_unknown_method(cpu):
    with pytest.raises(ValueError, match='Unknown layout parser method'):
        pp.layout_parser_factory(section(METHOD='NOPE'), cpu)


@pytest.mark.parametrize('method', METHODS)
def test_every_engine_returns_a_page_layout_of_the_same_page(method, parsenet_path, region_sorter_path, fake_yolo,
                                                            cpu):
    """process_page(image, layout) returns the PageLayout the next section gets: the same page, with regions."""
    result = make(method, parsenet_path, region_sorter_path, cpu).process_page(PAGE, page_with_a_region())
    assert isinstance(result, PageLayout)
    assert (result.id, tuple(result.page_size)) == ('page', PAGE.shape[:2])
    assert all(isinstance(region, RegionLayout) for region in result.regions)


@pytest.mark.parametrize('method', ['LAYOUT_CNN', 'REGION_SORTER_TRANSFORMER'])
def test_model_path_relative_to_the_config_path(method, parsenet_path, region_sorter_path, cpu):
    """The page parser passes the directory of the config; the methods with a model resolve MODEL_PATH against it.
    (LAYOUT_YOLO: test/engines/test_layout_yolo.py, its model is loaded by ultralytics.)"""
    import os
    path = parsenet_path if method == 'LAYOUT_CNN' else region_sorter_path
    engine = make(method, parsenet_path, region_sorter_path, cpu, config_path=os.path.dirname(path),
                  MODEL_PATH=os.path.basename(path))
    assert isinstance(engine.process_page(PAGE, page_with_a_region()), PageLayout)
