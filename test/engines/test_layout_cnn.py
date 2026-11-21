"""The LAYOUT_CNN stage (LayoutExtractor in pero_ocr.document_ocr.page_parser) as a black box: a config section and a
page in, a PageLayout out. LayoutEngine underneath, which turns ParseNet maps into baselines, is tested in
test_layout_engine.py.

The tiny ParseNet (test/tiny_models.py) puts a baseline under every bright block and gives constant heights, so a
synthetic page with bright bars (test/synthetic.py) has one text line per bar."""
import numpy as np
import pytest
import torch

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr.page_parser import LayoutExtractor
from test import tiny_models
from test.configs import section
from test.synthetic import page_image

PAGE = page_image(3)  # bars at y 100-140, 200-240, 300-340, x 100-700
INK_BOTTOMS = [139, 239, 339]
OFF = dict(DETECT_REGIONS='no', DETECT_LINES='no', DETECT_STRAIGHT_LINES_IN_REGIONS='no', MERGE_LINES='no',
           ADJUST_HEIGHTS='no', ADJUST_BASELINES='no', MULTI_ORIENTATION='no')


def make(parsenet_path, device=torch.device('cpu'), config_path='', **options):
    config = dict(MODEL_PATH=parsenet_path, USE_CPU='yes', DOWNSAMPLE=4,
                  DETECTION_THRESHOLD=0.2, MAX_MEGAPIXELS=5, **OFF)
    config.update(options)
    return LayoutExtractor(section(**config), device, config_path=config_path)


def new_page(regions=(), image=PAGE):
    layout = PageLayout(id='p', page_size=image.shape[:2])
    layout.regions = list(regions)
    return layout


def whole_page_region(region_id='r000', category='text'):
    return RegionLayout(region_id, np.array([[0, 0], [800, 0], [800, 600], [0, 600]]), category=category)


def line_at(line_id, y, heights=(5.0, 5.0), category='text'):
    return TextLine(id=line_id, baseline=np.array([[100.0, y], [700.0, y]]), heights=list(heights), category=category,
                    polygon=np.array([[100, y - heights[0]], [700, y - heights[0]], [700, y + heights[1]],
                                      [100, y + heights[1]]]))


def mean_ys(lines):
    return [float(np.mean(line.baseline[:, 1])) for line in lines]


# --- what is detected ---------------------------------------------------------------------------------------------

def test_detects_regions_and_lines(parsenet_path):
    layout = make(parsenet_path, DETECT_REGIONS='yes', DETECT_LINES='yes').process_page(PAGE, new_page())
    region, = layout.regions
    assert region.id == 'r000' and region.category == 'text'
    assert [line.id for line in region.lines] == ['r000-l001', 'r000-l002', 'r000-l003']
    assert all(line.category == 'text' and line.polygon is not None for line in region.lines)
    assert all(abs(y - bottom) <= 8 for y, bottom in zip(mean_ys(region.lines), INK_BOTTOMS))


def test_regions_without_lines(parsenet_path):
    layout = make(parsenet_path, DETECT_REGIONS='yes').process_page(PAGE, new_page())
    assert [(region.id, region.lines) for region in layout.regions] == [('r000', [])]


def test_lines_into_existing_regions(parsenet_path):
    """DETECT_REGIONS = no: the regions stay, their old lines are replaced by the detected ones."""
    region = whole_page_region('mine')
    region.lines = [line_at('old', 500.0)]
    layout = make(parsenet_path, DETECT_LINES='yes').process_page(PAGE, new_page([region]))
    assert layout.regions == [region]
    assert [line.id for line in region.lines] == ['mine-l001', 'mine-l002', 'mine-l003']


def test_nothing_detected_when_both_are_off(parsenet_path):
    region = whole_page_region()
    region.lines = [line_at('r000-l000', 500.0)]
    layout = make(parsenet_path).process_page(PAGE, new_page([region]))
    assert layout.regions == [region] and [line.id for line in region.lines] == ['r000-l000']


def test_non_text_regions_are_kept(parsenet_path):
    image = RegionLayout('r009', np.array([[0, 500], [50, 500], [50, 550], [0, 550]]), category='image')
    layout = make(parsenet_path, DETECT_REGIONS='yes', DETECT_LINES='yes').process_page(PAGE, new_page([image]))
    assert [(region.id, region.category) for region in layout.regions] == [('r000', 'text'), ('r009', 'image')]
    assert layout.regions[1] is image


def test_detection_threshold(parsenet_path):
    """The tiny ParseNet's baseline map is 1 under the ink: a threshold above it finds nothing."""
    layout = make(parsenet_path, DETECT_REGIONS='yes', DETECT_LINES='yes',
                  DETECTION_THRESHOLD=1.5).process_page(PAGE, new_page())
    assert list(layout.lines_iterator()) == []


def test_multi_orientation_finds_vertical_lines(parsenet_path):
    """On a page turned by 90 degrees, the text lines are vertical. MULTI_ORIENTATION also runs the detection on the
    page rotated by 90 and 270 degrees and adds what it finds as regions r<n>_<rotation>, in page coordinates."""
    vertical = np.ascontiguousarray(np.rot90(PAGE))  # 800 x 600, bars at x 100-140, 200-240, 300-340

    def is_vertical(line):
        return np.ptp(line.baseline[:, 1]) > 10 * np.ptp(line.baseline[:, 0])

    detect = dict(DETECT_REGIONS='yes', DETECT_LINES='yes')
    single = make(parsenet_path, **detect).process_page(vertical, new_page(image=vertical))
    assert not any(is_vertical(line) for line in single.lines_iterator())

    multi = make(parsenet_path, DETECT_REGIONS='yes', DETECT_LINES='yes',
                 MULTI_ORIENTATION='yes').process_page(vertical, new_page(image=vertical))
    rotated = [region for region in multi.regions if region.id in ('r000_1', 'r000_3')]
    assert rotated  # the tiny ParseNet reads bars in both directions, so both rotations find them
    for region in rotated:
        assert len(region.lines) == 3 and all(is_vertical(line) for line in region.lines)
        for line in region.lines:
            assert 90 <= line.baseline[:, 0].mean() <= 350  # at the bars, in page coordinates
            assert line.baseline[:, 1].min() < 150 and line.baseline[:, 1].max() > 650  # along the whole bar


# --- adjusting lines ----------------------------------------------------------------------------------------------

SPLIT_PAGE = np.zeros((600, 800, 3), dtype=np.uint8)
SPLIT_PAGE[100:140, 100:380] = 255  # one line with a gap in the middle
SPLIT_PAGE[100:140, 420:700] = 255
SPLIT_PAGE[200:240, 100:700] = 255


def test_merge_lines_joins_the_parts_of_a_line(parsenet_path):
    def lines(merge):
        layout = make(parsenet_path, DETECT_LINES='yes', MERGE_LINES=merge).process_page(
            SPLIT_PAGE, new_page([whole_page_region()]))
        return sorted(layout.regions[0].lines, key=lambda line: line.baseline[:, 1].mean())

    assert len(lines('no')) == 3
    assert [round(y) for y in mean_ys(lines('yes'))] == [132, 232]


@pytest.mark.xfail(reason='MERGE_LINES truncates merged lines: the merged baseline is straight up to float rounding, '
                          'shapely splits its intersection with the region at every vertex, and '
                          'mask_textline_by_region keeps only the longest piece', strict=True)
def test_merged_line_spans_both_parts(parsenet_path):
    layout = make(parsenet_path, DETECT_LINES='yes', MERGE_LINES='yes').process_page(
        SPLIT_PAGE, new_page([whole_page_region()]))
    merged = min(layout.regions[0].lines, key=lambda line: line.baseline[:, 1].mean())
    assert merged.baseline[:, 0].min() < 110 and merged.baseline[:, 0].max() > 690


def test_adjust_heights_and_baselines(parsenet_path):
    """Lines from elsewhere (e.g. PAGE XML) get heights and baselines from the ParseNet maps. The heights are read
    along the given baseline: on the bottom row of the ink (map row 34 at downsample 4, y = 136) the full 40 px line
    height, one map row higher (y = 132, where detection with SMOOTH_LINE_PREDICTIONS puts the baseline) 4 px less."""
    region = whole_page_region()
    for y, ascender in [(136.0, 40.0), (132.0, 36.0)]:
        region.lines = [line_at('r000-l000', y)]
        line = region.lines[0]
        make(parsenet_path, ADJUST_HEIGHTS='yes').process_page(PAGE, new_page([region]))
        np.testing.assert_allclose(line.heights, [ascender, ascender * tiny_models.PARSENET_DESCENDER_RATIO])
        assert line.baseline[0, 1] == y and line.polygon[:, 1].min() == pytest.approx(y - ascender, abs=1)

    region.lines = [line_at('r000-l000', 130.0, heights=(40.0, 12.0))]
    line = region.lines[0]
    make(parsenet_path, ADJUST_BASELINES='yes').process_page(PAGE, new_page([region]))
    assert abs(line.baseline[:, 1].mean() - INK_BOTTOMS[0]) < abs(130.0 - INK_BOTTOMS[0])


def test_categories_limit_the_adjustments(parsenet_path):
    region = whole_page_region()
    region.lines = [line_at('r000-l000', 130.0), line_at('r000-l001', 230.0, category='title')]
    make(parsenet_path, ADJUST_HEIGHTS='yes', ADJUST_BASELINES='yes', CATEGORIES=['text']).process_page(
        PAGE, new_page([region]))
    text, title = region.lines
    assert list(text.heights) != [5.0, 5.0] and text.baseline[0, 1] != 130.0
    assert list(title.heights) == [5.0, 5.0] and title.baseline[0, 1] == 230.0


def test_straight_lines_in_regions(parsenet_path):
    """DETECT_STRAIGHT_LINES_IN_REGIONS replaces the lines of every region by straight lines across the region."""
    region = whole_page_region()
    region.lines = [line_at('old', 500.0)]
    make(parsenet_path, DETECT_STRAIGHT_LINES_IN_REGIONS='yes').process_page(PAGE, new_page([region]))
    assert len(region.lines) == 3
    assert all(abs(y - bottom) <= 8 for y, bottom in zip(mean_ys(region.lines), INK_BOTTOMS))
    assert all(np.ptp(line.baseline[:, 1]) < 1 for line in region.lines)


# --- config -------------------------------------------------------------------------------------------------------

def test_model_path_relative_to_the_config(parsenet_path):
    import os
    extractor = make(os.path.basename(parsenet_path), config_path=os.path.dirname(parsenet_path),
                     DETECT_REGIONS='yes', DETECT_LINES='yes')
    assert len(list(extractor.process_page(PAGE, new_page()).lines_iterator())) == 3


def test_use_cpu_overrides_the_device(parsenet_path):
    assert make(parsenet_path, device=torch.device('cuda')).device == torch.device('cpu')
