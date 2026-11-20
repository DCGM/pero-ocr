"""The geometric reading-order sorters as black boxes (config section and PageLayout in, PageLayout out):
REGION_SORTER_SMART (SmartRegionSorter, used by most production engines) and REGION_SORTER_NAIVE (NaiveRegionSorter).
The transformer sorter has its own file, test_region_sorter_transformer.py."""
import numpy as np
import pytest

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.layout_engines.naive_sorter import NaiveRegionSorter
from pero_ocr.layout_engines.smart_sorter import SmartRegionSorter
from test.configs import section

PAGE_SIZE = (1000, 800)  # height, width
IMAGE = np.zeros(PAGE_SIZE + (3,), dtype=np.uint8)
NAIVE_BUG = pytest.mark.xfail(reason='NaiveRegionSorter.sort_regions clusters columns by y_min instead of x_min',
                              strict=True)
SORTERS = ['REGION_SORTER_SMART', pytest.param('REGION_SORTER_NAIVE', marks=NAIVE_BUG)]


def make(method, **options):
    sorter = {'REGION_SORTER_SMART': SmartRegionSorter, 'REGION_SORTER_NAIVE': NaiveRegionSorter}[method]
    return sorter(section(**options))


def region(region_id, l, t, r, b, category=None, angle=0.0):
    """A region with one line per 40 px, baselines tilted by ``angle`` degrees around the page origin."""
    corners = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float64)
    result = RegionLayout(region_id, corners, category=category)
    for i, y in enumerate(range(t + 30, b, 40)):
        baseline = np.array([[l + 5.0, y], [r - 5.0, y]])
        result.lines.append(TextLine(id=f'{region_id}-l{i}', baseline=baseline, heights=[20.0, 5.0],
                                     polygon=np.array([[l + 5, y - 20], [r - 5, y - 20], [r - 5, y + 5],
                                                       [l + 5, y + 5]], dtype=np.float64)))
    if angle:
        a = np.deg2rad(angle)
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        result.polygon = result.polygon @ rotation.T
        for line in result.lines:
            line.baseline, line.polygon = line.baseline @ rotation.T, line.polygon @ rotation.T
    return result


def page(*regions):
    layout = PageLayout(id='p', page_size=PAGE_SIZE)
    layout.regions = list(regions)
    return layout


def ids(layout):
    return [r.id for r in layout.regions]


# two columns under a full-width title, given in a scrambled order
TITLE = (40, 40, 760, 120)
LEFT_TOP, LEFT_BOTTOM = (40, 160, 380, 500), (40, 540, 380, 940)
RIGHT_TOP, RIGHT_BOTTOM = (420, 160, 760, 700), (420, 740, 760, 940)  # the gaps of the columns do not line up


def scrambled(**kwargs):
    return page(region('right_bottom', *RIGHT_BOTTOM, **kwargs), region('left_bottom', *LEFT_BOTTOM, **kwargs),
                region('title', *TITLE, **kwargs), region('right_top', *RIGHT_TOP, **kwargs),
                region('left_top', *LEFT_TOP, **kwargs))


IN_ORDER = ['title', 'left_top', 'left_bottom', 'right_top', 'right_bottom']


@pytest.mark.parametrize('method', SORTERS)
def test_title_then_columns_left_to_right(method):
    assert ids(make(method).process_page(IMAGE, scrambled())) == IN_ORDER


@pytest.mark.parametrize('method', SORTERS)
def test_side_by_side_columns(method):
    columns = page(region('right', 500, 50, 700, 400), region('left', 20, 50, 220, 400))
    layout = make(method).process_page(IMAGE, columns)
    assert ids(layout) == ['left', 'right']


@pytest.mark.parametrize('method, n', [
    ('REGION_SORTER_SMART', 0), ('REGION_SORTER_SMART', 1), ('REGION_SORTER_NAIVE', 1),
    pytest.param('REGION_SORTER_NAIVE', 0, marks=pytest.mark.xfail(
        reason='NaiveRegionSorter crashes on a page without regions: DBSCAN gets an empty array', strict=True)),
])
def test_fewer_than_two_regions(method, n):
    regions = [region('only', *TITLE)][:n]
    assert ids(make(method).process_page(IMAGE, page(*regions))) == ['only'][:n]


@pytest.mark.parametrize('method', ['REGION_SORTER_SMART', 'REGION_SORTER_NAIVE'])
def test_regions_keep_their_content(method):
    layout = scrambled()
    before = {r.id: (r, r.polygon.copy(), [line.baseline.copy() for line in r.lines]) for r in layout.regions}
    layout = make(method).process_page(IMAGE, layout)
    for r in layout.regions:
        original, polygon, baselines = before[r.id]
        assert r is original
        np.testing.assert_allclose(r.polygon, polygon, atol=1e-6)
        for line, baseline in zip(r.lines, baselines):
            np.testing.assert_allclose(line.baseline, baseline, atol=1e-6)


@pytest.mark.parametrize('method', SORTERS)
def test_categories_sort_only_those_regions(method):
    """With CATEGORIES, only regions of those categories are sorted; the others are put first, in their order."""
    layout = scrambled(category='text')
    layout.regions[1:1] = [region('stamp', 600, 950, 700, 990, category='stamp'),
                           region('image', 600, 0, 700, 30, category='image')]
    layout = make(method, CATEGORIES=['text']).process_page(IMAGE, layout)
    assert ids(layout) == ['stamp', 'image'] + IN_ORDER


def test_smart_sorter_on_a_tilted_page():
    """The smart sorter estimates the page rotation from the lines of the biggest region and sorts in that frame."""
    layout = make('REGION_SORTER_SMART').process_page(IMAGE, scrambled(angle=4.0))
    assert ids(layout) == IN_ORDER


def test_naive_sorter_column_gap_from_image_width_denominator():
    """Columns closer than page width / ImageWidthDenominator are one cluster, read top to bottom."""
    near = page(region('right', 230, 50, 430, 400), region('left', 20, 420, 220, 800))
    assert ids(make('REGION_SORTER_NAIVE', ImageWidthDenominator=2).process_page(IMAGE, near)) == ['right', 'left']


@pytest.mark.parametrize('parameter, expected', [
    (0.1, ['left_top', 'right_top', 'left_bottom', 'right_bottom']),  # the default: a 5 % overlap does not count
    pytest.param(0.01, ['left_top', 'left_bottom', 'right_top', 'right_bottom'], marks=pytest.mark.xfail(
        reason='SmartRegionSorter reads FakeIntersectionParameter, but CoupledRegions.divide_and_order calls '
               'intersect() without it, so the default 0.1 always applies', strict=True)),
])
def test_smart_sorter_fake_intersection_parameter(parameter, expected):
    """Regions that overlap vertically by less than FakeIntersectionParameter (of both heights) count as one above the
    other. Here the right column's upper region reaches 20 px (5 %) below the top of the left column's lower one."""
    layout = page(region('right_bottom', 420, 600, 760, 940), region('left_bottom', 40, 540, 380, 940),
                  region('right_top', 420, 160, 760, 560), region('left_top', 40, 160, 380, 500))
    assert ids(make('REGION_SORTER_SMART', FakeIntersectionParameter=parameter).process_page(IMAGE, layout)) == expected
