"""PAGE XML import of inputs that pero-ocr does not produce itself (other tools, older versions)."""
import numpy as np
import pytest

from pero_ocr.core.layout import PageLayout

HEADER_2019 = ('<?xml version="1.0" encoding="UTF-8"?>'
               '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">')
HEADER_2013 = ('<?xml version="1.0" encoding="UTF-8"?>'
               '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">')


def page(body: str, header: str = HEADER_2019) -> PageLayout:
    layout = PageLayout()
    layout.from_pagexml_string(f'{header}<Page imageFilename="img.jpg" imageWidth="500" imageHeight="400">'
                               f'{body}</Page></PcGts>')
    return layout


LINE = ('<TextLine id="{id}" {attrs}><Coords points="10,10 200,10 200,30 10,30"/>'
        '<Baseline points="10,25 200,25"/>{text}</TextLine>')
REGION = '<TextRegion id="{id}"{attrs}>{coords}{lines}</TextRegion>'
COORDS = '<Coords points="0,0 300,0 300,100 0,100"/>'


def region(id='r1', lines='', coords=COORDS, attrs=''):
    return REGION.format(id=id, attrs=attrs, coords=coords, lines=lines)


def line(id='l1', attrs='', text='<TextEquiv conf="0.75"><Unicode>abc</Unicode></TextEquiv>'):
    return LINE.format(id=id, attrs=attrs, text=text)


def test_basic_import():
    layout = page(region(lines=line()))
    assert layout.id == 'img.jpg'
    assert layout.page_size == (400, 500)
    text_line = layout.regions[0].lines[0]
    assert text_line.transcription == 'abc'
    assert text_line.transcription_confidence == pytest.approx(0.75)
    np.testing.assert_array_equal(text_line.baseline, [[10, 25], [200, 25]])


def test_points_as_point_elements_page_2013():
    coords = '<Coords><Point x="0" y="0"/><Point x="300" y="0"/><Point x="300" y="100"/><Point x="0" y="100"/></Coords>'
    layout = page(region(coords=coords, lines=line()), header=HEADER_2013)
    np.testing.assert_array_equal(layout.regions[0].polygon, [[0, 0], [300, 0], [300, 100], [0, 100]])


def test_line_without_baseline_is_skipped():
    no_baseline = '<TextLine id="l2"><Coords points="10,10 200,10 200,30 10,30"/></TextLine>'
    layout = page(region(lines=line() + no_baseline))
    assert [text_line.id for text_line in layout.regions[0].lines] == ['l1']


def test_line_without_polygon_gets_one_from_baseline_and_heights():
    no_polygon = '<TextLine id="l1" custom="{&quot;heights&quot;: [12, 4]}"><Baseline points="10,25 200,25"/></TextLine>'
    text_line = page(region(lines=no_polygon)).regions[0].lines[0]
    assert text_line.polygon is not None
    assert text_line.polygon[:, 1].min() == pytest.approx(13, abs=1)
    assert text_line.polygon[:, 1].max() == pytest.approx(29, abs=1)


@pytest.mark.xfail(reason='#96: the polygon estimate (baseline_to_textline) crashes on a single-point baseline, so the '
                          'whole page fails to load', strict=True)
def test_line_without_polygon_and_with_a_single_point_baseline():
    one_point = '<TextLine id="l2" custom="{&quot;heights&quot;: [12, 4]}"><Baseline points="10,25"/></TextLine>'
    lines = page(region(lines=line() + one_point)).regions[0].lines
    assert [text_line.id for text_line in lines] == ['l1', 'l2']
    assert lines[1].polygon[:, 1].min() == pytest.approx(13, abs=1)
    assert lines[1].polygon[:, 1].max() == pytest.approx(29, abs=1)


def test_region_without_coords_is_estimated_from_lines():
    """Issue #79."""
    layout = page(region(coords='', lines=line()))
    assert len(layout.regions) == 1
    assert layout.regions[0].polygon is not None


def test_region_without_coords_and_lines_is_skipped():
    layout = page(region(id='empty', coords='') + region(id='r2', lines=line()))
    assert [r.id for r in layout.regions] == ['r2']


def test_empty_unicode_is_empty_string():
    text_line = page(region(lines=line(text='<TextEquiv><Unicode/></TextEquiv>'))).regions[0].lines[0]
    assert text_line.transcription == ''
    assert text_line.transcription_confidence is None


def test_line_without_text():
    text_line = page(region(lines=line(text=''))).regions[0].lines[0]
    assert text_line.transcription is None


def test_missing_index_falls_back_to_position():
    layout = page(region(lines=line('a') + line('b') + line('c', attrs='index="7"')))
    assert [text_line.index for text_line in layout.regions[0].lines] == [0, 1, 7]


@pytest.mark.parametrize('custom, heights', [
    ('{&quot;heights&quot;: [12.5, 4.0]}', [12.5, 4.0]),
    ('heights_v2:[19.0,6.0]', [19.0, 6.0]),
    ('heights {top:5; bottom:7; total:9;}', [7.0, 4.0]),
])
def test_custom_heights_formats(custom, heights):
    text_line = page(region(lines=line(attrs=f'custom="{custom}"'))).regions[0].lines[0]
    np.testing.assert_allclose(text_line.heights, heights)


def test_custom_category():
    attrs = 'custom="{&quot;category&quot;: &quot;title&quot;}"'
    layout = page(region(lines=line(attrs=attrs), attrs=' ' + attrs))
    assert layout.regions[0].category == 'title'
    assert layout.regions[0].lines[0].category == 'title'


def test_invalid_region_custom_is_ignored():
    layout = page(region(lines=line(), attrs=' custom="readingOrder {index:0;}"'))
    assert layout.regions[0].category is None


@pytest.mark.xfail(reason='sort_regions_by_reading_order looks up RegionLayout objects in a dict keyed by region id, '
                          'so regions are never reordered', strict=True)
def test_reading_order_sorts_regions():
    order = ('<ReadingOrder><OrderedGroup id="ro"><RegionRefIndexed regionRef="r2" index="0"/>'
             '<RegionRefIndexed regionRef="r1" index="1"/></OrderedGroup></ReadingOrder>')
    layout = PageLayout()
    layout.from_pagexml_string(f'{HEADER_2019}<Page imageFilename="img.jpg" imageWidth="500" imageHeight="400">'
                               f'{order}{region("r1", line("a"))}{region("r2", line("b"))}</Page></PcGts>')
    assert layout.reading_order == {'r2': 0, 'r1': 1}
    layout.sort_regions_by_reading_order()
    assert [r.id for r in layout.regions] == ['r2', 'r1']


def test_page_layout_from_file(tmp_path):
    path = tmp_path / 'page.xml'
    path.write_text(f'{HEADER_2019}<Page imageFilename="img.jpg" imageWidth="500" imageHeight="400">'
                    f'{region(lines=line())}</Page></PcGts>', encoding='utf-8')
    layout = PageLayout(file=str(path))
    assert layout.regions[0].lines[0].transcription == 'abc'
