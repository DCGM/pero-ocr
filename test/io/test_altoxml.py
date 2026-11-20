import lxml.etree as ET
import numpy as np
import pytest

from pero_ocr.core.layout import ALTOVersion, PageLayout, RegionLayout, TextLine
from test.io.golden import check_golden_xml
from test.io.reference_layout import build_reference_layout, ctc_logits, CHARACTERS

ALTO_V2 = 'http://www.loc.gov/standards/alto/ns-v2#'
ALTO_V4 = 'http://www.loc.gov/standards/alto/ns-v4#'


@pytest.fixture
def layout():
    return build_reference_layout()


def strings(xml: str):
    root = ET.fromstring(xml.encode())
    return [s.get('CONTENT') for s in root.iter('{*}String')]


def test_golden_alto_v2(layout, update_golden):
    check_golden_xml('reference_page.alto_v2.xml', layout.to_altoxml_string(), update_golden)


def test_golden_alto_v4(layout, update_golden):
    check_golden_xml('reference_page.alto_v4.xml', layout.to_altoxml_string(version=ALTOVersion.ALTO_v4_4),
                     update_golden)


@pytest.mark.parametrize('version, namespace', [(ALTOVersion.ALTO_v2_x, ALTO_V2), (ALTOVersion.ALTO_v4_4, ALTO_V4)])
def test_namespace(layout, version, namespace):
    root = ET.fromstring(layout.to_altoxml_string(version=version).encode())
    assert ET.QName(root).namespace == namespace


def test_words_match_transcriptions(layout):
    expected = []
    for region in layout.regions:
        for line in region.lines:
            if line.transcription and line.transcription.strip():
                expected += line.transcription.split() if line.category in (None, 'text') else [line.transcription]
    assert strings(layout.to_altoxml_string()) == expected


def test_hyphenated_word_gets_subs_content(layout):
    root = ET.fromstring(layout.to_altoxml_string().encode())
    subs = {s.get('CONTENT'): (s.get('SUBS_TYPE'), s.get('SUBS_CONTENT')) for s in root.iter('{*}String')
            if s.get('SUBS_TYPE')}
    assert subs == {'hyphen-': ('HypPart1', 'hyphenated'), 'ated': ('HypPart2', 'hyphenated')}


def test_no_word_splitters(layout):
    xml = layout.to_altoxml_string(word_splitters=None)
    assert 'SUBS_TYPE' not in xml


def test_min_line_confidence_drops_lines(layout):
    all_lines = len(ET.fromstring(layout.to_altoxml_string().encode()).findall('.//{*}TextLine'))
    none_left = ET.fromstring(layout.to_altoxml_string(min_line_confidence=1.01).encode()).findall('.//{*}TextLine')
    assert all_lines == 5
    assert none_left == []


def test_word_boxes_inside_line_boxes(layout):
    root = ET.fromstring(layout.to_altoxml_string().encode())
    for line in root.iter('{*}TextLine'):
        lx, ly = int(line.get('HPOS')), int(line.get('VPOS'))
        lw, lh = int(line.get('WIDTH')), int(line.get('HEIGHT'))
        for s in line.iter('{*}String'):
            x, w = int(s.get('HPOS')), int(s.get('WIDTH'))
            assert lx - 20 <= x and x + w <= lx + lw + 20, (line.get('ID'), s.get('CONTENT'))
            assert 0 < w


def test_round_trip(layout):
    loaded = PageLayout()
    loaded.from_altoxml_string(layout.to_altoxml_string())
    assert loaded.page_size == layout.page_size
    assert [r.id for r in loaded.regions] == ['block_r001', 'block_r002', 'block_r003']
    texts = [line.transcription for r in loaded.regions for line in r.lines]
    assert texts == ['hello world, hyphen-', 'ated text.', 'Příliš žluťoučký kůň', 'Header line', 'margin note']


def test_round_trip_v4_keeps_baseline_points(layout):
    loaded = PageLayout()
    loaded.from_altoxml_string(layout.to_altoxml_string(version=ALTOVersion.ALTO_v4_4))
    line = loaded.regions[0].lines[0]
    np.testing.assert_array_equal(line.baseline, np.round(layout.regions[0].lines[0].baseline))


def test_line_without_logits_falls_back_to_even_word_spacing():
    page = PageLayout(id='p', page_size=(100, 300))
    region = RegionLayout('r1', np.array([[0, 0], [300, 0], [300, 100], [0, 100]]))
    line = TextLine(id='l1', baseline=np.array([[10, 50], [290, 50]]), heights=[10, 5],
                    polygon=np.array([[10, 40], [290, 40], [290, 55], [10, 55]]), transcription='two words')
    region.lines = [line]
    page.regions = [region]
    assert strings(page.to_altoxml_string()) == ['two', 'words']


@pytest.mark.xfail(reason='issue #91: a TAB surrounded by spaces breaks the word split in ALTO export', strict=True)
def test_tab_surrounded_by_spaces():
    text = 'ab \t cd'
    page = PageLayout(id='p', page_size=(100, 300))
    region = RegionLayout('r1', np.array([[0, 0], [300, 0], [300, 100], [0, 100]]))
    line = TextLine(id='l1', baseline=np.array([[10, 50], [290, 50]]), heights=[10, 5],
                    polygon=np.array([[10, 40], [290, 40], [290, 55], [10, 55]]), transcription=text,
                    logits=ctc_logits(text), characters=CHARACTERS)
    line.logit_coords = [0, line.logits.shape[0]]
    region.lines = [line]
    page.regions = [region]
    assert strings(page.to_altoxml_string()) == ['ab', 'cd']


@pytest.mark.xfail(reason='ALTO export does not clamp coordinates to the page like the PAGE export does (cf. #84)',
                   strict=True)
def test_coords_are_non_negative(layout):
    root = ET.fromstring(layout.to_altoxml_string().encode())
    for el in root.iter():
        for attr in ('HPOS', 'VPOS', 'WIDTH', 'HEIGHT'):
            if el.get(attr) is not None:
                assert int(el.get(attr)) >= 0, (ET.QName(el).localname, el.get('ID'), attr, el.get(attr))
