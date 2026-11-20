import json

import lxml.etree as ET
import pytest

from pero_ocr.core.layout import PageLayout, PAGEVersion
from test.io.reference_layout import assert_layouts_equal, build_reference_layout, build_edge_layout, build_empty_layout

PAGE_2019 = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'


@pytest.fixture
def layout():
    return build_reference_layout()


def test_to_pagexml_file_equals_string(layout, tmp_path):
    path = tmp_path / 'page.xml'
    layout.to_pagexml(str(path))
    root = ET.parse(str(path)).getroot()
    assert ET.QName(root).namespace == PAGE_2019
    assert path.read_text(encoding='utf-8').startswith("<?xml version='1.0' encoding='utf-8'?>")


def test_to_pagexml_tree_matches_string(layout):
    tree_xml = ET.tostring(layout.to_pagexml_tree(), encoding='unicode')
    assert 'Příliš žluťoučký kůň' in tree_xml
    assert ET.QName(layout.to_pagexml_tree()).localname == 'PcGts'


def test_validate_id_prefixes_ids(layout):
    root = ET.fromstring(layout.to_pagexml_string(validate_id=True).encode())
    ids = [el.get('id') for el in root.iter('{*}TextRegion', '{*}TextLine')]
    assert ids and all(i.startswith('id_') for i in ids)


def test_coords_are_non_negative_integers(layout):
    """Issue #84: PAGE requires non-negative integer coordinates."""
    root = ET.fromstring(layout.to_pagexml_string().encode())
    for el in root.iter('{*}Coords', '{*}Baseline'):
        for point in el.get('points').split():
            x, y = point.split(',')
            assert x.isdigit() and y.isdigit(), point


def test_regions_follow_reading_order(layout):
    root = ET.fromstring(layout.to_pagexml_string().encode())
    assert [r.get('id') for r in root.iter('{*}TextRegion')] == ['r001', 'r002', 'r003']
    refs = {r.get('regionRef'): int(r.get('index')) for r in root.iter('{*}RegionRefIndexed')}
    assert refs == {'r001': 0, 'r002': 1, 'r003': 2}


def test_round_trip(layout):
    loaded = PageLayout()
    loaded.from_pagexml_string(layout.to_pagexml_string())
    assert_layouts_equal(loaded, layout)


def test_round_trip_page_2013(layout):
    loaded = PageLayout()
    loaded.from_pagexml_string(layout.to_pagexml_string(version=PAGEVersion.PAGE_2013_07_15))
    assert [r.id for r in loaded.regions] == ['r001', 'r002', 'r003']
    assert sum(len(r.lines) for r in loaded.regions) == 6


def test_custom_attribute_is_json(layout):
    root = ET.fromstring(layout.to_pagexml_string().encode())
    for el in root.iter('{*}TextRegion', '{*}TextLine'):
        if el.get('custom') is not None:
            json.loads(el.get('custom'))


# --- edge cases --------------------------------------------------------------------------------------------------


def test_edge_page_round_trip():
    edge = build_edge_layout()
    loaded = PageLayout()
    loaded.from_pagexml_string(edge.to_pagexml_string())
    assert loaded.id == 'scan 01#(a).tif'
    r1, r2, r3 = loaded.regions
    assert r1.transcription == 'region level text'
    assert [line.transcription for line in r1.lines] == ['', 'text without logits', edge.regions[0].lines[2].transcription,
                                                         'abc def']
    assert r1.lines[0].transcription_confidence is None
    assert (r2.category, r2.detection_confidence) == ('barcode', 0.512)
    assert r2.lines[0].category == 'barcode' and r2.lines[0].transcription is None
    assert (r3.category, r3.lines[0].transcription) == ('title', 'Big Title')


def test_empty_page_round_trip():
    loaded = PageLayout()
    loaded.from_pagexml_string(build_empty_layout().to_pagexml_string())
    assert (loaded.id, loaded.page_size, loaded.regions) == ('empty_page.jpg', (400, 300), [])
