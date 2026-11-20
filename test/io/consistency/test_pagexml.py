"""PAGE XML export must keep producing the committed golden IO files (test/io/consistency/data/page_xml)."""
import pytest

from pero_ocr.core.layout import PAGEVersion
from test.io.consistency.golden_io import check_golden_io_xml
from test.io.reference_layout import build_reference_layout, build_edge_layout, build_empty_layout


@pytest.fixture
def layout():
    return build_reference_layout()


def test_golden_io_page_2019(layout, update_golden_io):
    check_golden_io_xml('page_xml/reference_page.2019.xml', layout.to_pagexml_string(), update_golden_io)


def test_golden_io_page_2013(layout, update_golden_io):
    check_golden_io_xml('page_xml/reference_page.2013.xml',
                     layout.to_pagexml_string(version=PAGEVersion.PAGE_2013_07_15), update_golden_io)


def test_golden_io_page_2019_validate_id(layout, update_golden_io):
    check_golden_io_xml('page_xml/reference_page.2019.validate_id.xml', layout.to_pagexml_string(validate_id=True),
                     update_golden_io)


@pytest.mark.parametrize('version, suffix', [(PAGEVersion.PAGE_2019_07_15, '2019'),
                                             (PAGEVersion.PAGE_2013_07_15, '2013')])
def test_golden_io_edge_page(version, suffix, update_golden_io):
    check_golden_io_xml(f'page_xml/edge_page.{suffix}.xml', build_edge_layout().to_pagexml_string(version=version),
                     update_golden_io)


def test_golden_io_empty_page(update_golden_io):
    check_golden_io_xml('page_xml/empty_page.2019.xml', build_empty_layout().to_pagexml_string(), update_golden_io)
