"""ALTO export must keep producing the committed golden IO files (test/io/consistency/data/alto)."""
import pytest

from pero_ocr.core.layout import ALTOVersion
from test.io.consistency.golden_io import check_golden_io_xml
from test.io.reference_layout import build_reference_layout, build_edge_layout, build_empty_layout


@pytest.fixture
def layout():
    return build_reference_layout()


def test_golden_io_alto_v2(layout, update_golden_io):
    check_golden_io_xml('alto/reference_page.v2.xml', layout.to_altoxml_string(), update_golden_io)


def test_golden_io_alto_v4(layout, update_golden_io):
    check_golden_io_xml('alto/reference_page.v4.xml', layout.to_altoxml_string(version=ALTOVersion.ALTO_v4_4),
                     update_golden_io)


def test_golden_io_alto_with_page_uuid_and_processing_element(layout, update_golden_io):
    from pero_ocr.core.layout import create_ocr_processing_element
    processing = create_ocr_processing_element(id='IdOcrCustom', software_creator_str='Creator',
                                               software_name_str='Name', software_version_str='v9.9.9',
                                               processing_datetime='2026-01-01T00:00:00')
    xml = layout.to_altoxml_string(ocr_processing_element=processing, page_uuid='0f1e2d3c-page-uuid')
    check_golden_io_xml('alto/reference_page.v2.page_uuid.xml', xml, update_golden_io)


@pytest.mark.parametrize('version, suffix', [(ALTOVersion.ALTO_v2_x, 'v2'), (ALTOVersion.ALTO_v4_4, 'v4')])
def test_golden_io_edge_page(version, suffix, update_golden_io):
    check_golden_io_xml(f'alto/edge_page.{suffix}.xml', build_edge_layout().to_altoxml_string(version=version), update_golden_io)


def test_golden_io_empty_page(update_golden_io):
    check_golden_io_xml('alto/empty_page.v2.xml', build_empty_layout().to_altoxml_string(), update_golden_io)
