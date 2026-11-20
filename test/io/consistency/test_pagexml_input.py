"""PAGE XML files written by pero-ocr must stay readable by pero-ocr (it also takes PAGE XML as input, e.g.
parse_folder.py -x). The committed golden files serve as previously written files."""
import pytest

from pero_ocr.core.layout import PageLayout
from test.io.consistency.golden import golden_path
from test.io.reference_layout import assert_layouts_equal, build_reference_layout, build_edge_layout


def test_golden_file_loads(update_golden):
    if update_golden:
        pytest.skip('golden files are being rewritten')
    loaded = PageLayout(file=golden_path('page_xml/reference_page.2019.xml'))
    assert_layouts_equal(loaded, build_reference_layout())


@pytest.mark.parametrize('name', ['page_xml/edge_page.2019.xml', 'page_xml/edge_page.2013.xml'])
def test_golden_edge_page_loads(name, update_golden):
    if update_golden:
        pytest.skip('golden files are being rewritten')
    edge = build_edge_layout()
    loaded = PageLayout(file=golden_path(name))
    assert [r.id for r in loaded.regions] == [r.id for r in edge.regions]
    assert loaded.regions[0].transcription == 'region level text'
    assert [line.transcription for line in loaded.lines_iterator()] == \
        [line.transcription for line in edge.lines_iterator()]
    assert [line.category for line in loaded.lines_iterator()] == [line.category for line in edge.lines_iterator()]


def test_golden_validate_id_page_loads(update_golden):
    if update_golden:
        pytest.skip('golden files are being rewritten')
    loaded = PageLayout(file=golden_path('page_xml/reference_page.2019.validate_id.xml'))
    reference = build_reference_layout()
    assert [line.id for line in loaded.lines_iterator()] == ['id_' + line.id for line in reference.lines_iterator()]


def test_golden_empty_page_loads(update_golden):
    if update_golden:
        pytest.skip('golden files are being rewritten')
    loaded = PageLayout(file=golden_path('page_xml/empty_page.2019.xml'))
    assert (loaded.id, loaded.page_size, loaded.regions) == ('empty_page.jpg', (400, 300), [])
