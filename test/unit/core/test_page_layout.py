import numpy as np
import pytest
from scipy import sparse

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine, normalize_text
from pero_ocr.ocr_engine.softmax import softmax
from test.io.reference_layout import build_reference_layout


@pytest.fixture
def layout():
    return build_reference_layout()


def test_lines_iterator_all(layout):
    assert len(list(layout.lines_iterator())) == 6


def test_lines_iterator_categories(layout):
    """Lines without a category match every category filter."""
    assert len(list(layout.lines_iterator(['title']))) == 6
    assert 'r002-l002' not in [line.id for line in layout.lines_iterator(['text'])]
    assert len(list(layout.lines_iterator(['text']))) == 5


def test_calculate_confidence(layout):
    layout.calculate_confidence()
    assert 0.9 < layout.confidence <= 1.0


def test_calculate_confidence_ignores_lines_without_confidence():
    """Issue #73: lines without logits have no confidence and must not crash the page confidence."""
    page = PageLayout(id='p', page_size=(100, 100))
    region = RegionLayout('r1', np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
    region.lines = [TextLine(id='l1', transcription='a'), TextLine(id='l2', transcription='b')]
    page.regions = [region]
    for line in page.lines_iterator():
        line.calculate_confidences()
    page.calculate_confidence()
    assert page.confidence == 0


def test_rename_region_id_renames_lines(layout):
    layout.rename_region_id('r001', 'top')
    assert layout.regions[0].id == 'top'
    assert [line.id for line in layout.regions[0].lines] == ['top-l001', 'top-l002', 'top-l003']


def test_region_bounding_box(layout):
    assert layout.regions[0].get_polygon_bounding_box() == (40, 50, 760, 230)


def test_render_to_image(layout):
    image = np.zeros((1000, 800, 3), dtype=np.uint8)
    rendered = layout.render_to_image(image.copy(), render_order=True, render_category=True)
    assert rendered.shape == image.shape
    assert rendered.any()


def test_get_dense_logits_fills_zeros(layout):
    line = layout.regions[0].lines[0]
    dense = line.get_dense_logits(-50)
    assert dense.min() == -50
    logprobs = line.get_full_logprobs()
    np.testing.assert_allclose(np.exp(logprobs).sum(axis=1), 1, rtol=1e-5)


def test_calculate_confidences_without_logits():
    line = TextLine(id='l1', transcription='abc')
    line.calculate_confidences(default_transcription_confidence=0.5)
    assert line.character_confidences is None
    assert line.transcription_confidence is None


def test_calculate_confidences(layout):
    line = layout.regions[0].lines[0]
    assert len(line.character_confidences) == len(line.transcription)
    assert 0 < min(line.character_confidences) <= max(line.character_confidences) <= 1


def test_normalize_text():
    assert normalize_text('Příliš žluťoučký') == 'Prilis zlutoucky'


def test_dense_logits_from_sparse_logits():
    """Logits stored sparse (entries with a negligible probability dropped) come back dense, with the dropped entries
    filled in with the given value."""
    logits = np.array([[1.0, -20.0, -19.0], [0.1, 0.1, -21.0]])
    sparse_logits = logits.copy()
    sparse_logits[softmax(logits, axis=1) < 0.0001] = 0
    line = TextLine(logits=sparse.csc_matrix(sparse_logits))
    np.testing.assert_array_equal(line.get_dense_logits(-50.0), [[1.0, -50.0, -50.0], [0.1, 0.1, -50.0]])
