"""A hand-built PageLayout covering the features the exporters handle. It does not depend on any model."""
import numpy as np
import pytest
from scipy import sparse

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine

CHARACTERS = list(' -.,abcdefghijklmnopqrstuvwxyzHPZáíúýčňřšťůž\t')
TARGET_LOGIT = 8.0
RIVAL_LOGIT = 3.0


def ctc_logits(text: str, characters=CHARACTERS, rival_every: int = 3) -> sparse.csc_matrix:
    """Sparse CTC logits that decode to ``text``: blank, (char, blank)*. Blank is the last column.

    Every ``rival_every``-th character frame also gets a weaker rival class so that confidences are below 1."""
    blank = len(characters)
    frames = [blank]
    for c in text:
        frames += [characters.index(c), blank]
    dense = np.zeros((len(frames), len(characters) + 1), dtype=np.float32)
    for t, k in enumerate(frames):
        dense[t, k] = TARGET_LOGIT
        if k != blank and (t // 2) % rival_every == 0:
            dense[t, (k + 1) % blank] = RIVAL_LOGIT
    return sparse.csc_matrix(dense)


def _line(id, index, x0, x1, y, text, heights=(12.4, 4.6), category=None, with_logits=True, characters=CHARACTERS):
    baseline = np.array([[x0, y], [(x0 + x1) / 2, y + 1.4], [x1, y]], dtype=float)
    top = baseline - [0, heights[0]]
    bottom = baseline + [0, heights[1]]
    polygon = np.concatenate([top, bottom[::-1]])
    line = TextLine(id=id, index=index, baseline=baseline, polygon=polygon, heights=list(heights),
                    transcription=text, category=category)
    if with_logits and text is not None:
        line.logits = ctc_logits(text, characters)
        line.characters = characters
        line.logit_coords = [0, line.logits.shape[0]]
        line.calculate_confidences()
    return line


def build_reference_layout() -> PageLayout:
    page = PageLayout(id='reference_page.jpg', page_size=(1000, 800))

    r1 = RegionLayout('r001', np.array([[40.3, 50.6], [760, 50], [760, 230], [40, 230]]))
    r1.lines = [
        _line('r001-l001', 0, 60, 740, 100.4, 'hello world, hyphen-'),
        _line('r001-l002', 1, 60, 520, 150, 'ated text.'),
        _line('r001-l003', 2, 60, 300, 200, None, with_logits=False),
    ]

    r2 = RegionLayout('r002', np.array([[40, 300], [760, 300], [760, 480], [40, 480]]),
                      region_type='paragraph', category='text', detection_confidence=0.87654)
    r2.lines = [
        _line('r002-l001', 0, 60, 740, 350, 'Příliš žluťoučký kůň'),
        _line('r002-l002', 1, 60, 400, 400, 'Header line', category='title'),
    ]

    # region partly outside of the page, coordinates must be clamped to 0 (issue #84)
    r3 = RegionLayout('r003', np.array([[-15, 520], [300, 520], [300, 600], [-15, 600]]))
    r3.lines = [_line('r003-l001', 0, -10, 280, 570, 'margin note')]

    page.regions = [r1, r2, r3]
    page.reading_order = {'r002': 1, 'r001': 0, 'r003': 2}
    return page


ARABIC_LABEL_FORM = 'ملاعلاب ابحرم'  # ArabicHelper().string_to_label_form('مرحبا بالعالم')
EDGE_CHARACTERS = CHARACTERS + ['B', 'T'] + sorted(set(ARABIC_LABEL_FORM) - set(CHARACTERS))


def _box(x0, y0, x1, y1):
    """Closed axis-aligned rectangle, as the YOLO layout engine writes it."""
    return np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]], dtype=float)


def build_edge_layout() -> PageLayout:
    """Output features the main reference page does not have."""
    page = PageLayout(id='scan 01#(a).tif', page_size=(600, 500))  # characters ALTO has to sanitize in the page ID

    r1 = RegionLayout('r001', np.array([[20, 20], [480, 20], [480, 260], [20, 260]]))
    r1.transcription = 'region level text'
    r1.lines = [
        _line('r001-l001', 0, 40, 460, 60, '', with_logits=False),  # recognized as empty
        _line('r001-l002', 1, 40, 460, 110, 'text without logits', with_logits=False),
        _line('r001-l003', 2, 40, 460, 160, ARABIC_LABEL_FORM, characters=EDGE_CHARACTERS),
        _line('r001-l004', 3, 40, 460, 210, 'abc def', characters=EDGE_CHARACTERS),
    ]

    # regions as LayoutExtractorYolo produces them: a box with a single line of the same category
    r2 = RegionLayout('r002', _box(30, 300, 230, 380), category='barcode', detection_confidence=0.51234)
    r2_line = TextLine(id='r002-l000', index=0, polygon=_box(30, 300, 230, 380),
                       baseline=np.array([[30, 340.0], [230, 340.0]]), heights=np.array([40.0, 40.0]),
                       category='barcode')
    r2.lines = [r2_line]

    r3 = RegionLayout('r003', _box(250, 300, 470, 340), category='title', detection_confidence=0.9)
    r3_line = _line('r003-l000', 0, 250, 470, 330, 'Big Title', category='title', characters=EDGE_CHARACTERS)
    r3.lines = [r3_line]

    page.regions = [r1, r2, r3]
    return page


def build_empty_layout() -> PageLayout:
    return PageLayout(id='empty_page.jpg', page_size=(400, 300))


def assert_layouts_equal(a: PageLayout, b: PageLayout):
    assert a.id == b.id
    assert a.page_size == b.page_size
    assert [r.id for r in a.regions] == [r.id for r in b.regions]
    for ra, rb in zip(a.regions, b.regions):
        assert ra.region_type == rb.region_type
        assert ra.category == rb.category
        assert ra.detection_confidence == pytest.approx(rb.detection_confidence if rb.detection_confidence is None
                                                        else round(rb.detection_confidence, 3))
        np.testing.assert_array_equal(np.asarray(ra.polygon), np.maximum(np.round(rb.polygon), 0))
        assert [line.id for line in ra.lines] == [line.id for line in rb.lines]
        for la, lb in zip(ra.lines, rb.lines):
            assert la.index == lb.index
            assert la.category == lb.category
            assert la.transcription == lb.transcription
            np.testing.assert_array_equal(la.baseline, np.maximum(np.round(lb.baseline), 0))
            np.testing.assert_array_equal(la.polygon, np.maximum(np.round(lb.polygon), 0))
            np.testing.assert_allclose(la.heights, np.round(lb.heights, 1))
            if lb.transcription is None:
                assert la.transcription_confidence is None
            else:
                assert la.transcription_confidence == pytest.approx(lb.transcription_confidence, abs=5e-4)
