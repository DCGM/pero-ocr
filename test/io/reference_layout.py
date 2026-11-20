"""A hand-built PageLayout covering the features the exporters handle. It does not depend on any model."""
import numpy as np
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


def _line(id, index, x0, x1, y, text, heights=(12.4, 4.6), category=None, with_logits=True):
    baseline = np.array([[x0, y], [(x0 + x1) / 2, y + 1.4], [x1, y]], dtype=float)
    top = baseline - [0, heights[0]]
    bottom = baseline + [0, heights[1]]
    polygon = np.concatenate([top, bottom[::-1]])
    line = TextLine(id=id, index=index, baseline=baseline, polygon=polygon, heights=list(heights),
                    transcription=text, category=category)
    if with_logits and text is not None:
        line.logits = ctc_logits(text)
        line.characters = CHARACTERS
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
