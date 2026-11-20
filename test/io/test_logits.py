import pickle

import numpy as np
import pytest
from scipy import sparse

from pero_ocr.core.layout import PageLayout

from test.io.reference_layout import build_reference_layout

META_KEYS = {'line_characters', 'logit_coords'}


@pytest.fixture
def layout():
    return build_reference_layout()


def test_logits_file_structure(layout, tmp_path):
    path = tmp_path / 'page.logits'
    layout.save_logits(str(path), missing_line_logits_ok=True)
    raw = path.read_bytes()
    assert raw[:2] == b'\x80\x04', 'logits are pickled with protocol 4'
    data = pickle.loads(raw)
    line_ids = [line.id for line in layout.lines_iterator()]
    assert set(data) == set(line_ids) | META_KEYS
    for line in layout.lines_iterator():
        if line.logits is None:
            continue
        assert isinstance(data[line.id], sparse.csc_matrix)
        assert data[line.id].dtype == np.float32
        assert data['line_characters'][line.id] == line.characters
        assert data['logit_coords'][line.id] == line.logit_coords


def test_missing_logits_raise(layout, tmp_path):
    with pytest.raises(Exception, match='Missing logits'):
        layout.save_logits(str(tmp_path / 'page.logits'))


@pytest.mark.xfail(reason='_gen_logits "continue"s an inner loop, so lines without logits are stored as None '
                          'instead of being left out', strict=True)
def test_missing_logits_ok_leaves_lines_out(layout):
    data = pickle.loads(layout.save_logits_bytes(missing_line_logits_ok=True))
    assert 'r001-l003' not in data


def test_bytes_round_trip(layout):
    loaded = build_reference_layout()
    for line in loaded.lines_iterator():
        line.logits = line.characters = line.logit_coords = None
    loaded.load_logits(layout.save_logits_bytes(missing_line_logits_ok=True))
    for a, e in zip(loaded.lines_iterator(), layout.lines_iterator()):
        assert a.characters == e.characters
        assert a.logit_coords == e.logit_coords
        if e.logits is not None:
            np.testing.assert_array_equal(a.logits.toarray(), e.logits.toarray())


def test_load_logits_matches_lines_by_id(layout, tmp_path):
    path = str(tmp_path / 'page.logits')
    layout.save_logits(path, missing_line_logits_ok=True)
    other = PageLayout(id='other', page_size=layout.page_size)
    other.regions = build_reference_layout().regions[1:2]
    for line in other.lines_iterator():
        line.logits = None
    other.load_logits(path)
    assert all(line.logits is not None for line in other.lines_iterator())


def test_load_legacy_logits_without_metadata(layout, tmp_path):
    """Old logit files hold only {line_id: matrix}."""
    legacy = {line.id: line.logits for line in layout.lines_iterator() if line.logits is not None}
    path = tmp_path / 'legacy.logits'
    path.write_bytes(pickle.dumps(legacy))
    loaded = build_reference_layout()
    loaded.load_logits(str(path))
    line = loaded.regions[0].lines[0]
    assert line.characters is None
    assert line.logit_coords == [None, None]


# --- variants ----------------------------------------------------------------------------------------------------


def test_empty_page_logits():
    from test.io.reference_layout import build_empty_layout
    data = pickle.loads(build_empty_layout().save_logits_bytes())
    assert data == {'line_characters': {}, 'logit_coords': {}}
