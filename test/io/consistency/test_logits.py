"""Logits export must keep producing the committed golden files (test/io/consistency/data/logits).

Pickles are compared by content (keys, matrix type, dtype, shape, values) and by pickle protocol, not byte by byte."""
import pickle
import shutil

import pytest

from test.io.consistency.golden import assert_logit_dicts_equal, golden_path
from test.io.reference_layout import build_reference_layout


@pytest.fixture
def layout():
    return build_reference_layout()


GOLDEN_LOGITS = 'logits/reference_page.logits'


def test_golden_logits(layout, tmp_path, update_golden):
    path = str(tmp_path / 'page.logits')
    layout.save_logits(path, missing_line_logits_ok=True)
    if update_golden:
        shutil.copyfile(path, golden_path(GOLDEN_LOGITS))
        return
    with open(path, 'rb') as f:
        actual = pickle.load(f)
    with open(golden_path(GOLDEN_LOGITS), 'rb') as f:
        expected = pickle.load(f)
    assert_logit_dicts_equal(actual, expected)


GOLDEN_LOGITS_BYTES = 'logits/reference_page.bytes.logits'


GOLDEN_EDGE_LOGITS = 'logits/edge_page.bytes.logits'


def _check_golden_pickle(name, data: bytes, update_golden):
    if update_golden:
        with open(golden_path(name), 'wb') as f:
            f.write(data)
        return
    with open(golden_path(name), 'rb') as f:
        golden = f.read()
    assert data[:2] == golden[:2], f'pickle protocol changed: {data[:2]!r} vs golden {golden[:2]!r}'
    assert_logit_dicts_equal(pickle.loads(data), pickle.loads(golden))


def test_golden_logits_bytes(layout, update_golden):
    """save_logits_bytes() pickles with HIGHEST_PROTOCOL, unlike the protocol 4 of save_logits()."""
    _check_golden_pickle(GOLDEN_LOGITS_BYTES, layout.save_logits_bytes(missing_line_logits_ok=True), update_golden)


def test_golden_edge_page_logits(update_golden):
    from test.io.reference_layout import build_edge_layout
    _check_golden_pickle(GOLDEN_EDGE_LOGITS, build_edge_layout().save_logits_bytes(missing_line_logits_ok=True),
                         update_golden)
