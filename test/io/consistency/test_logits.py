"""Logits export must keep producing the committed golden IO files (test/io/consistency/data/logits).

Pickles are compared by content (keys, matrix type, dtype, shape, values) and by pickle protocol, not byte by byte."""
import pickle
import shutil

import pytest

from test.io.consistency.golden_io import assert_logit_dicts_equal, golden_io_path
from test.io.reference_layout import build_reference_layout


@pytest.fixture
def layout():
    return build_reference_layout()


GOLDEN_IO_LOGITS = 'logits/reference_page.logits'


def test_golden_io_logits(layout, tmp_path, update_golden_io):
    path = str(tmp_path / 'page.logits')
    layout.save_logits(path, missing_line_logits_ok=True)
    if update_golden_io:
        shutil.copyfile(path, golden_io_path(GOLDEN_IO_LOGITS))
        return
    with open(path, 'rb') as f:
        actual = pickle.load(f)
    with open(golden_io_path(GOLDEN_IO_LOGITS), 'rb') as f:
        expected = pickle.load(f)
    assert_logit_dicts_equal(actual, expected)


GOLDEN_IO_LOGITS_BYTES = 'logits/reference_page.bytes.logits'


GOLDEN_IO_EDGE_LOGITS = 'logits/edge_page.bytes.logits'


def _check_golden_io_pickle(name, data: bytes, update_golden_io):
    if update_golden_io:
        with open(golden_io_path(name), 'wb') as f:
            f.write(data)
        return
    with open(golden_io_path(name), 'rb') as f:
        golden_io = f.read()
    assert data[:2] == golden_io[:2], f'pickle protocol changed: {data[:2]!r} vs golden IO {golden_io[:2]!r}'
    assert_logit_dicts_equal(pickle.loads(data), pickle.loads(golden_io))


def test_golden_io_logits_bytes(layout, update_golden_io):
    """save_logits_bytes() pickles with HIGHEST_PROTOCOL, unlike the protocol 4 of save_logits()."""
    _check_golden_io_pickle(GOLDEN_IO_LOGITS_BYTES, layout.save_logits_bytes(missing_line_logits_ok=True), update_golden_io)


def test_golden_io_edge_page_logits(update_golden_io):
    from test.io.reference_layout import build_edge_layout
    _check_golden_io_pickle(GOLDEN_IO_EDGE_LOGITS, build_edge_layout().save_logits_bytes(missing_line_logits_ok=True),
                         update_golden_io)
