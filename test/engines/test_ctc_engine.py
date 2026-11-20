import os
import shutil

import numpy as np
import pytest
from scipy import sparse

from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR, greedy_decode_ctc
from test import tiny_models
from test.conftest import FROZEN_MODELS
from test.synthetic import line_image

N_OUT = len(tiny_models.CHARACTERS) + 1


@pytest.fixture(scope='module')
def engine(ctc_json, cpu):
    return PytorchEngineLineOCR(ctc_json, cpu)


def test_characters_get_blank_appended(engine):
    assert engine.characters == tiny_models.CHARACTERS + ['​']


@pytest.mark.parametrize('n_runs', [0, 1, 2, 7])
def test_transcription(engine, n_runs):
    transcriptions, _, _ = engine.process_lines([line_image(n_runs)])
    assert transcriptions == ['a' * n_runs]


def test_logits_format(engine):
    lines = [line_image(3), line_image(1)]
    _, logits, coords = engine.process_lines(lines)
    for line, line_logits, line_coords in zip(lines, logits, coords):
        assert isinstance(line_logits, sparse.csc_matrix)
        assert line_logits.dtype == np.float32
        assert line_logits.shape[1] == N_OUT
        assert line_coords == [engine.line_padding_px // 4, (engine.line_padding_px + line.shape[1]) // 4]


def test_dense_logits(engine):
    _, logits, _ = engine.process_lines([line_image(2)], sparse_logits=False)
    assert isinstance(logits[0], np.ndarray)


def test_tight_crop_logits(engine):
    line = line_image(2)
    _, logits, coords = engine.process_lines([line], tight_crop_logits=True)
    assert coords == [[None, None]]
    assert logits[0].shape[0] == line.shape[1] // 4


def test_no_logits(engine):
    transcriptions, logits, coords = engine.process_lines([line_image(2)], no_logits=True)
    assert transcriptions == ['aa']
    assert logits == [None]


def test_order_is_kept_with_batching_by_width(engine):
    """Lines are batched sorted by width; results must come back in input order."""
    runs = [3, 0, 9, 1, 5, 2] * 12
    transcriptions, logits, _ = engine.process_lines([line_image(n) for n in runs])
    assert transcriptions == ['a' * n for n in runs]
    assert all(m is not None for m in logits)


def test_extreme_line_sizes(engine):
    lines = [np.zeros((tiny_models.LINE_PX_HEIGHT, 1, 3), dtype=np.uint8),
             line_image(1, run_width=4, gap=4),
             line_image(400)]
    transcriptions, _, _ = engine.process_lines(lines)
    assert transcriptions[0] == ''
    assert transcriptions[1] == 'a'
    assert transcriptions[2].count('a') > 100


def test_empty_input(engine):
    assert engine.process_lines([]) == ([], [], [])


@pytest.mark.parametrize('embed_id, expected_char', [(2, 'c'), ('mean', 'd')])
def test_embedding_id(tmp_path, cpu, embed_id, expected_char):
    json_path = tiny_models.write_ctc(str(tmp_path), embed_num=4, extra={'embed_id': embed_id})
    engine = PytorchEngineLineOCR(json_path, cpu)
    assert engine.process_lines([line_image(2)])[0] == [expected_char * 2]


def test_checkpoint_path_relative_to_json(tmp_path, cpu, ctc_json):
    target = tmp_path / 'elsewhere'
    shutil.copytree(os.path.dirname(ctc_json), target)
    engine = PytorchEngineLineOCR(str(target / 'ocr_engine.json'), cpu)
    assert engine.checkpoint == str(target / 'model.pt.cpu')


def test_cpu_requires_cpu_checkpoint(tmp_path, cpu):
    json_path = tiny_models.write_ctc(str(tmp_path))
    os.remove(os.path.join(str(tmp_path), 'model.pt.cpu'))
    with pytest.raises(Exception):
        PytorchEngineLineOCR(json_path, cpu)


def test_greedy_decode_ctc_collapses_repeats_and_blanks():
    import torch
    chars = ['x', 'y']
    blank = 2
    frames = [0, 0, blank, 0, 1, 1, blank]
    scores = torch.full((1, 3, len(frames)), -10.0)
    for t, k in enumerate(frames):
        scores[0, k, t] = 10.0
    assert greedy_decode_ctc(scores, chars) == ['xxy']


def test_frozen_model_still_works(engine, cpu):
    """A model exported by an older pero-ocr/torch (test/engines/data) loads and gives the same output."""
    frozen = PytorchEngineLineOCR(os.path.join(FROZEN_MODELS, 'ctc', 'ocr_engine.json'), cpu)
    lines = [line_image(n) for n in (0, 1, 4)]
    frozen_out, fresh_out = frozen.process_lines(lines), engine.process_lines(lines)
    assert frozen_out[0] == fresh_out[0] == ['', 'a', 'aaaa']
    for a, b in zip(frozen_out[1], fresh_out[1]):
        np.testing.assert_array_equal(a.toarray(), b.toarray())
