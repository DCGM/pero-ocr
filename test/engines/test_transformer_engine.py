"""The transformer OCR engine (TransformerEngineLineOCR) with the tiny transformer: one character per block of ink,
the character given by the block's colour (test/tiny_models.py)."""
import numpy as np
import pytest
from scipy import sparse

from pero_ocr.ocr_engine.transformer_ocr_engine import TransformerEngineLineOCR
from test import tiny_models
from test.synthetic import line_image, text_line

CHARS = tiny_models.CHARACTERS
N_OUT = len(CHARS) + 2


def expected_text(n_runs):
    """A text of n_runs characters that uses the whole alphabet: 'abcd abcd ...'."""
    return ''.join(CHARS[1 + i % (len(CHARS) - 1)] for i in range(n_runs))


def lines(*runs):
    return [text_line(expected_text(n)) for n in runs]


@pytest.fixture(scope='module')
def engine(transformer_json, cpu):
    return TransformerEngineLineOCR(transformer_json, cpu)


def test_loads_torchscript(engine):
    assert engine.exported
    assert engine.characters[-2:] == ['​', '']
    assert engine.sentence_boundary_ind == len(CHARS)


def test_exported_as_string_true(tmp_path, cpu):
    """Production configs write "exported": "true"."""
    json_path = tiny_models.write_transformer(str(tmp_path), extra={'exported': 'true'})
    assert TransformerEngineLineOCR(json_path, cpu).exported


@pytest.mark.parametrize('n_runs', [0, 1, 3, 6])
def test_transcription(engine, n_runs):
    transcriptions, _, _ = engine.process_lines(lines(n_runs))
    assert transcriptions == [expected_text(n_runs)]


def test_lines_of_a_batch_finish_at_different_steps(engine):
    runs = [5, 1, 0, 3, 8, 2]
    transcriptions, _, _ = engine.process_lines(lines(*runs))
    assert transcriptions == [expected_text(n) for n in runs]


def test_logits_format(engine):
    runs = [3, 1]
    _, logits, coords = engine.process_lines(lines(*runs))
    for n, line_logits, line_coords in zip(runs, logits, coords):
        assert isinstance(line_logits, sparse.csc_matrix)
        assert line_logits.shape == (n, N_OUT)
        assert line_coords == [0, n]


@pytest.mark.parametrize('beam_size', [2, 4])
@pytest.mark.parametrize('beam_merging', ['best', 'cn'])
def test_beam_search_matches_greedy_on_unambiguous_input(transformer_json, cpu, beam_size, beam_merging):
    engine = TransformerEngineLineOCR(transformer_json, cpu, beam_size=beam_size, beam_merging=beam_merging)
    runs = [4, 0, 2]
    transcriptions, logits, _ = engine.process_lines(lines(*runs))
    assert transcriptions == [expected_text(n) for n in runs]
    assert [m.shape[0] for m in logits] == runs


def test_unknown_beam_merging(transformer_json, cpu):
    engine = TransformerEngineLineOCR(transformer_json, cpu, beam_size=2, beam_merging='nonsense')
    with pytest.raises(ValueError):
        engine.process_lines([line_image(1)])


def test_decoding_without_end_token_is_bounded(tmp_path, cpu):
    json_path = tiny_models.write_transformer(str(tmp_path), stop=False)
    engine = TransformerEngineLineOCR(json_path, cpu)
    transcriptions, logits, _ = engine.process_lines([line_image(1)])
    assert len(transcriptions[0]) == engine.max_decoded_seq_length
    assert logits[0].shape[0] == engine.max_decoded_seq_length


def test_long_lines_are_split_and_merged(tmp_path, cpu):
    json_path = tiny_models.write_transformer(str(tmp_path), extra={'max_line_width': 256})
    engine = TransformerEngineLineOCR(json_path, cpu)
    line = line_image(30)
    transcriptions, logits, coords = engine.process_lines([line])
    assert transcriptions[0]
    assert logits[0].shape[0] == len(transcriptions[0])
    assert coords == [[0, len(transcriptions[0])]]


def test_order_is_kept_with_batching_by_width(engine):
    runs = [3, 0, 9, 1, 5, 2] * 8
    transcriptions, _, _ = engine.process_lines(lines(*runs))
    assert transcriptions == [expected_text(n) for n in runs]


@pytest.mark.parametrize('atomic', [True, False])
def test_output_substitution_table_from_the_json(tmp_path, cpu, atomic):
    """As for the CTC engine: the json's output_substitution_table becomes a translator, atomic per line by default."""
    json_path = tiny_models.write_transformer(str(tmp_path), extra={'output_substitution_table': {'a': 'ALPHA'}})
    substitution = TransformerEngineLineOCR(json_path, cpu, substitute_output_atomic=atomic).output_substitution
    assert substitution('a a') == 'ALPHA ALPHA'
    assert substitution('a b') == (None if atomic else 'ALPHA b')


def test_no_output_substitution_without_a_table(engine):
    assert engine.output_substitution is None


def test_white_blocks_read_as_the_first_character(engine):
    transcriptions, _, _ = engine.process_lines([line_image(3)])
    assert transcriptions == ['aaa']


def test_doubt_makes_the_previous_character_likely(engine):
    """A 'c' drawn with doubt 0.45 is still read as 'c', with 'b' close behind; with doubt 0.6 'b' wins."""
    transcriptions, logits, _ = engine.process_lines([text_line('ac', [0.0, 0.45]), text_line('ac', [0.0, 0.6])])
    assert transcriptions == ['ac', 'ab']
    second = logits[0].toarray()[1]
    probs = np.exp(second - np.logaddexp.reduce(second))
    assert probs[CHARS.index('c')] > probs[CHARS.index('b')] > 0.1


def test_reads_the_same_text_as_the_ctc_model(engine, ctc_json, cpu):
    """Like a production CTC + transformer pair: both read the same text from the same crops."""
    from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR
    crops = [text_line('bad cab'), text_line('d'), line_image(0)]
    assert engine.process_lines(crops)[0] == PytorchEngineLineOCR(ctc_json, cpu).process_lines(crops)[0] \
        == ['bad cab', 'd', '']
