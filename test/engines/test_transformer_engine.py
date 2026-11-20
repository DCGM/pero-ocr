import pytest
from scipy import sparse

from pero_ocr.ocr_engine.transformer_ocr_engine import TransformerEngineLineOCR
from test import tiny_models
from test.synthetic import line_image

CHARS = tiny_models.CHARACTERS
N_OUT = len(CHARS) + 2


def expected_text(n_runs):
    return ''.join(CHARS[1 + i % (len(CHARS) - 1)] for i in range(n_runs))


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
    transcriptions, _, _ = engine.process_lines([line_image(n_runs)])
    assert transcriptions == [expected_text(n_runs)]


def test_lines_of_a_batch_finish_at_different_steps(engine):
    runs = [5, 1, 0, 3, 8, 2]
    transcriptions, _, _ = engine.process_lines([line_image(n) for n in runs])
    assert transcriptions == [expected_text(n) for n in runs]


def test_logits_format(engine):
    runs = [3, 1]
    _, logits, coords = engine.process_lines([line_image(n) for n in runs])
    for n, line_logits, line_coords in zip(runs, logits, coords):
        assert isinstance(line_logits, sparse.csc_matrix)
        assert line_logits.shape == (n, N_OUT)
        assert line_coords == [0, n]


@pytest.mark.parametrize('beam_size', [2, 4])
@pytest.mark.parametrize('beam_merging', ['best', 'cn'])
def test_beam_search_matches_greedy_on_unambiguous_input(transformer_json, cpu, beam_size, beam_merging):
    engine = TransformerEngineLineOCR(transformer_json, cpu, beam_size=beam_size, beam_merging=beam_merging)
    runs = [4, 0, 2]
    transcriptions, logits, _ = engine.process_lines([line_image(n) for n in runs])
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
    transcriptions, _, _ = engine.process_lines([line_image(n) for n in runs])
    assert transcriptions == [expected_text(n) for n in runs]
