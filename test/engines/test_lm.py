"""The language model engine: loading a brnolm torchscript LM, wrapping it for the decoder, and LM-guided decoding.

The tiny LM (test/tiny_models.py) prefers 'b' after 'a' and is uniform otherwise; its state survives line ends."""
import configparser
import math

import numpy as np
import pytest
import torch
from scipy import sparse

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.decoding.decoders import BLANK_SYMBOL, CTCPrefixLogRawNumpyDecoder
from pero_ocr.decoding.decoding_itf import construct_lm, decoder_factory
from pero_ocr.decoding.lm_wrapper import LMWrapper
from pero_ocr.document_ocr.page_parser import PageDecoder
from test import tiny_models

CHARS = tiny_models.CHARACTERS
N_CLASSES = len(CHARS) + 1  # + blank, last
CPU = torch.device('cpu')


def frames(*specs):
    """Log-probability CTC frames, e.g. frames({'a': .98}, {'blank': .98}, {'c': .6, 'b': .4})."""
    out = []
    for spec in specs:
        f = np.full(N_CLASSES, 1e-4)
        for symbol, p in spec.items():
            f[N_CLASSES - 1 if symbol == 'blank' else CHARS.index(symbol)] = p
        out.append(np.log(f / f.sum()))
    return np.stack(out)


A = {'a': .98}
BLANK = {'blank': .98}
C_OR_B = {'c': .6, 'b': .4}  # visually 'c' wins, an LM preferring 'b' can override it
AMBIGUOUS_AFTER_A = frames(A, BLANK, C_OR_B, BLANK)


@pytest.fixture(scope='module')
def lm(lm_path):
    return construct_lm(lm_path)


@pytest.fixture(scope='module')
def wrapper(lm):
    return LMWrapper(lm, CHARS, CPU)


def lm_decoder(wrapper, lm_scale=1.0, beam=8):
    return CTCPrefixLogRawNumpyDecoder(CHARS + [BLANK_SYMBOL], beam, wrapper, lm_scale)


# --- loading ------------------------------------------------------------------------------------------------------

def test_loads_brnolm_torchscript_lm(lm):
    assert lm._unused_prefix_len == 2
    assert lm.vocab['</s>'] == 0
    assert [lm.vocab[c] for c in CHARS] == list(range(2, 2 + len(CHARS)))


def test_path_relative_to_config_path(tiny_model_dir):
    assert construct_lm('lm/lm.pt', config_path=tiny_model_dir).vocab == construct_lm(f'{tiny_model_dir}/lm/lm.pt').vocab


def test_file_that_is_not_an_lm_raises(tmp_path):
    path = tmp_path / 'lm.pt'
    path.write_bytes(b'not a language model')
    with pytest.raises(Exception):
        construct_lm(str(path))


def test_ocr_characters_missing_from_lm_vocabulary_raise(lm):
    with pytest.raises(KeyError):
        LMWrapper(lm, CHARS + ['x'], CPU)


# --- the wrapper --------------------------------------------------------------------------------------------------

def test_initial_state_is_uniform(wrapper, lm):
    """log_probs() drops the two special tokens without renormalizing, so each character gets 1 / len(vocab)."""
    np.testing.assert_allclose(np.exp(wrapper.log_probs(wrapper.initial_h(1))), 1 / len(lm.vocab), rtol=1e-5)


def test_advancing_changes_the_prediction(wrapper):
    h = wrapper.advance_h0(np.array([CHARS.index('a')]), wrapper.initial_h(1))
    probs = np.exp(wrapper.log_probs(h))[0]
    assert probs.argmax() == CHARS.index('b')
    assert probs[CHARS.index('b')] > 0.9


def test_line_end_keeps_the_state(wrapper):
    h = wrapper.advance_h0(np.array([CHARS.index('a')]), wrapper.initial_h(1))
    after_end = wrapper.add_line_end(h)
    np.testing.assert_allclose(wrapper.log_probs(after_end), wrapper.log_probs(h))


def test_initial_h_from_line(wrapper):
    probs = np.exp(wrapper.log_probs(wrapper.initial_h_from_line('da')))[0]
    assert probs.argmax() == CHARS.index('b')


def test_eos_scores(wrapper):
    scores = wrapper.eos_scores(wrapper.initial_h(2))
    assert scores.shape == (2,)


# --- decoding -----------------------------------------------------------------------------------------------------

def test_without_lm_the_visual_model_decides():
    decoder = CTCPrefixLogRawNumpyDecoder(CHARS + [BLANK_SYMBOL], 8, None, 0.0)
    assert decoder(AMBIGUOUS_AFTER_A).best_hyp() == 'ac'


def test_lm_overrides_an_ambiguous_character(wrapper):
    assert lm_decoder(wrapper)(AMBIGUOUS_AFTER_A).best_hyp() == 'ab'


@pytest.mark.xfail(reason='BagOfHypotheses.best_hyp() adds the LM score with weight 1 instead of lm_weight, so the '
                          'final pick ignores LM_SCALE', strict=True)
def test_lm_scale_zero_disables_the_lm(wrapper):
    assert lm_decoder(wrapper, lm_scale=0.0)(AMBIGUOUS_AFTER_A).best_hyp() == 'ac'


def test_lm_does_not_override_a_clear_character(wrapper):
    clear_c = frames(A, BLANK, {'c': .98}, BLANK)
    assert lm_decoder(wrapper)(clear_c).best_hyp() == 'ac'


def test_decoder_factory_builds_lm_decoder(tiny_model_dir):
    config = configparser.ConfigParser()
    config.read_string('[DECODER]\nTYPE = FAST-LOG-RAW\nBEAM_SIZE = 8\nLM_SCALE = 1.0\nLM = lm/lm.pt\n')
    decoder = decoder_factory(config['DECODER'], list(CHARS), CPU, config_path=tiny_model_dir)
    assert isinstance(decoder._lm, LMWrapper)
    assert decoder(AMBIGUOUS_AFTER_A).best_hyp() == 'ab'


# --- PageDecoder: decoding a page with an LM ----------------------------------------------------------------------

def page_with_lines(*line_logits):
    """A page whose lines carry the given log-prob frames as (sparse) logits."""
    page = PageLayout(id='p', page_size=(100, 100))
    region = RegionLayout('r1', np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
    for i, logits in enumerate(line_logits):
        line = TextLine(id=f'r1-l{i}', transcription='?', category='text',
                        logits=None if logits is None else sparse.csc_matrix(logits.astype(np.float32)))
        region.lines.append(line)
    page.regions = [region]
    return page


B_OR_C_AT_LINE_START = frames(C_OR_B, BLANK)


@pytest.mark.parametrize('carry_h_over, expected', [(False, ['a', 'c']), (True, ['a', 'b'])])
def test_carry_h_over_lets_the_previous_line_guide_the_next(wrapper, carry_h_over, expected):
    page = page_with_lines(frames(A, BLANK), B_OR_C_AT_LINE_START)
    PageDecoder(lm_decoder(wrapper), line_confidence_threshold=math.inf, carry_h_over=carry_h_over).process_page(page)
    assert [line.transcription for line in page.lines_iterator()] == expected


def test_confident_lines_are_not_decoded(wrapper):
    page = page_with_lines(frames(A, BLANK), AMBIGUOUS_AFTER_A)
    page_decoder = PageDecoder(lm_decoder(wrapper), line_confidence_threshold=0.9)
    page_decoder.process_page(page)
    assert [line.transcription for line in page.lines_iterator()] == ['?', 'ab']  # the confident line keeps its text
    assert (page_decoder.lines_examined, page_decoder.lines_decoded) == (2, 1)
    assert page_decoder.decoding_summary().startswith('Ran on 2, decoded 1 lines (50.0 %)')


def test_a_failing_line_does_not_stop_the_page(wrapper):
    page = page_with_lines(None, AMBIGUOUS_AFTER_A)
    PageDecoder(lm_decoder(wrapper), line_confidence_threshold=math.inf).process_page(page)
    assert [line.transcription for line in page.lines_iterator()] == ['?', 'ab']


def test_summary_before_decoding(wrapper):
    assert PageDecoder(lm_decoder(wrapper)).decoding_summary() == 'This PageDecoder has not processed a single line yet'
