"""The language model engine: loading a brnolm torchscript LM, wrapping it for the decoder, and LM-guided decoding.

The tiny LM (test/tiny_models.py) prefers 'b' after 'a' and is uniform otherwise; its state survives line ends."""

import unittest

import numpy as np
import pytest
import torch

from pero_ocr.decoding.decoders import BLANK_SYMBOL, CTCPrefixLogRawNumpyDecoder
from pero_ocr.decoding.decoding_itf import construct_lm
from pero_ocr.decoding.lm_wrapper import HiddenState, LMWrapper
from test import tiny_models
from test.dummy_lm import DummyLm
from test.synthetic import A, AMBIGUOUS_AFTER_A, BLANK, ctc_frames as frames

CHARS = tiny_models.CHARACTERS
CPU = torch.device('cpu')




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
    assert construct_lm('lm/model.pt', config_path=tiny_model_dir).vocab == construct_lm(f'{tiny_model_dir}/lm/model.pt').vocab


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


# --- LMWrapper on a hand-weighted LM --------------------------------------------------------------------------------
# Exact scores through the public interface of the wrapper (advance_h0, log_probs, initial_h, translate, eos_scores).

class LMWrapperTemplate:
    '''Set of tests that any LM wrapper needs to pass.

    Inheriting class needs to:
    - instantiate an LM wrapper in self._wrapper
    - prepare proper PyTorch device in self._device
    '''
    def test_single_advance(self):
        h0 = HiddenState(torch.tensor([-2.0]).to(self._device))
        h1 = self._wrapper.advance_h0(np.asarray([2]), h0)
        self.assertTrue((h1._h == torch.tensor([-2.0]).to(self._device)).all())

    def test_advance_batch(self):
        h0 = HiddenState(torch.tensor([[-2.0], [2.0]]).to(self._device))
        h1 = self._wrapper.advance_h0(np.asarray([2, 0]), h0)
        expected = torch.tensor([[-2.0], [4.0]]).to(self._device)
        self.assertTrue((h1._h == expected).all())

    def test_single_output(self):
        h = HiddenState(torch.tensor([[[1.0]]]).to(self._device))
        y = self._wrapper.log_probs(h)
        self.assertTrue((y == np.asarray([3, 3, 1])).all())

    def test_batch_output(self):
        h = HiddenState(torch.tensor([[[1.0], [-2.0]]]).to(self._device))
        y = self._wrapper.log_probs(h)
        self.assertTrue((y == np.asarray([[3, 3, 1], [-3, 3, -14]])).all())

    def test_initial_state(self):
        h_init = self._wrapper.initial_h(1)
        self.assertTrue((h_init._h == torch.tensor([[19.0]]).to(self._device)).all())

    def test_single_translation(self):
        translation = self._wrapper.translate(np.asarray([0]))
        self.assertTrue((translation == np.asarray([1])).all())

    def test_batch_translation(self):
        translation = self._wrapper.translate(np.asarray([0, 1]))
        self.assertTrue((translation == np.asarray([1, 2])).all())

    def test_prefix_cutting_translation(self):
        translation = self._wrapper.translate(np.asarray([0, 1]))
        self.assertTrue((translation == np.asarray([1, 2])).all())

    def test_eos_score(self):
        h = HiddenState(torch.tensor([[[0.0], [0.2]]]).to(self._device))
        eos_scores = self._wrapper.eos_scores(h)
        self.assertTrue((eos_scores == np.asarray([-100.0, -120.0])).all())


class TorchCPULmWrapperTests(LMWrapperTemplate, unittest.TestCase):
    def setUp(self):
        self._device = torch.device('cpu')
        self._wrapper = LMWrapper(DummyLm(), ['a', 'b', 'c'], device=self._device)

