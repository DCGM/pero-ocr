import unittest

import numpy as np
import torch

from pero_ocr.decoding.decoders import BLANK_SYMBOL
from pero_ocr.decoding.decoders import find_new_prefixes

from pero_ocr.decoding.decoders import GreedyDecoder
from pero_ocr.decoding.decoders import CTCPrefixLogRawNumpyDecoder
from pero_ocr.decoding.decoders import get_old_prefixes_positions, get_new_prefixes_positions
from pero_ocr.decoding.decoders import update_lm_things

from pero_ocr.decoding.lm_wrapper import HiddenState

from .test_lm_wrapper import DummyLm


class CTCPrefixDecodersBeam1Tests:
    def test_single_frame(self):
        logits = np.asarray([
            [0, -80.0, -80.0, -80.0],
        ])

        boh = self.decoder(logits)
        hyp = boh.best_hyp()

        self.assertEqual(hyp, 'a')

    def test_single_blank_score(self):
        logits = np.asarray([
            [-80.0, -80.0, -80.0, -5.0],
        ])

        boh = self.decoder(logits, max_unnormalization=np.inf)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, '')
        self.assertEqual(boh._hyps[0].vis_sc, -5.0)

    def test_trivial(self):
        logits = np.asarray([
            [0, -80.0, -80.0, -80.0],
            [0, -80.0, -80.0, -80.0],
        ])

        boh = self.decoder(logits)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'a')

    def test_double_symbol(self):
        logits = np.asarray([
            [0, -80.0, -80.0, -80.0],
            [-80.0, -80.0, -80.0, 0.0],
            [0, -80.0, -80.0, -80.0],
        ])

        boh = self.decoder(logits)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'aa')

    def test_two_symbols_immediate(self):
        logits = np.asarray([
            [0, -80.0, -80.0, -80.0],
            [-80.0, 0.0, -80.0, -80.0],
        ])

        boh = self.decoder(logits)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'ab')

    def test_continued_symbol(self):
        logits = np.asarray([
            [0, -80.0, -80.0, -80.0],
            [0, -80.0, -80.0, -80.0],
            [-80.0, -80.0, -80.0, 0.0],
        ])

        boh = self.decoder(logits)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'a')

    def test_continued_symbol_regression(self):
        logits = np.asarray([
            [-7e-2, -80.0, -80.0, -2.0],
            [-4e-4, -80.0, -80.0, -7.0],
            [-9e-1, -80.0, -80.0, -5e-1],
            [-80.0, -80.0, -80.0, 0.0],
        ])

        boh = self.decoder(logits, max_unnormalization=np.inf)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'a')

    def test_require_log_probs(self):
        logits = np.asarray([
            [-10.0, -80.0, -80.0, -10.0],
        ])

        self.assertRaises(ValueError, self.decoder, logits)


class CTCPrefixDecoderWiderBeamTests:
    def test_prefix_joining_regression(self):
        logits = np.asarray([
            [-2, -10, -80.0, -2.0],
            [-4e-4, -80.0, -80.0, -7.0],
            [-9e-1, -80.0, -80.0, -5e-1],
            [-80.0, -80.0, -80.0, 0.0],
        ])

        boh = self.decoder(logits, max_unnormalization=np.inf)
        all_transcripts = list(hyp.transcript for hyp in boh)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'a')
        self.assertEqual(len(set(all_transcripts)), 2)
        self.assertEqual(set(all_transcripts), {'a', ''})


class GreedyDecoderTests(CTCPrefixDecodersBeam1Tests, unittest.TestCase):
    def setUp(self):
        letters = ['a', 'b', 'c']
        self.decoder = GreedyDecoder(letters+[BLANK_SYMBOL])


class CTCPrefixLogRawNumpyDecoderBeam2Tests(CTCPrefixDecodersBeam1Tests, CTCPrefixDecoderWiderBeamTests, unittest.TestCase):
    def setUp(self):
        letters = ['a', 'b', 'c']
        self.decoder = CTCPrefixLogRawNumpyDecoder(letters+[BLANK_SYMBOL], k=2)


class BlankCheckTests(unittest.TestCase):
    def test_greedy_decoder_uniqueness(self):
        self.assertRaises(ValueError, GreedyDecoder, ['a', BLANK_SYMBOL, 'b'] + [BLANK_SYMBOL])

    def test_greedy_decoder_blank_at_end(self):
        self.assertRaises(ValueError, GreedyDecoder, ['a', BLANK_SYMBOL, 'b'])

    def test_greedy_decoder_missing_blank(self):
        self.assertRaises(ValueError, GreedyDecoder, ['a', 'b'])

    def test_rawlog_decoder_uniqueness(self):
        self.assertRaises(ValueError, CTCPrefixLogRawNumpyDecoder, ['a', BLANK_SYMBOL, 'b'] + [BLANK_SYMBOL], k=2)

    def test_rawlog_decoder_blank_at_end(self):
        self.assertRaises(ValueError, CTCPrefixLogRawNumpyDecoder, ['a', BLANK_SYMBOL, 'b'], k=2)

    def test_rawlog_decoder_missing_blank(self):
        self.assertRaises(ValueError, CTCPrefixLogRawNumpyDecoder, ['a', 'b'], k=2)


class CTCPrefixLogRawNumpyDecoderBeam1Tests(CTCPrefixDecodersBeam1Tests, unittest.TestCase):
    def setUp(self):
        letters = ['a', 'b', 'c']
        self.decoder = CTCPrefixLogRawNumpyDecoder(letters+[BLANK_SYMBOL], k=1)

    def test_beam_not_int(self):
        letters = ['a', 'b', 'c']
        self.assertRaises(TypeError, CTCPrefixLogRawNumpyDecoder, letters+[BLANK_SYMBOL], k=None)

    def test_beam_not_positive(self):
        letters = ['a', 'b', 'c']
        self.assertRaises(ValueError, CTCPrefixLogRawNumpyDecoder, letters+[BLANK_SYMBOL], k=0)


class CTCDecodingWithLMTests:
    def get_lm(self, a=-10.0, b=-10.0, c=-10.0):
        lm = DummyLm()
        lm.decoder._model_o.weight[1, 0] = 0.0
        lm.decoder._model_o.weight[2, 0] = 0.0
        lm.decoder._model_o.weight[3, 0] = 0.0
        lm.decoder._model_o.bias[1] = a
        lm.decoder._model_o.bias[2] = b
        lm.decoder._model_o.bias[3] = c
        return lm

    def test_single_selection_a(self):
        lm = self.get_lm(a=-1)
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=1,
            lm=lm
        )
        logits = np.asarray([
            [-1, -1, -80.0, -80.0],
        ])

        boh = decoder(logits, max_unnormalization=np.inf)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'a')

        for h in boh:
            self.assertEqual(h.lm_sc, lm.single_sentence_nll(list(h.transcript), '</s>'))

    def test_single_selection_b(self):
        lm = self.get_lm(b=-1)
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=1,
            lm=lm
        )
        logits = np.asarray([
            [-1, -1, -80.0, -80.0],
        ])

        boh = decoder(logits, max_unnormalization=np.inf)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'b')

        for h in boh:
            self.assertEqual(h.lm_sc, lm.single_sentence_nll(list(h.transcript), '</s>'))

    def test_insertion_bonus(self):
        lm = self.get_lm(a=-1, b=-1, c=-1)
        insertion_bonus = 0.5
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=1,
            lm=lm,
            insertion_bonus=insertion_bonus,
        )
        logits = np.asarray([
            [-80.0, -0.1, -80.0, -0.7],
        ])

        boh = decoder(logits, max_unnormalization=np.inf)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'b')

        for h in boh:
            true_lm_nll = lm.single_sentence_nll(list(h.transcript), '</s>')
            self.assertEqual(h.lm_sc, true_lm_nll + insertion_bonus*len(h.transcript))

    def test_single_selection_repeated_b(self):
        lm = self.get_lm(b=-1)
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=1,
            lm=lm
        )
        logits = np.asarray([
            [-1, -1, -80.0, -80.0],
            [-1, -1, -80.0, -80.0],
        ])

        boh = decoder(logits, max_unnormalization=np.inf)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'b')

        for h in boh:
            self.assertEqual(h.lm_sc, lm.single_sentence_nll(list(h.transcript), '</s>'))

    def get_bying_lm(self):
        lm = DummyLm()
        lm.model._model_r.weight[0, 0] = 2
        lm.model._model_r.bias[0] = 0
        lm.model._model_i.weight[0, 0] = 0
        lm.decoder._model_o.weight[1, 0] = -0.0
        lm.decoder._model_o.weight[2, 0] = -1.0
        lm.decoder._model_o.weight[3, 0] = -2.0
        lm.decoder._model_o.bias[1] = -10
        lm.decoder._model_o.bias[2] = 0
        lm.decoder._model_o.bias[3] = 30
        return lm

    def test_switching_lm_b(self):
        lm = self.get_bying_lm()
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=1,
            lm=lm
        )
        logits = np.asarray([
            [-1, -80.0, -80.0, -80.0],
            [-80.0, -1.0, -1.0, -80.0],
        ])

        boh = decoder(logits, max_unnormalization=np.inf)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'ab')

        for h in boh:
            self.assertEqual(h.lm_sc, lm.single_sentence_nll(list(h.transcript), '</s>'))

    def get_cying_lm(self):
        lm = DummyLm()
        lm.model._model_r.weight[0, 0] = 2
        lm.model._model_r.bias[0] = 0
        lm.model._model_i.weight[0, 0] = 0
        lm.decoder._model_o.weight[1, 0] = -0.0
        lm.decoder._model_o.weight[2, 0] = -2.0
        lm.decoder._model_o.weight[3, 0] = -1.0
        lm.decoder._model_o.bias[1] = -10
        lm.decoder._model_o.bias[2] = 30
        lm.decoder._model_o.bias[3] = 0
        return lm

    def test_switching_lm_c(self):
        lm = self.get_cying_lm()
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=1,
            lm=lm
        )
        logits = np.asarray([
            [-1, -80.0, -80.0, -80.0],
            [-80.0, -1.0, -1.0, -80.0],
        ])

        boh = decoder(logits, max_unnormalization=np.inf)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'ac')

        for h in boh:
            self.assertEqual(h.lm_sc, lm.single_sentence_nll(list(h.transcript), '</s>'))

    def get_eosing_lm(self):
        lm = DummyLm()
        lm.model._model_r.weight[0, 0] = 2
        lm.model._model_r.bias[0] = 0
        lm.model._model_i.weight[0, 0] = 0
        lm.model._model_i.weight[1, 0] = 1
        lm.model._model_i.weight[0, 0] = 0
        lm.decoder._model_o.weight[0, 0] = 1.0
        lm.decoder._model_o.weight[1, 0] = 0.0
        lm.decoder._model_o.weight[2, 0] = 0.0
        lm.decoder._model_o.weight[3, 0] = 0.0
        lm.decoder._model_o.bias[0] = -2
        lm.decoder._model_o.bias[1] = -1
        lm.decoder._model_o.bias[2] = -1
        lm.decoder._model_o.bias[3] = -1
        return lm

    def test_respecting_eos(self):
        lm = self.get_eosing_lm()
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=2,
            lm=lm
        )
        logits = np.asarray([
            [-80.0, -2.0, -80.0, -1.0],
        ])

        boh = decoder(logits, model_eos=True, max_unnormalization=np.inf)
        hyp = boh.best_hyp()
        self.assertEqual(hyp, 'b')

        for h in boh:
            self.assertEqual(h.lm_sc, lm.single_sentence_nll(list(h.transcript) + ['</s>'], '</s>'))

    def test_archiving_lm_scale(self):
        lm = self.get_eosing_lm()
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=2,
            lm=lm,
            lm_scale=0.1
        )
        logits = np.asarray([
            [-80.0, -2.0, -80.0, -1.0],
        ])

        boh = decoder(logits, model_eos=True, max_unnormalization=np.inf)
        self.assertEqual(boh.lm_weight, 0.1)

    def test_beam_2(self):
        lm = self.get_cying_lm()
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=2,
            lm=lm
        )
        logits = np.asarray([
            [-1, -80.0, -80.0, -80.0],
            [-80.0, -1.0, -1.0, -80.0],
        ])

        boh = decoder(logits, max_unnormalization=np.inf)
        hyp = boh.best_hyp()
        self.assertEqual(len(boh), 2)
        self.assertEqual(hyp, 'ac')

        for h in boh:
            self.assertEqual(h.lm_sc, lm.single_sentence_nll(list(h.transcript), '</s>'))

    def test_decoder_returns_hidden_state_of_best_hyp(self):
        lm = self.get_cying_lm()
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=2,
            lm=lm
        )
        logits = np.asarray([
            [-1, -80.0, -80.0, -80.0],
            [-80.0, -1.0, -1.0, -80.0],
        ])

        boh, last_h = decoder(logits, max_unnormalization=np.inf, return_h=True)
        hyp = boh.best_hyp()
        self.assertEqual(len(boh), 2)
        self.assertEqual(hyp, 'ac')

        self.assertEqual(last_h._h, torch.tensor([85.0]))

    def test_decoder_accepts_hidden_state(self):
        lm = self.get_cying_lm()
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=2,
            lm=lm
        )
        logits = np.asarray([
            [-1, -80.0, -80.0, -80.0],
            [-80.0, -2.0, -1.0, -80.0],
        ])

        init_h = HiddenState(torch.tensor([[[1.0]]]))
        boh = decoder(logits, max_unnormalization=np.inf, init_h=init_h)
        hyp = boh.best_hyp()
        self.assertEqual(len(boh), 2)
        self.assertEqual(hyp, 'ab')

    def test_decoder_hidden_state_propagates(self):
        lm = self.get_cying_lm()
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=2,
            lm=lm
        )

        logits_1 = np.asarray([
            [-1, -80.0, -80.0, -80.0],
        ])
        logits_2 = np.asarray([
            [-80.0, -0.8, -1.0, -80.0],
        ])

        _, last_h = decoder(logits_1, max_unnormalization=np.inf, return_h=True)
        boh = decoder(logits_2, max_unnormalization=np.inf, init_h=last_h)
        hyp = boh.best_hyp()
        self.assertEqual(len(boh), 2)
        self.assertEqual(hyp, 'c')

    def test_wide_beam_regression(self):
        decoder = self._decoder_constructor(
            self._decoder_symbols,
            k=2,
        )
        logits = np.asarray([
            [-0.1, -8.0, -80.0, -2.0],
            [-0.0, -5.0, -80.0, -80.0],
        ])

        boh = decoder(logits, max_unnormalization=np.inf)
        a_hyp = [hyp for hyp in boh if hyp.transcript == 'a']
        assert(len(a_hyp) == 1)  # this is really a plain assert. There SHALL NOT be multiple hypotheses of the same text
        a_hyp = a_hyp[0]
        self.assertEqual(a_hyp.vis_sc, np.logaddexp(-0.1, -2.0))


class CTCPrefixLogRawNumpyDecoderLMTests(CTCDecodingWithLMTests, unittest.TestCase):
    def setUp(self):
        self._decoder_symbols = ['a', 'b', 'c', BLANK_SYMBOL]
        self._decoder_constructor = CTCPrefixLogRawNumpyDecoder


class FindNewPrefixesTests(unittest.TestCase):
    def setUp(self):
        self.letters = ['a', 'b', 'c', BLANK_SYMBOL]
        self.blank_ind = 3

    def test_old_carry_over(self):
        A_prev = ['aaa', 'aab', 'aac']
        l_last = np.asarray([0, 1, 2])
        best_inds = (np.asarray([0, 1, 2]), np.asarray([3, 3, 3]))

        A_new, l_last_new = find_new_prefixes(l_last, best_inds, A_prev, self.letters, self.blank_ind)

        self.assertEqual(A_new, A_prev)
        self.assertEqual(set(l_last_new.tolist()), set(l_last.tolist()))

    def test_all_new(self):
        A_prev = ['aaa', 'aab', 'aac']
        l_last = np.asarray([0, 1, 2])
        best_inds = (np.asarray([0, 1, 2]), np.asarray([1, 1, 1]))
        A_exp = ['aaab', 'aabb', 'aacb']
        l_last_exp = np.asarray([1, 1, 1])

        A_new, l_last_new = find_new_prefixes(l_last, best_inds, A_prev, self.letters, self.blank_ind)

        self.assertEqual(A_new, A_exp)
        self.assertEqual(set(l_last_new.tolist()), set(l_last_exp.tolist()))

    def test_all_mixed(self):
        A_prev = ['aaa', 'aab', 'aac']
        l_last = np.asarray([0, 1, 2])
        best_inds = (np.asarray([0, 1, 2]), np.asarray([1, 3, 0]))
        A_exp = ['aaab', 'aab', 'aaca']
        l_last_exp = np.asarray([1, 1, 0])

        A_new, l_last_new = find_new_prefixes(l_last, best_inds, A_prev, self.letters, self.blank_ind)

        self.assertEqual(set(A_new), set(A_exp))
        self.assertEqual(set(l_last_new.tolist()), set(l_last_exp.tolist()))

    def test_regression1(self):
        A_prev = ['b', 'a']
        l_last = np.asarray([1, 0])
        best_inds = (np.asarray([1, 1]), np.asarray([3, 1]))
        A_exp = ['ab', 'a']
        l_last_exp = np.asarray([1, 0])

        A_new, l_last_new = find_new_prefixes(l_last, best_inds, A_prev, self.letters, self.blank_ind)

        self.assertEqual(set(A_new), set(A_exp))
        self.assertEqual(set(l_last_new.tolist()), set(l_last_exp.tolist()))


class UpdateLMThingsTests(unittest.TestCase):
    def test_shuffling_partial_update(self):
        h_prev = np.asarray([11, 12])
        lm_preds = np.asarray([[1, 2], [3, 4]])
        best_inds = np.asarray([1, 0]), np.asarray([1, 2])

        class Object:
            pass

        lm = Object()
        lm.advance_h0 = lambda _, h: 2*h
        lm.log_probs = lambda _: np.asarray([5, 6])

        expected_h = np.asarray([24, 11])
        expected_preds = np.asarray([[5, 6], [1, 2]])

        h_new, lm_pred_new = update_lm_things(lm, h_prev, lm_preds, best_inds, blank_ind=2)
        self.assertTrue(np.array_equal(h_new, expected_h))
        self.assertTrue(np.array_equal(lm_pred_new, expected_preds))


class HelpersTests(unittest.TestCase):
    def test_picking_old(self):
        best_inds = np.asarray([0, 1, 2]), np.asarray([3, 2, 3])
        picks = get_old_prefixes_positions(best_inds, 3)
        self.assertEqual(picks, [0, 2])

    def test_picking_new(self):
        best_inds = np.asarray([0, 1, 2]), np.asarray([3, 2, 3])
        picks = get_new_prefixes_positions(best_inds, 3)
        self.assertEqual(picks, [1])
