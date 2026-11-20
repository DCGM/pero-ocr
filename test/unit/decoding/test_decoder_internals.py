"""Internals of the CTC prefix decoder (pero_ocr.decoding.decoders): finding new prefixes, updating LM states,
picking old and new prefixes. The decoders as black boxes are tested in test/engines/test_decoders.py."""
import unittest

import numpy as np

from pero_ocr.decoding.decoders import BLANK_SYMBOL, find_new_prefixes, get_new_prefixes_positions, \
    get_old_prefixes_positions, update_lm_things


def make_unique(array):
    return set(tuple(x) for i, x in enumerate(array) if array.index(x) == i)


class FindNewPrefixesTests(unittest.TestCase):
    def setUp(self):
        self.letters = ['a', 'b', 'c', BLANK_SYMBOL]
        self.blank_ind = 3

    def test_old_carry_over(self):
        A_prev = [[0, 0, 0], [0, 0, 1], [0, 0, 2]]
        l_last = np.asarray([0, 1, 2])
        best_inds = (np.asarray([0, 1, 2]), np.asarray([3, 3, 3]))

        A_new, l_last_new = find_new_prefixes(l_last, best_inds, A_prev, self.blank_ind)

        self.assertEqual(A_new, A_prev)
        self.assertEqual(set(l_last_new.tolist()), set(l_last.tolist()))

    def test_all_new(self):
        A_prev = [[0, 0, 0], [0, 0, 1], [0, 0, 2]]
        l_last = np.asarray([0, 1, 2])
        best_inds = (np.asarray([0, 1, 2]), np.asarray([1, 1, 1]))
        A_exp = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 2, 1]]
        l_last_exp = np.asarray([1, 1, 1])

        A_new, l_last_new = find_new_prefixes(l_last, best_inds, A_prev, self.blank_ind)

        self.assertEqual(A_new, A_exp)
        self.assertEqual(set(l_last_new.tolist()), set(l_last_exp.tolist()))

    def test_all_mixed(self):
        A_prev = [[0, 0, 0], [0, 0, 1], [0, 0, 2]]
        l_last = np.asarray([0, 1, 2])
        best_inds = (np.asarray([0, 1, 2]), np.asarray([1, 3, 0]))
        A_exp = [[0, 0, 0, 1], [0, 0, 1], [0, 0, 2, 0]]
        l_last_exp = np.asarray([1, 1, 0])

        A_new, l_last_new = find_new_prefixes(l_last, best_inds, A_prev, self.blank_ind)

        self.assertEqual(make_unique(A_new), make_unique(A_exp))
        self.assertEqual(set(l_last_new.tolist()), set(l_last_exp.tolist()))

    def test_regression1(self):
        A_prev = [[1], [0]]
        l_last = np.asarray([1, 0])
        best_inds = (np.asarray([1, 1]), np.asarray([3, 1]))
        A_exp = [[0, 1], [0]]
        l_last_exp = np.asarray([1, 0])

        A_new, l_last_new = find_new_prefixes(l_last, best_inds, A_prev, self.blank_ind)

        self.assertEqual(make_unique(A_new), make_unique(A_exp))
        self.assertEqual(set(l_last_new.tolist()), set(l_last_exp.tolist()))


class FindNewPrefixesSymbolicTests(unittest.TestCase):
    def setUp(self):
        self.letters = [0, 1, 2, BLANK_SYMBOL]
        self.blank_ind = 3

    def test_old_carry_over(self):
        A_prev = [[0, 0, 0,], [0, 0, 1], [0, 0, 2]]
        l_last = np.asarray([0, 1, 2])
        best_inds = (np.asarray([0, 1, 2]), np.asarray([3, 3, 3]))

        A_new, l_last_new = find_new_prefixes(l_last, best_inds, A_prev, self.blank_ind)

        self.assertEqual(A_new, A_prev)
        self.assertEqual(set(l_last_new.tolist()), set(l_last.tolist()))



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
