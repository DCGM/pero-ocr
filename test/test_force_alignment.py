import unittest
import numpy as np

from pero_ocr.force_alignment import hmm_trans_from_string, complete_state_seq
from pero_ocr.force_alignment import initial_cost, final_cost
from pero_ocr.force_alignment import backtrack, expand_logits
from pero_ocr.force_alignment import viterbi_align, force_align


class TestHmmTransitionCreation(unittest.TestCase):
    def test_trivial(self):
        desired = np.asarray([
            [0.0,    0.0,    np.inf],
            [np.inf, 0.0,    0.0],
            [np.inf, np.inf, 0.0],
        ])

        np.testing.assert_array_equal(hmm_trans_from_string([1]), desired)

    def test_two_letter_different(self):
        desired = np.full((5, 5), np.inf)
        desired[0, 0] = 0.0
        desired[0, 1] = 0.0
        desired[1, 1] = 0.0
        desired[1, 2] = 0.0
        desired[1, 3] = 0.0
        desired[2, 2] = 0.0
        desired[2, 3] = 0.0
        desired[3, 3] = 0.0
        desired[3, 4] = 0.0
        desired[4, 4] = 0.0
        np.testing.assert_array_equal(hmm_trans_from_string([1, 2]), desired)

    def test_two_letter_same(self):
        desired = np.full((5, 5), np.inf)
        desired[0, 0] = 0.0
        desired[0, 1] = 0.0
        desired[1, 1] = 0.0
        desired[1, 2] = 0.0
        desired[2, 2] = 0.0
        desired[2, 3] = 0.0
        desired[3, 3] = 0.0
        desired[3, 4] = 0.0
        desired[4, 4] = 0.0
        np.testing.assert_array_equal(hmm_trans_from_string([1, 1]), desired)

    def test_declines_empty(self):
        self.assertRaises(ValueError, hmm_trans_from_string, [])


class TestSymbolSequenceCompletion(unittest.TestCase):
    def test_trivial(self):
        char_inds_seq = [0, 1, 0]
        positions_seq = [-1, 0, -1]

        indexes, positions = complete_state_seq([1], 0)

        self.assertTrue((indexes == char_inds_seq).all())
        self.assertTrue((positions == positions_seq).all())

    def test_two_letter(self):
        char_inds_seq = [0, 1, 0, 2, 0]
        positions_seq = [-1, 0, -1, 1, -1]

        indexes, positions = complete_state_seq([1, 2], 0)

        self.assertTrue((indexes == char_inds_seq).all())
        self.assertTrue((positions == positions_seq).all())

    def test_checks_for_blank_in_nonblanks(self):
        self.assertRaises(ValueError, complete_state_seq, [1, 0, 2], 0)


class TestInitialCost(unittest.TestCase):
    def test_trivial(self):
        desired = np.asarray([0.0, 0.0, np.inf])
        np.testing.assert_array_equal(initial_cost(3), desired)

    def test_5_state(self):
        desired = np.asarray([0.0, 0.0, np.inf, np.inf, np.inf])
        np.testing.assert_array_equal(initial_cost(5), desired)

    def test_reject_less_2(self):
        self.assertRaises(ValueError, initial_cost, 1)


class TestFinalCost(unittest.TestCase):
    def test_trivial(self):
        desired = np.asarray([np.inf, 0.0, 0.0])
        np.testing.assert_array_equal(final_cost(3), desired)

    def test_5_state(self):
        desired = np.asarray([np.inf, np.inf, np.inf, 0.0, 0.0])
        np.testing.assert_array_equal(final_cost(5), desired)

    def test_reject_less_2(self):
        self.assertRaises(ValueError, final_cost, 1)


class TestBacktracking(unittest.TestCase):
    def test_trivial(self):
        backpointers = np.asarray([
            [-1, -1],
            [0, 1]
        ], dtype=np.int)

        self.assertEqual(backtrack(backpointers, 0), [0, 0])

    def test_respect_final_state(self):
        backpointers = np.asarray([
            [-1, -1],
            [0, 1]
        ], dtype=np.int)

        self.assertEqual(backtrack(backpointers, 1), [1, 1])

    def test_longer_seq(self):
        backpointers = np.asarray([
            [-1, -1],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 0],
        ], dtype=np.int)

        self.assertEqual(backtrack(backpointers, 1), [0, 0, 1, 0, 1])

    def test_multiple_states(self):
        backpointers = np.asarray([
            [-1, -1, -1],
            [0, 1, 2],
            [1, 1, 0],
            [2, 1, 2],
            [1, 0, 2],
        ], dtype=np.int)

        self.assertEqual(backtrack(backpointers, 1), [0, 0, 2, 0, 1])


class TestLogitsExpansion(unittest.TestCase):
    def test_trivial(self):
        logits_T = np.asarray([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ], dtype=np.int)
        logits = logits_T.T

        np.testing.assert_array_equal(expand_logits(logits, [0, 1, 2]), logits)

    def test_single_repeating(self):
        logits_T = np.asarray([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ], dtype=np.int)
        logits = logits_T.T

        expanded_logits_T = np.asarray([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [2, 2, 2],
            [0, 0, 0],
        ], dtype=np.int)
        expanded_logits = expanded_logits_T.T

        np.testing.assert_array_equal(expand_logits(logits, [0, 1, 0, 2, 0]), expanded_logits)


class TestViterbiAlignment(unittest.TestCase):
    def test_trivial(self):
        neg_logits = np.asarray([
            [0.0, 10.0],
            [10.0, 0.0]
        ])

        A = np.asarray([
            [0.0, 0.0],
            [np.inf, 0.0]
        ])

        self.assertEqual(viterbi_align(neg_logits, A), [0, 1])

    def test_trivial_two_blank(self):
        neg_logits = np.asarray([
            [0.0, 10.0],
            [0.0, 10.0],
            [10.0, 0.0]
        ])

        A = np.asarray([
            [0.0, 0.0],
            [np.inf, 0.0]
        ])

        self.assertEqual(viterbi_align(neg_logits, A), [0, 0, 1])

    def test_requires_matching_dimensions(self):
        neg_logits = np.asarray([
            [0.0, 10.0],
            [0.0, 10.0],
            [10.0, 0.0]
        ])

        A = np.asarray([
            [0.0, 0.0, np.inf],
            [np.inf, 0.0, 0.0],
            [np.inf, np.inf, 0.0]
        ])

        self.assertRaises(ValueError, viterbi_align, neg_logits, A)

    def test_reports_impossibility_of_alignmnent(self):
        neg_logits = np.asarray([
            [0.0, np.inf, 0.0],
            [0.0, np.inf, 0.0],
            [0.0, np.inf, 0.0],
        ])

        A = np.asarray([
            [0.0, 0.0, np.inf],
            [np.inf, 0.0, 0.0],
            [np.inf, np.inf, 0.0]
        ])

        self.assertRaises(ValueError, viterbi_align, neg_logits, A)

    def test_single_symbol_multi_blank(self):
        neg_logits = np.asarray([
            [0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0],
            [0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
        ])

        A = np.asarray([
            [0.0, 0.0, np.inf],
            [np.inf, 0.0, 0.0],
            [np.inf, np.inf, 0.0]
        ])

        self.assertEqual(viterbi_align(neg_logits, A), [0, 0, 0, 1, 2, 2])

    def test_multi_frame_symbol(self):
        neg_logits = np.asarray([
            [0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0],
            [10.0, 0.0, 10.0],
            [10.0, 0.0, 10.0],
            [0.0, 10.0, 0.0],
        ])

        A = np.asarray([
            [0.0, 0.0, np.inf],
            [np.inf, 0.0, 0.0],
            [np.inf, np.inf, 0.0]
        ])

        self.assertEqual(viterbi_align(neg_logits, A), [0, 0, 1, 1, 1, 2])

    def test_respect_final_state(self):
        neg_logits = np.asarray([
            [0.0, 10.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 10.0, 0.0],
        ])

        A = np.asarray([
            [0.0, 0.0, np.inf],
            [np.inf, 0.0, 0.0],
            [np.inf, np.inf, 0.0]
        ])

        self.assertEqual(viterbi_align(neg_logits, A), [0, 1, 2])


class TestTopLevelAlignment(unittest.TestCase):
    def test_trivial(self):
        neg_logits = np.asarray([
            [0.0, 10.0],
            [10.0, 0.0]
        ])

        self.assertEqual(force_align(neg_logits, [1], 0), [0, 1])

    def test_single_symbol_multi_blank(self):
        neg_logits = np.asarray([
            [0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0],
            [0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
        ])

        self.assertEqual(force_align(neg_logits, [1], 0), [0, 0, 0, 1, 0, 0])

    def test_multi_symbol_regression(self):
        neg_logits = np.asarray([
            [0.0, 10.0, 10.0],
            [10.0, 10.0, 0.0],
            [5.0, 10.0, 5.0],
            [10.0, 10.0, 0.0],
        ])

        self.assertEqual(force_align(neg_logits, [2, 2], 0), [0, 2, 0, 2])

    def test_skipping_first_regression(self):
        neg_logits = np.asarray([
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 10.0],
        ])

        self.assertEqual(force_align(neg_logits, [1, 2], 0), [1, 2])
