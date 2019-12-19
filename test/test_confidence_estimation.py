import unittest
import numpy as np

from src.confidence_estimation import normalize_logits
from src.confidence_estimation import pick_elements
from src.confidence_estimation import group_elements_by_symbols
from src.confidence_estimation import squeeze
from src.confidence_estimation import get_letter_confidence


class NormalizeLogitsTests(unittest.TestCase):
    def test_single_elem(self):
        logits = np.asarray([[1.0]])
        log_probs = normalize_logits(logits)
        expected = np.asarray([[0.]])
        self.assertTrue(np.allclose(log_probs, expected))

    def test_array_of_single_elems(self):
        logits = np.asarray([[1.0], [-2.0]])
        log_probs = normalize_logits(logits)
        expected = np.asarray([[0], [0.0]])
        self.assertTrue(np.allclose(log_probs, expected))

    def test_binary_no_bias(self):
        logits = np.log(np.asarray([[0.25, 0.75]]))
        log_probs = normalize_logits(logits)
        expected = logits
        self.assertTrue(np.allclose(log_probs, expected))

    def test_array_of_binary_with_bias(self):
        expected = np.log(np.asarray([[0.25, 0.75], [0.33, 0.67], [0.9, 0.1]]))
        logits = expected - np.asarray([4, 7, -12])[:, np.newaxis]
        log_probs = normalize_logits(logits)
        self.assertTrue(np.allclose(log_probs, expected))


class PickElemsTests(unittest.TestCase):
    def test_single_elem(self):
        elems = np.asarray([[1]])
        pick = pick_elements(elems, [0])
        expected = np.asarray([1])
        self.assertTrue(np.array_equal(pick, expected))

    def test_single_frame(self):
        elems = np.asarray([[1, 2]])
        pick = pick_elements(elems, [0])
        expected = np.asarray([1])
        self.assertTrue(np.array_equal(pick, expected))

    def test_multiple_frames(self):
        elems = np.asarray([[1, 2], [3, 4], [5, 6]])
        pick = pick_elements(elems, [0, 1, 1])
        expected = np.asarray([1, 4, 6])
        self.assertTrue(np.array_equal(pick, expected))


class GroupElementsTests(unittest.TestCase):
    def test_single_elem(self):
        elems = np.asarray([1])
        alignment = [0]
        expected = [[1]]
        self.assertEqual(group_elements_by_symbols(elems, alignment), expected)

    def test_array_changing_symbols(self):
        elems = np.asarray([1, 2, 3, 4])
        alignment = [0, 1, 0, 1]
        expected = [[1], [2], [3], [4]]
        self.assertEqual(group_elements_by_symbols(elems, alignment), expected)

    def test_array_grouping(self):
        elems = np.asarray([1, 2, 3, 4])
        alignment = [0, 0, 0, 1]
        expected = [[1, 2, 3], [4]]
        self.assertEqual(group_elements_by_symbols(elems, alignment), expected)


class SqueezeTests(unittest.TestCase):
    def test_single_elem(self):
        sequence = [1]
        self.assertEqual(squeeze(sequence), sequence)

    def test_empty(self):
        sequence = []
        self.assertEqual(squeeze(sequence), sequence)

    def test_switching_seq(self):
        sequence = [1, 2, 1, 3]
        self.assertEqual(squeeze(sequence), sequence)

    def test_squeezing(self):
        sequence = [1, 1, 1, 3]
        self.assertEqual(squeeze(sequence), [1, 3])


class GetLetterConfidenceTests(unittest.TestCase):
    def test_single_elem(self):
        logits = np.log(np.asarray([[0.25, 0.75]]))
        alignment = [1]
        letter_probs = get_letter_confidence(logits, alignment, blank_ind=0)
        expected = [np.log(0.75)]

        self.assertEqual(len(letter_probs), len(expected))
        for p, e in zip(letter_probs, expected):
            self.assertAlmostEqual(p, e)

    def test_multiple_elem(self):
        logits = np.log(np.asarray([[0.25, 0.5, 0.25], [0.2, 0.1, 0.7], [0.05, 0.9, 0.05]]))
        alignment = [1, 2, 1]
        letter_probs = get_letter_confidence(logits, alignment, blank_ind=0)
        expected = [np.log(0.5), np.log(0.7), np.log(0.9)]

        self.assertEqual(len(letter_probs), len(expected))
        for p, e in zip(letter_probs, expected):
            self.assertAlmostEqual(p, e)

    def test_continued_elem(self):
        logits = np.log(np.asarray([[0.25, 0.5, 0.25], [0.2, 0.1, 0.7], [0.05, 0.9, 0.05]]))
        alignment = [1, 2, 2]
        letter_probs = get_letter_confidence(logits, alignment, blank_ind=0)
        expected = [np.log(0.5), np.log(0.7)]

        self.assertEqual(len(letter_probs), len(expected))
        for p, e in zip(letter_probs, expected):
            self.assertAlmostEqual(p, e)

    def test_blanks(self):
        logits = np.log(np.asarray([[0.25, 0.5, 0.25], [0.2, 0.1, 0.7], [0.05, 0.9, 0.05]]))
        alignment = [1, 0, 1]
        letter_probs = get_letter_confidence(logits, alignment, blank_ind=0)
        expected = [np.log(0.5), np.log(0.9)]

        self.assertEqual(len(letter_probs), len(expected))
        for p, e in zip(letter_probs, expected):
            self.assertAlmostEqual(p, e)
