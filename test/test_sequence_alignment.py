import unittest
from pero_ocr.sequence_alignment import levenshtein_distance
from pero_ocr.sequence_alignment import levenshtein_alignment
from pero_ocr.sequence_alignment import levenshtein_alignment_path
from pero_ocr.sequence_alignment import levenshtein_distance_substring
from pero_ocr.sequence_alignment import levenshtein_alignment_substring


class TestLevenshteinDistance(unittest.TestCase):
    def test_trivial_match(self):
        a = [1]
        b = [1]
        self.assertEqual(levenshtein_distance(a, b), 0)

    def test_trivial_substitution(self):
        a = [1]
        b = [2]
        self.assertEqual(levenshtein_distance(a, b), 1)

    def test_trivial_insertion(self):
        a = [1]
        b = [2, 1]
        self.assertEqual(levenshtein_distance(a, b), 1)

    def test_trivial_deletion(self):
        a = [1, 2]
        b = [1]
        self.assertEqual(levenshtein_distance(a, b), 1)

    def test_inner_replacement(self):
        a = [1, 2, 3]
        b = [1, -1, -2, 3]
        self.assertEqual(levenshtein_distance(a, b), 2)

    def test_inner_replacement_rev(self):
        a = [1, 2, 3]
        b = [1, -1, -2, 3]
        self.assertEqual(levenshtein_distance(b, a), 2)

    def test_deletion_only(self):
        a = [1, 2, 3]
        b = []
        self.assertEqual(levenshtein_distance(a, b), 3)

    def test_insertion_only(self):
        a = []
        b = [1, 2, 3]
        self.assertEqual(levenshtein_distance(a, b), 3)


class TestLevenshteinAlignment(unittest.TestCase):
    def test_trivial_match(self):
        a = [1]
        b = [1]
        self.assertEqual(levenshtein_alignment(a, b), [(1, 1)])

    def test_trivial_substitution(self):
        a = [1]
        b = [2]
        self.assertEqual(levenshtein_alignment(a, b), [(1, 2)])

    def test_trivial_insertion(self):
        a = [1]
        b = [2, 1]
        self.assertEqual(levenshtein_alignment(a, b), [(None, 2), (1, 1)])

    def test_trivial_deletion(self):
        a = [1, 2]
        b = [1]
        self.assertEqual(levenshtein_alignment(a, b), [(1, 1), (2, None)])

    def test_inner_replacement(self):
        a = [1, 2, 3]
        b = [1, -1, -2, 3]

        self.assertTrue(
            levenshtein_alignment(a, b) in [
                [(1, 1), (2, -1), (None, -2), (3, 3)],
                [(1, 1), (None, -1), (2, -2), (3, 3)],
            ]
        )

    def test_inner_replacement_rev(self):
        a = [1, -1, -2, 3]
        b = [1, 2, 3]

        self.assertTrue(
            levenshtein_alignment(a, b) in [
                [(1, 1), (-1, None), (-2, 2), (3, 3)],
                [(1, 1), (-1, 2), (-2, None), (3, 3)],
            ]
        )

    def test_deletion_only(self):
        a = [1, 2, 3]
        b = []
        self.assertEqual(levenshtein_alignment(a, b), [(1, None), (2, None), (3, None)])

    def test_insertion_only(self):
        a = []
        b = [1, 2, 3]
        self.assertEqual(levenshtein_alignment(a, b), [(None, 1), (None, 2), (None, 3)])

    def test_alignment_to_eps(self):
        a = [1, None, 3]
        b = [1, 2, 3]
        self.assertEqual(levenshtein_alignment(a, b), [(1, 1), (None, 2), (3, 3)])

    def test_alignment_to_eps_rev(self):
        a = [1, 2, 3]
        b = [1, None, 3]
        self.assertEqual(levenshtein_alignment(a, b), [(1, 1), (2, None), (3, 3)])


class TestLevenshteinAlignmentPath(unittest.TestCase):
    def test_trivial_match(self):
        a = [1]
        b = [1]
        self.assertEqual(levenshtein_alignment_path(a, b), [0])

    def test_trivial_substitution(self):
        a = [1]
        b = [2]
        self.assertEqual(levenshtein_alignment_path(a, b), [0])

    def test_trivial_insertion(self):
        a = [1]
        b = [2, 1]
        self.assertEqual(levenshtein_alignment_path(a, b), [-1, 0])

    def test_trivial_deletion(self):
        a = [1, 2]
        b = [1]
        self.assertEqual(levenshtein_alignment_path(a, b), [0, 1])

    def test_inner_replacement(self):
        a = [1, 2, 3]
        b = [1, -1, -2, 3]

        self.assertTrue(
            levenshtein_alignment_path(a, b) in [
                [0, 0, -1, 0],
                [0, -1, 0, 0],
            ]
        )

    def test_inner_replacement_rev(self):
        a = [1, -1, -2, 3]
        b = [1, 2, 3]

        self.assertTrue(
            levenshtein_alignment_path(a, b) in [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        )

    def test_deletion_only(self):
        a = [1, 2, 3]
        b = []
        self.assertEqual(levenshtein_alignment_path(a, b), [1, 1, 1])

    def test_insertion_only(self):
        a = []
        b = [1, 2, 3]
        self.assertEqual(levenshtein_alignment_path(a, b), [-1, -1, -1])

    def test_alignment_to_eps(self):
        a = [1, None, 3]
        b = [1, 2, 3]
        self.assertEqual(levenshtein_alignment_path(a, b), [0, 0, 0])

    def test_alignment_to_eps_rev(self):
        a = [1, 2, 3]
        b = [1, None, 3]
        self.assertEqual(levenshtein_alignment_path(a, b), [0, 0, 0])


class TestLevenshteinDistanceSubstring(unittest.TestCase):
    def test_trivial_match(self):
        a = [1]
        b = [1]
        self.assertEqual(levenshtein_distance_substring(a, b), 0)

    def test_trivial_substitution(self):
        a = [1]
        b = [2]
        self.assertEqual(levenshtein_distance_substring(a, b), 1)

    def test_trivial_insertion(self):
        a = [1]
        b = [2, 1]
        self.assertEqual(levenshtein_distance_substring(a, b), 0)

    def test_trivial_deletion(self):
        a = [1, 2]
        b = [1]
        self.assertEqual(levenshtein_distance_substring(a, b), 0)

    def test_inner_replacement(self):
        a = [1, 2, 3]
        b = [1, -1, -2, 3]
        self.assertEqual(levenshtein_distance_substring(a, b), 2)

    def test_inner_replacement_rev(self):
        a = [1, 2, 3]
        b = [1, -1, -2, 3]
        self.assertEqual(levenshtein_distance_substring(b, a), 2)

    def test_deletion_only(self):
        a = [1, 2, 3]
        b = []
        self.assertEqual(levenshtein_distance_substring(a, b), 0)

    def test_insertion_only(self):
        a = []
        b = [1, 2, 3]
        self.assertEqual(levenshtein_distance_substring(a, b), 0)

    def test_substitution_in_the_middle(self):
        a = [1, 2, 3]
        b = [2]
        self.assertEqual(levenshtein_distance_substring(a, b), 0)

    def test_false_start(self):
        a = [1, 2, -1, 1, 2, 3]
        b = [1, 2, 3]
        self.assertEqual(levenshtein_distance_substring(a, b), 0)

    # questionable behaviour -- symmetricity?
    def test_inevitable_error(self):
        a = [1, -1]
        b = [1, 2, 3]
        self.assertEqual(levenshtein_distance_substring(a, b), 1)


class TestLevenshteinAlignmentSubstring(unittest.TestCase):
    def test_trivial_match(self):
        a = [1]
        b = [1]
        self.assertEqual(levenshtein_alignment_substring(a, b), [(1, 1)])

    def test_trivial_substitution(self):
        a = [1]
        b = [2]
        self.assertEqual(levenshtein_alignment_substring(a, b), [(1, 2)])

    def test_trivial_insertion(self):
        a = [1]
        b = [2, 1]
        self.assertEqual(levenshtein_alignment_substring(a, b), [(None, 2), (1, 1)])

    def test_trivial_deletion(self):
        a = [1, 2]
        b = [1]
        self.assertEqual(levenshtein_alignment_substring(a, b), [(1, 1), (2, None)])

    def test_inner_replacement(self):
        a = [1, 2, 3]
        b = [1, -1, -2, 3]

        self.assertTrue(
            levenshtein_alignment_substring(a, b) in [
                [(1, 1), (2, -1), (None, -2), (3, 3)],
                [(1, 1), (None, -1), (2, -2), (3, 3)],
            ]
        )

    def test_inner_replacement_rev(self):
        a = [1, -1, -2, 3]
        b = [1, 2, 3]

        self.assertTrue(
            levenshtein_alignment_substring(a, b) in [
                [(1, 1), (-1, None), (-2, 2), (3, 3)],
                [(1, 1), (-1, 2), (-2, None), (3, 3)],
            ]
        )

    def test_deletion_only(self):
        a = [1, 2, 3]
        b = []
        self.assertEqual(levenshtein_alignment_substring(a, b), [(1, None), (2, None), (3, None)])

    def test_insertion_only(self):
        a = []
        b = [1, 2, 3]
        self.assertEqual(levenshtein_alignment_substring(a, b), [(None, 1), (None, 2), (None, 3)])

    def test_alignment_to_eps(self):
        a = [1, None, 3]
        b = [1, 2, 3]
        self.assertEqual(levenshtein_alignment_substring(a, b), [(1, 1), (None, 2), (3, 3)])

    def test_alignment_to_eps_rev(self):
        a = [1, 2, 3]
        b = [1, None, 3]
        self.assertEqual(levenshtein_alignment_substring(a, b), [(1, 1), (2, None), (3, 3)])

    def test_alignment_simple_prefix(self):
        a = [1, 1, 1, 2, 3]
        b = [1, 2, 3]
        self.assertEqual(levenshtein_alignment_substring(a, b), [(1, None), (1, None), (1, 1), (2, 2), (3, 3)])

    def test_alignment_complex_prefix(self):
        a = [1, 2, 1, 2, 3]
        b = [1, 2, 3]
        self.assertEqual(levenshtein_alignment_substring(a, b), [(1, None), (2, None), (1, 1), (2, 2), (3, 3)])

    def test_alignment_simple_suffix(self):
        a = [1, 2, 3, 3, 3]
        b = [1, 2, 3]
        self.assertEqual(levenshtein_alignment_substring(a, b), [(1, 1), (2, 2), (3, 3), (3, None), (3, None)])

    def test_alignment_complex_suffix(self):
        a = [1, 2, 3, 2, 3]
        b = [1, 2, 3]
        self.assertEqual(levenshtein_alignment_substring(a, b), [(1, 1), (2, 2), (3, 3), (2, None), (3, None)])

    def test_alignment_repeating_sequence(self):
        a = [1, 2, 3, 1, 2, 3]
        b = [1, 2, 3]
        self.assertTrue(
            levenshtein_alignment_substring(a, b) in [
                [(1, 1), (2, 2), (3, 3), (1, None), (2, None), (3, None)],
                [(1, None), (2, None), (3, None), (1, 1), (2, 2), (3, 3)],
            ]
        )


