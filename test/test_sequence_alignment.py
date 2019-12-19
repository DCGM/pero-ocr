import unittest
from pero_ocr.sequence_alignment import levenshtein_distance
from pero_ocr.sequence_alignment import levenshtein_alignment
from pero_ocr.sequence_alignment import levenshtein_alignment_path


class TestLevenshteinDistance(unittest.TestCase):
    def test_trivial_match(self):
        a = ['a']
        b = ['a']
        self.assertEqual(levenshtein_distance(a, b), 0)

    def test_trivial_substitution(self):
        a = ['a']
        b = ['b']
        self.assertEqual(levenshtein_distance(a, b), 1)

    def test_trivial_insertion(self):
        a = ['a']
        b = ['b', 'a']
        self.assertEqual(levenshtein_distance(a, b), 1)

    def test_trivial_deletion(self):
        a = ['a', 'b']
        b = ['a']
        self.assertEqual(levenshtein_distance(a, b), 1)

    def test_inner_replacement(self):
        a = ['a', 'b', 'c']
        b = ['a', 'x', 'y', 'c']
        self.assertEqual(levenshtein_distance(a, b), 2)

    def test_inner_replacement_rev(self):
        a = ['a', 'b', 'c']
        b = ['a', 'x', 'y', 'c']
        self.assertEqual(levenshtein_distance(b, a), 2)

    def test_deletion_only(self):
        a = ['a', 'b', 'c']
        b = []
        self.assertEqual(levenshtein_distance(a, b), 3)

    def test_insertion_only(self):
        a = []
        b = ['a', 'b', 'c']
        self.assertEqual(levenshtein_distance(a, b), 3)


class TestLevenshteinAlignment(unittest.TestCase):
    def test_trivial_match(self):
        a = ['a']
        b = ['a']
        self.assertEqual(levenshtein_alignment(a, b), [('a', 'a')])

    def test_trivial_substitution(self):
        a = ['a']
        b = ['b']
        self.assertEqual(levenshtein_alignment(a, b), [('a', 'b')])

    def test_trivial_insertion(self):
        a = ['a']
        b = ['b', 'a']
        self.assertEqual(levenshtein_alignment(a, b), [(None, 'b'), ('a', 'a')])

    def test_trivial_deletion(self):
        a = ['a', 'b']
        b = ['a']
        self.assertEqual(levenshtein_alignment(a, b), [('a', 'a'), ('b', None)])

    def test_inner_replacement(self):
        a = ['a', 'b', 'c']
        b = ['a', 'x', 'y', 'c']

        self.assertTrue(
            levenshtein_alignment(a, b) in [
                [('a', 'a'), ('b', 'x'), (None, 'y'), ('c', 'c')],
                [('a', 'a'), (None, 'x'), ('b', 'y'), ('c', 'c')],
            ]
        )

    def test_inner_replacement_rev(self):
        a = ['a', 'x', 'y', 'c']
        b = ['a', 'b', 'c']

        self.assertTrue(
            levenshtein_alignment(a, b) in [
                [('a', 'a'), ('x', None), ('y', 'b'), ('c', 'c')],
                [('a', 'a'), ('x', 'b'), ('y', None), ('c', 'c')],
            ]
        )

    def test_deletion_only(self):
        a = ['a', 'b', 'c']
        b = []
        self.assertEqual(levenshtein_alignment(a, b), [('a', None), ('b', None), ('c', None)])

    def test_insertion_only(self):
        a = []
        b = ['a', 'b', 'c']
        self.assertEqual(levenshtein_alignment(a, b), [(None, 'a'), (None, 'b'), (None, 'c')])

    def test_alignment_to_eps(self):
        a = ['a', None, 'c']
        b = ['a', 'b', 'c']
        self.assertEqual(levenshtein_alignment(a, b), [('a', 'a'), (None, 'b'), ('c', 'c')])

    def test_alignment_to_eps_rev(self):
        a = ['a', 'b', 'c']
        b = ['a', None, 'c']
        self.assertEqual(levenshtein_alignment(a, b), [('a', 'a'), ('b', None), ('c', 'c')])


class TestLevenshteinAlignmentPath(unittest.TestCase):
    def test_trivial_match(self):
        a = ['a']
        b = ['a']
        self.assertEqual(levenshtein_alignment_path(a, b), [0])

    def test_trivial_substitution(self):
        a = ['a']
        b = ['b']
        self.assertEqual(levenshtein_alignment_path(a, b), [0])

    def test_trivial_insertion(self):
        a = ['a']
        b = ['b', 'a']
        self.assertEqual(levenshtein_alignment_path(a, b), [-1, 0])

    def test_trivial_deletion(self):
        a = ['a', 'b']
        b = ['a']
        self.assertEqual(levenshtein_alignment_path(a, b), [0, 1])

    def test_inner_replacement(self):
        a = ['a', 'b', 'c']
        b = ['a', 'x', 'y', 'c']

        self.assertTrue(
            levenshtein_alignment_path(a, b) in [
                [0, 0, -1, 0],
                [0, -1, 0, 0],
            ]
        )

    def test_inner_replacement_rev(self):
        a = ['a', 'x', 'y', 'c']
        b = ['a', 'b', 'c']

        self.assertTrue(
            levenshtein_alignment_path(a, b) in [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        )

    def test_deletion_only(self):
        a = ['a', 'b', 'c']
        b = []
        self.assertEqual(levenshtein_alignment_path(a, b), [1, 1, 1])

    def test_insertion_only(self):
        a = []
        b = ['a', 'b', 'c']
        self.assertEqual(levenshtein_alignment_path(a, b), [-1, -1, -1])

    def test_alignment_to_eps(self):
        a = ['a', None, 'c']
        b = ['a', 'b', 'c']
        self.assertEqual(levenshtein_alignment_path(a, b), [0, 0, 0])

    def test_alignment_to_eps_rev(self):
        a = ['a', 'b', 'c']
        b = ['a', None, 'c']
        self.assertEqual(levenshtein_alignment_path(a, b), [0, 0, 0])
