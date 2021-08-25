from unittest import TestCase
import math

from pero_ocr.decoding.bag_of_hypotheses import BagOfHypotheses
from pero_ocr.decoding.confusion_networks import produce_cn_from_boh
from pero_ocr.decoding.confusion_networks import get_pivot
from pero_ocr.decoding.confusion_networks import add_hypothese
from pero_ocr.decoding.confusion_networks import normalize_cn
from pero_ocr.decoding.confusion_networks import best_cn_path
from pero_ocr.decoding.confusion_networks import sorted_cn_paths


class TestGettingPivot(TestCase):
    def test_empty_cn(self):
        pivot = get_pivot([])
        self.assertEqual(pivot, [])

    def test_single_hypothese(self):
        pivot = get_pivot([{'a': 1.0}, {'b': 1.0}, {'c': 1.0}])
        self.assertEqual(pivot, ['a', 'b', 'c'])

    def test_multiple_hyps(self):
        pivot = get_pivot([{'a': 0.6, 'x': 0.4}, {'b': 1.0}, {'c': 1.0, 'z': 3.0}])
        self.assertEqual(pivot, ['a', 'b', 'z'])


class TestHypotheseNormalization(TestCase):
    def test_single_hypothese(self):
        cn = [{'a': 0.24}, {'b': 0.24}, {'c': 0.24}]
        normalized_cn = normalize_cn(cn)
        self.assertEqual(normalized_cn, [{'a': 1.0}, {'b': 1.0}, {'c': 1.0}])

    def test_multiple_hyps(self):
        cn = [{'a': 0.6, 'x': 0.4}, {'b': 1.0}, {'c': 1.0, 'z': 3.0}]
        normalized_cn = normalize_cn(cn)
        self.assertEqual(normalized_cn, [{'a': 0.6, 'x': 0.4}, {'b': 1.0}, {'c': 0.25, 'z': 0.75}])


class TestAddingHypothese(TestCase):
    def test_empty_cn(self):
        cn = []
        hyp = 'abc'
        new_cn = add_hypothese(cn, hyp, 0.24)
        self.assertEqual(new_cn, [{'a': 0.24}, {'b': 0.24}, {'c': 0.24}])

    def test_empty_cn_empty_hypothese(self):
        cn = []
        hyp = ''
        new_cn = add_hypothese(cn, hyp, 0.24)
        self.assertEqual(new_cn, [])

    def test_matches_only(self):
        cn = [{'a': 0.24}, {'b': 0.24}, {'c': 0.24}]
        hyp = 'abc'
        new_cn = add_hypothese(cn, hyp, 0.24)
        self.assertEqual(new_cn, [{'a': 0.48}, {'b': 0.48}, {'c': 0.48}])

    def test_matches_and_substitutions(self):
        cn = [{'a': 0.24}, {'x': 0.24}, {'c': 0.24}]
        hyp = 'abc'
        new_cn = add_hypothese(cn, hyp, 0.24)
        self.assertEqual(new_cn, [{'a': 0.48}, {'b': 0.24, 'x': 0.24}, {'c': 0.48}])

    def test_deletion(self):
        cn = [{'a': 0.24}, {'b': 0.24}]
        hyp = 'a'
        new_cn = add_hypothese(cn, hyp, 0.24)
        self.assertEqual(new_cn, [{'a': 0.48}, {'b': 0.24, None: 0.24}])

    def test_end_insertion(self):
        cn = [{'a': 0.24}, {'b': 0.24}]
        hyp = 'abc'
        new_cn = add_hypothese(cn, hyp, 0.24)
        self.assertEqual(new_cn, [{'a': 0.48}, {'b': 0.48}, {None: 0.24, 'c': 0.24}])

    def test_middle_insertion(self):
        cn = [{'a': 0.24}, {'c': 0.24}]
        hyp = 'abc'
        new_cn = add_hypothese(cn, hyp, 0.24)
        self.assertEqual(new_cn, [{'a': 0.48}, {None: 0.24, 'b': 0.24}, {'c': 0.48}])

    def test_beginning_insertion(self):
        cn = [{'b': 0.24}, {'c': 0.24}]
        hyp = 'abc'
        new_cn = add_hypothese(cn, hyp, 0.24)
        self.assertEqual(new_cn, [{None: 0.24, 'a': 0.24}, {'b': 0.48}, {'c': 0.48}])

    def test_match_epsilon(self):
        cn = [{'a': 0.24, 'x': 0.11}, {None: 0.24, 'y': 0.11}, {'c': 0.24, 'z': 0.11}]
        hyp = 'ac'
        new_cn = add_hypothese(cn, hyp, 0.24)
        self.assertEqual(new_cn, [{'a': 0.48, 'x': 0.11}, {None: 0.48, 'y': 0.11}, {'c': 0.48, 'z': 0.11}])

    def test_match_epsilon_to_more(self):
        cn = [{'a': 0.24, 'x': 0.11}, {None: 0.24, 'y': 0.11}, {None: 0.24, 'y': 0.11}, {'c': 0.24, 'z': 0.11}]
        hyp = 'ac'
        new_cn = add_hypothese(cn, hyp, 0.24)
        self.assertEqual(new_cn, [{'a': 0.48, 'x': 0.11}, {None: 0.48, 'y': 0.11}, {None: 0.48, 'y': 0.11}, {'c': 0.48, 'z': 0.11}])

    def test_match_letter_to_epsilon(self):
        cn = [{'a': 0.35}, {None: 0.24, 'b': 0.11}, {'c': 0.35}]
        hyp = 'axc'
        new_cn = add_hypothese(cn, hyp, 0.15)
        self.assertEqual(new_cn, [{'a': 0.50}, {None: 0.24, 'b': 0.11, 'x': 0.15}, {'c': 0.50}])

    def test_extending_before_epsilon(self):
        cn = [{'a': 0.12, None: 0.12}, {'b': 0.24}, {'c': 0.24}]
        hyp = 'xab'
        new_cn = add_hypothese(cn, hyp, 0.12)
        self.assertEqual(new_cn, [{None: 0.24, 'x': 0.12}, {'a': 0.24, None: 0.12}, {'b': 0.36}, {'c': 0.24, None: 0.12}])


class TestConfusionNetworkConstruction(TestCase):
    def test_single_hypothesis(self):
        boh = BagOfHypotheses()
        boh.add('abc', 23.0, 2.0)
        cn = produce_cn_from_boh(boh)

        self.assertEqual(cn, [{'a': 1.0}, {'b': 1.0}, {'c': 1.0}])

    def test_two_hyps(self):
        boh = BagOfHypotheses()
        boh.add('abc', 0.5)
        boh.add('ac', 0.5)
        cn = produce_cn_from_boh(boh)

        self.assertEqual(cn, [{'a': 1.0}, {'b': 0.5, None: 0.5}, {'c': 1.0}])

    def test_two_hyps_different_weight(self):
        boh = BagOfHypotheses()
        boh.add('abc', 0.0)
        boh.add('ac', 1.0)
        cn = produce_cn_from_boh(boh)

        first_prob = math.exp(0.0)
        second_prob = math.exp(1.0)
        total_prob = first_prob + second_prob

        self.assertEqual(cn, [{'a': 1.0}, {'b': first_prob/total_prob, None: second_prob/total_prob}, {'c': 1.0}])

    def test_lm_weight(self):
        boh = BagOfHypotheses()
        boh.add('abc', 0.0, 2.0)
        boh.add('ac', 1.0, -1.0)
        cn = produce_cn_from_boh(boh, lm_weight=2.0)

        first_prob = math.exp(0.0 + 2.0*2.0)
        second_prob = math.exp(1.0 + (-1.0)*2.0)
        total_prob = first_prob + second_prob

        self.assertEqual(cn, [{'a': 1.0}, {'b': first_prob/total_prob, None: second_prob/total_prob}, {'c': 1.0}])


class TestConfusionNetworkGreedyDecoder(TestCase):
    def test_empty_cn(self):
        cn = []
        self.assertEqual(best_cn_path(cn), '')

    def test_single_hyp_cn(self):
        cn = [{'a': 1.0}, {'b': 1.0}, {'c': 1.0}]
        self.assertEqual(best_cn_path(cn), 'abc')

    def test_two_hyp_cn(self):
        cn = [{'a': 1.0}, {'b': 0.3, 'y': 0.7}, {'c': 1.0}]
        self.assertEqual(best_cn_path(cn), 'ayc')

    def test_epsilon_removal(self):
        cn = [{'a': 1.0}, {'b': 0.3, None: 0.7}, {'c': 1.0}]
        self.assertEqual(best_cn_path(cn), 'ac')


class TestConfusionNetworkSortedPaths(TestCase):
    def test_empty_cn(self):
        cn = []

        self.assertEqual(sorted_cn_paths(cn), [])

    def test_single_path_cn(self):
        cn = [{'a': 1.0}, {'b': 1.0}]

        self.assertEqual(sorted_cn_paths(cn), [('ab', 1.0)])

    def test_two_path_cn(self):
        cn = [{'a': 0.75, 'b': 0.25}, {'b': 1.0}]

        self.assertEqual(sorted_cn_paths(cn), [('ab', 0.75), ('bb', 0.25)])

    def test_two_path_reverse_probs(self):
        cn = [{'a': 0.25, 'b': 0.75}, {'b': 1.0}]

        self.assertEqual(sorted_cn_paths(cn), [('bb', 0.75), ('ab', 0.25)])

    def test_two_choice_points(self):
        cn = [{'a': 0.25, 'b': 0.75}, {'c': 0.9, 'd': 0.1}]
        paths = [
            ('bc', 0.75*0.9),
            ('ac', 0.25*0.9),
            ('bd', 0.75*0.1),
            ('ad', 0.25*0.1),
        ]

        self.assertEqual(sorted_cn_paths(cn), paths)

    def test_epsilon(self):
        cn = [{'a': 0.75, None: 0.25}, {'b': 1.0}]

        self.assertEqual(sorted_cn_paths(cn), [('ab', 0.75), ('b', 0.25)])
