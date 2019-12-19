import unittest
from pero_ocr.error_summary import ErrorsSummary


class SingleReferenceTests(unittest.TestCase):
    def test_empty_ref_match(self):
        ref = ''
        hyp = ''
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.nb_errors, 0)
        self.assertEqual(summary.nb_subs, 0)
        self.assertEqual(summary.nb_inss, 0)
        self.assertEqual(summary.nb_dels, 0)
        self.assertEqual(summary.ref_len, 0)
        self.assertEqual(summary.nb_lines_summarized, 1)

    def test_match(self):
        ref = 'ab'
        hyp = 'ab'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.nb_errors, 0)
        self.assertEqual(summary.nb_subs, 0)
        self.assertEqual(summary.nb_inss, 0)
        self.assertEqual(summary.nb_dels, 0)
        self.assertEqual(summary.ref_len, 2)
        self.assertEqual(summary.nb_lines_summarized, 1)

    def test_deletion(self):
        ref = 'ab'
        hyp = 'a'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.nb_errors, 1)
        self.assertEqual(summary.nb_subs, 0)
        self.assertEqual(summary.nb_inss, 0)
        self.assertEqual(summary.nb_dels, 1)
        self.assertEqual(summary.ref_len, 2)
        self.assertEqual(summary.nb_lines_summarized, 1)

    def test_insertion(self):
        ref = 'ab'
        hyp = 'abc'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.nb_errors, 1)
        self.assertEqual(summary.nb_subs, 0)
        self.assertEqual(summary.nb_inss, 1)
        self.assertEqual(summary.nb_dels, 0)
        self.assertEqual(summary.ref_len, 2)
        self.assertEqual(summary.nb_lines_summarized, 1)

    def test_substitution(self):
        ref = 'ab'
        hyp = 'ac'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.nb_errors, 1)
        self.assertEqual(summary.nb_subs, 1)
        self.assertEqual(summary.nb_inss, 0)
        self.assertEqual(summary.nb_dels, 0)
        self.assertEqual(summary.ref_len, 2)
        self.assertEqual(summary.nb_lines_summarized, 1)


class ErrorAggregationTests(unittest.TestCase):
    def test_single_summary(self):
        partial_1 = ErrorsSummary.from_lists(list('abcd'), list('abb'))
        summary = ErrorsSummary.aggregate([partial_1])
        self.assertEqual(summary.nb_errors, 2)
        self.assertEqual(summary.nb_subs, 1)
        self.assertEqual(summary.nb_inss, 0)
        self.assertEqual(summary.nb_dels, 1)
        self.assertEqual(summary.ref_len, 4)
        self.assertEqual(summary.nb_lines_summarized, 1)

        self.assertEqual(summary.confusions['a']['a'], 1)
        self.assertEqual(summary.confusions['b']['a'], 0)
        self.assertEqual(summary.confusions['b']['b'], 1)

        extra_b_correctly_matched_to_c = (
            summary.confusions['c']['b'] == 1 and
            summary.confusions['d'][None] == 1
        )

        extra_b_correctly_matched_to_d = (
            summary.confusions['c'][None] == 1 and
            summary.confusions['d']['b'] == 1
        )

        self.assertTrue(extra_b_correctly_matched_to_c or extra_b_correctly_matched_to_d)
        self.assertEqual(sum(sum(k.values()) for k in summary.confusions.values()), len('abcd'))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 1)
        self.assertEqual(summary.ending_errors.correct, 0)

    def test_one_errorneuous_one_perfect(self):
        partial_1 = ErrorsSummary.from_lists(list('abcd'), list('abb'))
        partial_2 = ErrorsSummary.from_lists(list('ab'), list('ab'))
        summary = ErrorsSummary.aggregate([partial_1, partial_2])
        self.assertEqual(summary.nb_errors, 2)
        self.assertEqual(summary.nb_subs, 1)
        self.assertEqual(summary.nb_inss, 0)
        self.assertEqual(summary.nb_dels, 1)
        self.assertEqual(summary.ref_len, 6)
        self.assertEqual(summary.nb_lines_summarized, 2)

        self.assertEqual(summary.confusions['a']['a'], 2)
        self.assertEqual(summary.confusions['b']['a'], 0)
        self.assertEqual(summary.confusions['b']['b'], 2)

        extra_b_correctly_matched_to_c = (
            summary.confusions['c']['b'] == 1 and
            summary.confusions['d'][None] == 1
        )

        extra_b_correctly_matched_to_d = (
            summary.confusions['c'][None] == 1 and
            summary.confusions['d']['b'] == 1
        )

        self.assertTrue(extra_b_correctly_matched_to_c or extra_b_correctly_matched_to_d)
        self.assertEqual(sum(sum(k.values()) for k in summary.confusions.values()), len('abcd') + len('ab'))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 1)
        self.assertEqual(summary.ending_errors.correct, 1)

    def test_one_substituting_one_perfect(self):
        partial_1 = ErrorsSummary.from_lists(list('abcd'), list('abxy'))
        partial_2 = ErrorsSummary.from_lists(list('ab'), list('ab'))
        summary = ErrorsSummary.aggregate([partial_1, partial_2])
        self.assertEqual(summary.nb_errors, 2)
        self.assertEqual(summary.nb_subs, 2)
        self.assertEqual(summary.nb_inss, 0)
        self.assertEqual(summary.nb_dels, 0)
        self.assertEqual(summary.ref_len, 6)
        self.assertEqual(summary.nb_lines_summarized, 2)

        self.assertEqual(summary.confusions['a']['a'], 2)
        self.assertEqual(summary.confusions['b']['b'], 2)
        self.assertEqual(summary.confusions['c']['x'], 1)
        self.assertEqual(summary.confusions['d']['y'], 1)

        self.assertEqual(sum(sum(k.values()) for k in summary.confusions.values()), len('abcd') + len('ab'))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.pure_substitutions, 1)
        self.assertEqual(summary.ending_errors.correct, 1)

    def test_two_summaries_end_deleting(self):
        partial_1 = ErrorsSummary.from_lists(list('abcd'), list('abb'))
        partial_2 = ErrorsSummary.from_lists(list('abc'), list('ab'))
        summary = ErrorsSummary.aggregate([partial_1, partial_2])
        self.assertEqual(summary.nb_errors, 3)
        self.assertEqual(summary.nb_subs, 1)
        self.assertEqual(summary.nb_inss, 0)
        self.assertEqual(summary.nb_dels, 2)
        self.assertEqual(summary.ref_len, 7)
        self.assertEqual(summary.nb_lines_summarized, 2)

        self.assertEqual(summary.confusions['a']['a'], 2)
        self.assertEqual(summary.confusions['b']['b'], 2)

        extra_b_correctly_matched_to_c = (
            summary.confusions['c'][None] == 1 and
            summary.confusions['c']['b'] == 1 and
            summary.confusions['d'][None] == 1
        )

        extra_b_correctly_matched_to_d = (
            summary.confusions['c'][None] == 2 and
            summary.confusions['d']['b'] == 1
        )
        self.assertTrue(extra_b_correctly_matched_to_c or extra_b_correctly_matched_to_d)

        self.assertEqual(sum(sum(k.values()) for k in summary.confusions.values()), len('abcd') + len('abc'))
        self.assertEqual(summary.ending_errors.pure_deletions, 1)
        self.assertEqual(summary.ending_errors.mixed_deletions, 1)
        self.assertEqual(summary.ending_errors.correct, 0)

    def test_two_summaries_end_inserting(self):
        partial_1 = ErrorsSummary.from_lists(list('ab'), list('acd'))
        partial_2 = ErrorsSummary.from_lists(list('ab'), list('abc'))
        summary = ErrorsSummary.aggregate([partial_1, partial_2])
        self.assertEqual(summary.nb_errors, 3)
        self.assertEqual(summary.nb_subs, 1)
        self.assertEqual(summary.nb_inss, 2)
        self.assertEqual(summary.nb_dels, 0)
        self.assertEqual(summary.ref_len, 4)
        self.assertEqual(summary.nb_lines_summarized, 2)

        self.assertEqual(summary.confusions['a']['a'], 2)
        self.assertEqual(summary.confusions['b']['b'], 1)

        unmatched_b_correctly_matched_to_c = (
            summary.confusions['b']['c'] == 1 and
            summary.confusions[None]['d'] == 1
        )

        unmatched_b_correctly_matched_to_d = (
            summary.confusions[None]['c'] == 1 and
            summary.confusions['b']['d'] == 1
        )
        self.assertTrue(unmatched_b_correctly_matched_to_c or unmatched_b_correctly_matched_to_d)

        self.assertEqual(sum(sum(k.values()) for k in summary.confusions.values()), len('acd') + len('abc'))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.pure_insertions, 1)
        self.assertEqual(summary.ending_errors.mixed_insertions, 1)
        self.assertEqual(summary.ending_errors.correct, 0)


class DetailedSummaryTests(unittest.TestCase):
    def test_empty_ref_match(self):
        ref = ''
        hyp = ''
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.confusions, {})

    def test_match(self):
        ref = 'ab'
        hyp = 'ab'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.confusions['a']['a'], 1)
        self.assertEqual(summary.confusions['b']['b'], 1)

    def test_substitution(self):
        ref = 'ab'
        hyp = 'aa'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.confusions['a']['a'], 1)
        self.assertEqual(summary.confusions['b']['b'], 0)
        self.assertEqual(summary.confusions['b']['a'], 1)

    def test_insertion(self):
        ref = 'ab'
        hyp = 'abc'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.confusions['a']['a'], 1)
        self.assertEqual(summary.confusions['b']['b'], 1)
        self.assertEqual(summary.confusions[None]['c'], 1)

    def test_deletion(self):
        ref = 'ab'
        hyp = 'a'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.confusions['a']['a'], 1)
        self.assertEqual(summary.confusions['b']['b'], 0)
        self.assertEqual(summary.confusions['b'][None], 1)


class BoundaryErrorsTests(unittest.TestCase):
    def test_empty_ref_match(self):
        ref = ''
        hyp = ''
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.correct, 1)

    def test_match(self):
        ref = 'ab'
        hyp = 'ab'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.correct, 1)

    def test_substitution_only(self):
        ref = 'ab'
        hyp = 'cd'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.correct, 0)
        self.assertEqual(summary.ending_errors.pure_substitutions, 1)

    def test_end_substitution(self):
        ref = 'ab'
        hyp = 'ac'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.correct, 0)
        self.assertEqual(summary.ending_errors.pure_substitutions, 1)

    def test_start_substitution(self):
        ref = 'ab'
        hyp = 'xb'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.correct, 1)
        self.assertEqual(summary.ending_errors.pure_substitutions, 0)

    def test_end_deletion(self):
        ref = 'ab'
        hyp = 'a'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.ending_errors.pure_deletions, 1)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.correct, 0)

    def test_end_mixed_deletion(self):
        ref = 'abcd'
        hyp = 'abb'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 1)
        self.assertEqual(summary.ending_errors.correct, 0)

    def test_longer_end_deletion(self):
        ref = 'abc'
        hyp = 'a'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.ending_errors.pure_deletions, 1)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.correct, 0)

    def test_end_insertion(self):
        ref = 'a'
        hyp = 'ab'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.pure_insertions, 1)
        self.assertEqual(summary.ending_errors.mixed_insertions, 0)
        self.assertEqual(summary.ending_errors.correct, 0)

    def test_end_mixed_insertion(self):
        ref = 'ab'
        hyp = 'acd'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.pure_insertions, 0)
        self.assertEqual(summary.ending_errors.mixed_insertions, 1)
        self.assertEqual(summary.ending_errors.correct, 0)

    def test_start_deletion(self):
        ref = 'ab'
        hyp = 'b'
        summary = ErrorsSummary.from_lists(list(ref), list(hyp))
        self.assertEqual(summary.ending_errors.pure_deletions, 0)
        self.assertEqual(summary.ending_errors.mixed_deletions, 0)
        self.assertEqual(summary.ending_errors.correct, 1)
