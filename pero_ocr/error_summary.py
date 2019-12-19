import math
from pero_ocr.sequence_alignment import levenshtein_distance
from pero_ocr.sequence_alignment import levenshtein_alignment, edit_stats_for_alignment
from enum import Enum
from collections import defaultdict, Counter

MatchTypes = Enum('MatchTypes', 'C S I D')


def get_match_type(ref_sym, hyp_sym):
    if ref_sym is None and hyp_sym is None:
        raise AssertionError("Invalid alignment None-None")

    if ref_sym == hyp_sym:
        return MatchTypes.C
    elif ref_sym is None:
        return MatchTypes.I
    elif hyp_sym is None:
        return MatchTypes.D
    else:
        return MatchTypes.S


class BoundaryErrorsSummary:
    def __init__(self, boundary_alignment):
        if MatchTypes.I in boundary_alignment and MatchTypes.D in boundary_alignment:
            raise AssertionError('Got both insertion and deletion in the ending errors.')

        c = False
        pd = False
        pi = False
        md = False
        mi = False
        ps = False
        if len(boundary_alignment) == 0:
            c = True
        elif MatchTypes.S in boundary_alignment and MatchTypes.D in boundary_alignment:
            md = True
        elif MatchTypes.S in boundary_alignment and MatchTypes.I in boundary_alignment:
            mi = True
        elif MatchTypes.D in boundary_alignment:
            pd = True
        elif MatchTypes.I in boundary_alignment:
            pi = True
        elif MatchTypes.S in boundary_alignment:
            ps = True

        self.correct = c
        self.pure_deletions = pd
        self.mixed_deletions = md
        self.pure_insertions = pi
        self.mixed_insertions = mi
        self.pure_substitutions = ps

    def __eq__(self, other):
        return (
            self.pure_deletions == other.pure_deletions and
            self.mixed_deletions == other.mixed_deletions
        )

    def __iadd__(self, other):
        self.pure_deletions += other.pure_deletions
        self.mixed_deletions += other.mixed_deletions
        self.pure_insertions += other.pure_insertions
        self.mixed_insertions += other.mixed_insertions
        self.pure_substitutions += other.pure_substitutions
        self.correct += other.correct

        return self

    @staticmethod
    def empty_summary():
        summary = BoundaryErrorsSummary.__new__(BoundaryErrorsSummary)
        summary.correct = 0
        summary.pure_deletions = 0
        summary.mixed_deletions = 0
        summary.pure_insertions = 0
        summary.mixed_insertions = 0
        summary.pure_substitutions = 0

        return summary


def get_non_matching_prefix(alignment_types):
    prefix = []

    for align_type in alignment_types:
        if align_type == MatchTypes.C:
            break

        prefix.append(align_type)

    return prefix


def get_non_matching_suffix(alignment_types):
    rev_suffix = get_non_matching_prefix(reversed(alignment_types))
    return list(reversed(rev_suffix))


class ErrorsSummary:
    def __init__(self, nb_lines_summarized, ref_len, nb_errors, nb_subs, nb_inss, nb_dels, confusions, ending_errors):
        self.nb_lines_summarized = nb_lines_summarized
        self.nb_errors = nb_errors
        self.nb_subs = nb_subs
        self.nb_inss = nb_inss
        self.nb_dels = nb_dels
        self.ref_len = ref_len
        self.confusions = confusions
        self.ending_errors = ending_errors

        if self.ref_len > 0:
            self.error_rate = self.nb_errors / self.ref_len
        else:
            self.error_rate = math.inf

    @classmethod
    def from_lists(cls, ref, hyp):
        ref_len = len(ref)
        nb_errors = levenshtein_distance(ref, hyp)

        alignment = levenshtein_alignment(hyp, ref)
        _, _, nb_inss, nb_dels, nb_subs = edit_stats_for_alignment(alignment)

        confusions = defaultdict(Counter)
        for hyp_sym, ref_sym in alignment:
            confusions[ref_sym][hyp_sym] += 1

        match_types = [get_match_type(a[1], a[0]) for a in alignment]
        ending_mistakes = get_non_matching_suffix(match_types)
        end_errors = BoundaryErrorsSummary(ending_mistakes)

        return cls(1, ref_len, nb_errors, nb_subs, nb_inss, nb_dels, confusions, end_errors)

    @staticmethod
    def aggregate(errors):
        total_nb_lines = 0
        total_ref_len = 0
        total_nb_errors = 0
        total_nb_subs = 0
        total_nb_inss = 0
        total_nb_dels = 0
        total_ending_erros = BoundaryErrorsSummary.empty_summary()
        total_confusions = defaultdict(Counter)

        for err in errors:
            total_nb_lines += err.nb_lines_summarized
            total_ref_len += err.ref_len
            total_nb_errors += err.nb_errors
            total_nb_subs += err.nb_subs
            total_nb_inss += err.nb_inss
            total_nb_dels += err.nb_dels

            for k in err.confusions:
                total_confusions[k].update(err.confusions[k])

            total_ending_erros += err.ending_errors

        return ErrorsSummary(
            total_nb_lines, total_ref_len, total_nb_errors,
            total_nb_subs, total_nb_inss, total_nb_dels,
            total_confusions, total_ending_erros
        )

    def __str__(self):
        return "{:.2f} % ( {} / {} ; sub: {} ins: {} del: {} )".format(100.0*self.error_rate, self.nb_errors, self.ref_len, self.nb_subs, self.nb_inss, self.nb_dels)
