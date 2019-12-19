import itertools
import numpy as np

from .bag_of_hypotheses import BagOfHypotheses
from .multisort import top_k

from .lm_wrapper import LMWrapper


BLANK_SYMBOL = '<BLANK>'


def duplicit_elements(a_list):
    seen = set()
    duplicit = []

    for x in a_list:
        if x in seen:
            duplicit.append(x)
        else:
            seen.add(x)

    return duplicit


def assert_letters_valid(letters, blank_symbol):
    duplicates = duplicit_elements(letters)
    if duplicates:
        raise ValueError(f"Letters contain these duplicit elements: {duplicates}")

    blank_ind = letters.index(blank_symbol)
    if blank_ind != len(letters) - 1:
        raise ValueError(f"Expected {BLANK_SYMBOL} as the last of letters, it's instead at position {blank_ind}")


class GreedyDecoder:
    def __init__(self, letters):
        assert_letters_valid(letters, BLANK_SYMBOL)
        self._letters = letters
        self._blank_ind = letters.index(BLANK_SYMBOL)

    def __call__(self, logits):
        maxes = logits.max(axis=1)
        argmaxes = logits.argmax(axis=1)

        reduced = [g[0] for g in itertools.groupby(argmaxes)]
        decoded = ''.join(self._letters[ind] for ind in reduced if ind != self._blank_ind)

        bag_of_hyps = BagOfHypotheses()
        bag_of_hyps.add(decoded, np.log(np.sum(maxes)))

        return bag_of_hyps


def get_new_prefixes_positions(best_inds, blank_ind):
    return [i for i, c_ind in enumerate(best_inds[1]) if c_ind != blank_ind]


def get_old_prefixes_positions(best_inds, blank_ind):
    return [i for i, c_ind in enumerate(best_inds[1]) if c_ind == blank_ind]


def reorder_best_inds(best_inds, blank_ind):
    order = get_new_prefixes_positions(best_inds, blank_ind) + get_old_prefixes_positions(best_inds, blank_ind)
    return best_inds[0][order], best_inds[1][order]


def build_boh(prefixes, probs, lm_probs=None):
    bag_of_hyps = BagOfHypotheses()

    if lm_probs is not None:
        for l, P_l, P_lm in zip(prefixes, probs, lm_probs):
            bag_of_hyps.add(l, P_l, P_lm)
    else:
        for l, P_l in zip(prefixes, probs):
            bag_of_hyps.add(l, P_l, 0)

    bag_of_hyps.sort()
    return bag_of_hyps


def get_continuation_mask(nb_prefixes, nb_chars, last_chars, one=1.0, zero=0.0):
    delta = np.full((nb_prefixes, nb_chars), one)
    delta[(np.arange(delta.shape[0]), last_chars)] = zero
    return delta


def update_lm_things(lm, h_prev, lm_preds, best_inds_l, blank_ind):
    if not lm:
        pass
    elif len(get_new_prefixes_positions(best_inds_l, blank_ind)) == 0:
        old_prefix_l_inds = best_inds_l[0][get_old_prefixes_positions(best_inds_l, blank_ind)]
        h_prev = h_prev[old_prefix_l_inds]
        lm_preds = lm_preds[old_prefix_l_inds]
    else:
        new_prefix_positions = get_new_prefixes_positions(best_inds_l, blank_ind)
        new_prefix_l_inds = best_inds_l[0][new_prefix_positions]
        new_prefix_c_inds = best_inds_l[1][new_prefix_positions]
        h_new = lm.advance_h0(new_prefix_c_inds, h_prev[new_prefix_l_inds])
        lm_preds_new = lm.log_probs(h_new)

        old_prefix_l_inds = best_inds_l[0][get_old_prefixes_positions(best_inds_l, blank_ind)]
        h_retained = h_prev[old_prefix_l_inds]
        lm_preds_retained = lm_preds[old_prefix_l_inds]

        h_prev = h_new + h_retained
        lm_preds = np.concatenate([lm_preds_new, lm_preds_retained])

    return h_prev, lm_preds


def find_new_prefixes(prev_l_last, best_inds, A_prev, letters, blank_ind):
    new_l_last = np.ones_like(prev_l_last) * -1
    A_new = []
    new_l_last = []

    for i in get_new_prefixes_positions(best_inds, blank_ind):
        l_ind = best_inds[0][i]
        c_ind = best_inds[1][i]
        new_l_last.append(best_inds[1][i])
        A_new.append(A_prev[l_ind] + letters[c_ind])

    for i in get_old_prefixes_positions(best_inds, blank_ind):
        l_ind = best_inds[0][i]
        new_l_last.append(prev_l_last[l_ind])
        A_new.append(A_prev[l_ind])

    return A_new, np.asarray(new_l_last)


def find_matching(elems, pattern):
    return [i for i, p in enumerate(elems) if p == pattern]


def adjust_for_prefix_joining(P_visual, A_prev, l_lasts, blank_ind):
    for p_ind, prefix in enumerate(A_prev):
        if prefix == '':
            continue

        joinable_prefix_inds = find_matching(A_prev, prefix[:-1])
        if len(joinable_prefix_inds) == 0:
            continue

        assert(len(joinable_prefix_inds) == 1)
        joinable_prefix_ind = joinable_prefix_inds[0]

        original_P = P_visual[p_ind, blank_ind]
        joining_P = P_visual[joinable_prefix_ind, l_lasts[p_ind]]
        resulting_P = np.logaddexp(original_P, joining_P)

        P_visual[p_ind, blank_ind] = resulting_P
        P_visual[joinable_prefix_ind, l_lasts[p_ind]] = -np.inf


def assert_beam_size_valid(k):
    if not isinstance(k, int):
        raise TypeError("Beam size 'k' has to be int, got {} instead (value: {}).".format(type(k), k))

    if k < 1:
        raise ValueError("Beam size 'k' has to be positive, got {} instead.".format(k))


class CTCPrefixLogRawNumpyDecoder:
    def __init__(self, letters, k, lm=None, lm_scale=1.0, use_gpu=False):
        assert_letters_valid(letters, BLANK_SYMBOL)

        self._letters = letters

        assert_beam_size_valid(k)
        self._k = k
        self._lm_scale = lm_scale

        self._blank_ind = self._letters.index(BLANK_SYMBOL)

        if lm:
            self._lm = LMWrapper(lm, letters[:-1], lm_on_gpu=use_gpu)
        else:
            self._lm = None

        LOG_ZERO_PROBABILITY = -1000000  # infinities would not properly compare, leading to NaNs and problems
        self._zero_probs = lambda shape: np.full(shape, LOG_ZERO_PROBABILITY, dtype=np.float32)

    def compute_Pnb(self, Pnb_old, Pb_old, Pc, l_lasts):
        P_continued_letter = Pnb_old + Pc[l_lasts]  # multiplication of probabilities

        P_letter_from_blank = np.add.outer(Pb_old, Pc[:-1])
        delta = get_continuation_mask(Pb_old.shape[0], Pc[:-1].shape[0], l_lasts, one=0.0, zero=-np.inf)
        P_switching_letter = np.add.outer(Pnb_old, Pc[:-1]) + delta  # delta does masking, so anything cancelled is -inf
        Pnb_new_prefixes = np.logaddexp(P_letter_from_blank, P_switching_letter)  # summation of probabilities

        return np.concatenate([Pnb_new_prefixes, P_continued_letter[:, np.newaxis]], axis=1)

    def compute_Plm(self, Plm_old, lm_preds):
        new = Plm_old[:, np.newaxis] + lm_preds
        return np.concatenate([new, Plm_old[:, np.newaxis]], axis=1)

    def compute_Pb(self, Pb_old, Pnb_old, Pc):
        l_Pb = np.logaddexp(Pb_old, Pnb_old) + Pc[-1]  # (Pb_old + Pnb_old) * Pc[-1]
        lc_Pb = self._zero_probs((Pb_old.shape[0], Pc.shape[0]-1))

        return np.concatenate([lc_Pb, l_Pb[:, np.newaxis]], axis=1)

    def __call__(self, logits, model_eos=False):
        ''' inspired by https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306
        '''

        empty = ''
        A_prev = [empty]

        if self._lm:
            h_prev = self._lm.initial_h(1)
            lm_preds = self._lm.log_probs(h_prev)
        else:  # just to have them defined
            h_prev = None
            lm_preds = 0

        Pb_old = self._zero_probs((self._k,))
        Pnb_old = self._zero_probs((self._k,))
        Pb_old[0] = 0.0

        if self._lm:
            Plm_old = self._zero_probs((self._k,))
            Plm_old[0] = 0.0
        else:
            Plm_old = None

        l_lasts = np.zeros(Pb_old.shape, dtype=np.int32)

        for t, Pc in enumerate(logits):
            total_Pnb = self.compute_Pnb(Pnb_old, Pb_old, Pc, l_lasts)
            adjust_for_prefix_joining(total_Pnb, A_prev, l_lasts, self._blank_ind)
            total_Pb = self.compute_Pb(Pb_old, Pnb_old, Pc)
            if self._lm:
                total_Plm = self.compute_Plm(Plm_old, lm_preds)

            visual_P = np.logaddexp(total_Pb, total_Pnb)
            if self._lm:
                total_P = visual_P + total_Plm * self._lm_scale
            else:
                total_P = visual_P

            best_inds_l = top_k(total_P, k=self._k, reverse=True)

            A_prev, l_lasts = find_new_prefixes(l_lasts, best_inds_l, A_prev, self._letters, self._blank_ind)
            h_prev, lm_preds = update_lm_things(self._lm, h_prev, lm_preds, best_inds_l, self._blank_ind)

            new_order = reorder_best_inds(best_inds_l, self._blank_ind)
            Pb_old = total_Pb[new_order]
            Pnb_old = total_Pnb[new_order]
            if self._lm:
                Plm_old = total_Plm[new_order]

        if model_eos:
            eos_scores = self._lm.eos_scores(h_prev)
            Plm_old += eos_scores

        return build_boh(A_prev, np.logaddexp(Pb_old, Pnb_old), Plm_old)
