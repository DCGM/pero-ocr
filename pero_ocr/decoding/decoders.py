import itertools
import numpy as np

from .bag_of_hypotheses import BagOfHypotheses, logsumexp
from .multisort import top_k

from .lm_wrapper import LMWrapper, HiddenState


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


def logprobs_max_deviation(log_probs):
    probs = np.exp(log_probs)
    sums = np.sum(probs, axis=1)
    return np.max(np.abs(sums - 1))


class GreedyDecoder:
    def __init__(self, letters):
        assert_letters_valid(letters, BLANK_SYMBOL)
        self._letters = letters
        self._blank_ind = letters.index(BLANK_SYMBOL)

    def __call__(self, logits, max_unnormalization=1e-5):
        if logprobs_max_deviation(logits) > max_unnormalization:
            raise ValueError('Expected properly normalized logits')

        maxes = logits.max(axis=1)
        argmaxes = logits.argmax(axis=1)

        reduced = [g[0] for g in itertools.groupby(argmaxes)]
        decoded = ''.join(self._letters[ind] for ind in reduced if ind != self._blank_ind)

        bag_of_hyps = BagOfHypotheses()
        bag_of_hyps.add(decoded, logsumexp(maxes))

        return bag_of_hyps


def get_new_prefixes_positions(best_inds, blank_ind):
    return [i for i, c_ind in enumerate(best_inds[1]) if c_ind != blank_ind]


def get_old_prefixes_positions(best_inds, blank_ind):
    return [i for i, c_ind in enumerate(best_inds[1]) if c_ind == blank_ind]


def reorder_best_inds(best_inds, blank_ind):
    order = get_new_prefixes_positions(best_inds, blank_ind) + get_old_prefixes_positions(best_inds, blank_ind)
    return best_inds[0][order], best_inds[1][order]


def build_boh(prefixes, probs, lm_probs=None, lm_weight=1.0):
    bag_of_hyps = BagOfHypotheses(lm_weight)

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
        return h_prev, lm_preds

    h_new = h_prev[best_inds_l[0]]
    lm_preds_new = lm_preds[best_inds_l[0]]

    new_prefix_positions = get_new_prefixes_positions(best_inds_l, blank_ind)
    if new_prefix_positions:
        new_prefix_l_inds = best_inds_l[0][new_prefix_positions]
        new_prefix_c_inds = best_inds_l[1][new_prefix_positions]
        h_replacement = lm.advance_h0(new_prefix_c_inds, h_prev[new_prefix_l_inds])
        lm_preds_new[new_prefix_positions] = lm.log_probs(h_replacement)
        h_new[new_prefix_positions] = h_replacement

    return h_new, lm_preds_new


def find_new_prefixes(prev_l_last, best_inds, A_prev, letters, blank_ind):
    new_l_last = np.ones((len(best_inds[0]),)) * -1
    A_new = [None] * len(best_inds[0])

    for i in get_new_prefixes_positions(best_inds, blank_ind):
        l_ind = best_inds[0][i]
        c_ind = best_inds[1][i]
        new_l_last[i] = c_ind
        A_new[i] = A_prev[l_ind] + letters[c_ind]

    for i in get_old_prefixes_positions(best_inds, blank_ind):
        l_ind = best_inds[0][i]
        new_l_last[i] = prev_l_last[l_ind]
        A_new[i] = A_prev[l_ind]

    return A_new, new_l_last


def find_matching(elems, pattern):
    return [i for i, p in enumerate(elems) if p == pattern]


def adjust_for_prefix_joining(P_visual, A_prev, last_chars):
    for p_ind, prefix in enumerate(A_prev):
        if prefix == '':
            continue

        joinable_prefix_inds = find_matching(A_prev, prefix[:-1])
        if len(joinable_prefix_inds) == 0:
            continue

        assert(len(joinable_prefix_inds) == 1)
        joinable_prefix_ind = joinable_prefix_inds[0]

        original_P = P_visual[p_ind, -1]
        joining_P = P_visual[joinable_prefix_ind, last_chars[p_ind]]
        resulting_P = np.logaddexp(original_P, joining_P)

        P_visual[p_ind, -1] = resulting_P
        P_visual[joinable_prefix_ind, last_chars[p_ind]] = -np.inf


def assert_beam_size_valid(k):
    if not isinstance(k, int):
        raise TypeError("Beam size 'k' has to be int, got {} instead (value: {}).".format(type(k), k))

    if k < 1:
        raise ValueError("Beam size 'k' has to be positive, got {} instead.".format(k))


def select_relevant_logits(logits):
    return np.nonzero(logits > -10)


class CTCPrefixLogRawNumpyDecoder:
    def __init__(self, letters, k,
                 lm=None, lm_scale=1.0, insertion_bonus=0.0,
                 relevant_logits_selector=select_relevant_logits):
        assert_letters_valid(letters, BLANK_SYMBOL)

        self._letters = letters

        assert_beam_size_valid(k)
        self._k = k
        self._lm_scale = lm_scale
        self._insertion_bonus = insertion_bonus

        self._blank_ind = self._letters.index(BLANK_SYMBOL)
        self.select_relevant_logits = relevant_logits_selector

        self._lm = lm

        self.LOG_ZERO_PROBABILITY = -np.inf

    def compute_Pnb(self, Pnb_old, Pb_old, Pc, last_chars):
        P_continued_letter = Pnb_old + Pc[last_chars]  # multiplication of probabilities

        P_letter_from_blank = np.add.outer(Pb_old, Pc)
        delta = get_continuation_mask(Pb_old.shape[0], Pc.shape[0], last_chars, one=0.0, zero=-np.inf)
        P_switching_letter = np.add.outer(Pnb_old, Pc) + delta  # delta does masking, so anything cancelled is -inf
        Pnb_new_prefixes = np.logaddexp(P_letter_from_blank, P_switching_letter)  # summation of probabilities

        return np.concatenate([Pnb_new_prefixes, P_continued_letter[:, np.newaxis]], axis=1)

    def compute_Plm(self, Plm_old, lm_preds):  # TODO can be cached too
        new = Plm_old[:, np.newaxis] + lm_preds + self._insertion_bonus
        return np.concatenate([new, Plm_old[:, np.newaxis]], axis=1)

    def compute_Pb(self, Pb_old, Pnb_old, P_blank):
        return np.logaddexp(Pb_old, Pnb_old) + P_blank  # (Pb_old + Pnb_old) * P_blank

    def get_reduced_Pc(self, Pc, selected_chars):
        reduced_Pc = Pc[selected_chars]
        neginf = np.asarray([self.LOG_ZERO_PROBABILITY])
        return np.concatenate([reduced_Pc, neginf])

    def get_reduced_last_chars(self, last_chars, selected_chars, impossible_index):
        reduced_last_chars = last_chars.copy()
        inv_sel = {v: i for i, v in enumerate(selected_chars)}
        return np.asarray([(inv_sel[l] if l in inv_sel else impossible_index) for l in reduced_last_chars])

    def __call__(self, logits, model_eos=False, max_unnormalization=1e-5, return_h=False, init_h=None):
        ''' inspired by https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306
        '''
        if logprobs_max_deviation(logits) > max_unnormalization:
            raise ValueError('Expected properly normalized logits')

        empty = ''
        prefixes = [empty]

        if self._lm:
            if init_h is None:
                h_prev = self._lm.initial_h(1)
            else:
                h_prev = init_h
            lm_preds = self._lm.log_probs(h_prev)
        else:  # just to have them defined
            h_prev = None
            lm_preds = 0

        Pb = np.asarray([0.0])
        Pnb = np.asarray([self.LOG_ZERO_PROBABILITY])

        if self._lm:
            Plm = np.asarray([0.0])
        else:
            Plm = None

        last_chars = np.zeros(Pb.shape, dtype=np.int32)

        for t, Pc in enumerate(logits):
            P_blank = Pc[-1]

            selected_chars = self.select_relevant_logits(Pc[:-1])[0]
            if selected_chars.shape[0] == 0:
                Pb = self.compute_Pb(Pb, Pnb, P_blank)
                Pnb[...] = self.LOG_ZERO_PROBABILITY
                continue

            reduced_Pc = self.get_reduced_Pc(Pc, selected_chars)
            reduced_last_chars = self.get_reduced_last_chars(last_chars, selected_chars, reduced_Pc.shape[0]-1)

            total_Pnb = self.compute_Pnb(Pnb, Pb, reduced_Pc, reduced_last_chars)
            adjust_for_prefix_joining(total_Pnb, prefixes, reduced_last_chars)

            total_Pb = self.compute_Pb(Pb, Pnb, P_blank)

            visual_P = total_Pnb.copy()
            visual_P[:, -1] = np.logaddexp(total_Pb, visual_P[:, -1])

            randchar = np.asarray([-2, self._blank_ind])
            selected_chars = np.concatenate([selected_chars, randchar])
            if self._lm:
                total_Plm = self.compute_Plm(Plm, lm_preds)[:, selected_chars]
                total_P = visual_P + total_Plm * self._lm_scale
            else:
                total_P = visual_P

            best_inds = top_k(total_P, k=min([self._k, np.sum(np.isfinite(total_P))]), reverse=True)

            Pb = total_Pb[best_inds[0]]
            Pb[best_inds[1] != total_P.shape[1]-1] = self.LOG_ZERO_PROBABILITY
            Pnb = total_Pnb[best_inds]
            if self._lm:
                Plm = total_Plm[best_inds]

            best_inds = best_inds[0], np.asarray([selected_chars[x] for x in best_inds[1]])

            prefixes, last_chars = find_new_prefixes(last_chars, best_inds, prefixes, self._letters, self._blank_ind)
            h_prev, lm_preds = update_lm_things(self._lm, h_prev, lm_preds, best_inds, self._blank_ind)

        if model_eos:
            eos_scores = self._lm.eos_scores(h_prev)
            Plm += eos_scores

        Pom = np.logaddexp(Pb, Pnb)
        bag_of_hypotheses = build_boh(prefixes, Pom, Plm, lm_weight=self._lm_scale)
        if return_h:
            idx_of_best = np.argmax(Pom + Plm*self._lm_scale)
            return bag_of_hypotheses, h_prev[[idx_of_best]]  # a single-item list is needed to keep shape
        else:
            return bag_of_hypotheses
