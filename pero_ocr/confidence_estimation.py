import numpy as np
from scipy.special import logsumexp
import typing


def get_letter_confidence(logits: np.ndarray, alignment: typing.List[int], blank_ind: int) -> typing.List[float]:
    """Function which estimates confidence of characters as the maximal log-prob aligned to them.

    Args:
        logits: numpy array of (unnormalized) log-probabilities of symbols, organized as (time, symbol).
        alignment: a list of symbols assigned to indivudual time frames
        blank_symbol: index of CTC blank in logits, also its representation in alignment

    Returns:
        A list of log probabilities corresponding to non-blank symbols in the alignment.

    Raises:
        Only implicitly.
    """

    log_probs = normalize_logits(logits)
    per_frame_log_probs = pick_elements(log_probs, alignment)
    matched_symbols = squeeze(alignment)
    per_letter_probs = group_elements_by_symbols(per_frame_log_probs, alignment)
    per_letter_probs = [probs for probs, symbol in zip(per_letter_probs, matched_symbols) if symbol != blank_ind]

    return [max(probs) for probs in per_letter_probs]


def normalize_logits(logits):
    return logits - logsumexp(logits, axis=1)[:, np.newaxis]


def pick_elements(elems, inds):
    return elems[np.arange(elems.shape[0]), inds]


def group_elements_by_symbols(elems, symbols):
    grouped = []

    symbol = None
    for e, s in zip(elems, symbols):
        if symbol is None:
            symbol = s
            group = []
        elif s != symbol:
            grouped.append(group)
            group = []
            symbol = s

        group.append(e)
    grouped.append(group)

    return grouped


def squeeze(sequence):
    result = []
    last_symbol = None

    for c in sequence:
        if c == last_symbol:
            continue

        last_symbol = c
        result.append(c)

    return result


