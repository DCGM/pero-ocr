"""Module for forced alignment of transcriptions to log-probabilities.

    For purpose of aligning to log-probabilities produced by a CTC model,
    only function force_align() is relevant. Rest is worker functions that
    are used by it.
"""

import numpy as np
import typing
from pero_ocr.utils import jit


def force_align(neg_logprobs: np.ndarray, symbols_seq: typing.List[int], blank_symbol: int, return_seq_positions=False) -> typing.List[int]:
    """Function which force aligns a sequence of symbols to output of a CTC model.

    Args:
        neg_logprobs: numpy array of negative log-probabilities of symbols, organized as (time, symbol).
        symbols_seq: a list of symbols to be aligned with the log-probabilities
        blank_symbol: the CTC blank symbol

    Returns:
        A list of symbols corresponding to the most probable path, including CTC blanks.

    Raises:
        ValueError: On various occassions :-)
    """
    complete_seq, char_sequence = complete_state_seq(symbols_seq, blank_symbol)
    A = hmm_trans_from_string(symbols_seq)
    expanded_logits = expand_logits(neg_logprobs, complete_seq)

    original_align = viterbi_align(expanded_logits, A)

    if return_seq_positions:
        return [char_sequence[s] for s in original_align]
    else:
        return [complete_seq[s] for s in original_align]


def hmm_trans_from_string(elements: typing.List[int]) -> np.ndarray:
    nb_elements = len(elements)
    if nb_elements < 1:
        raise ValueError("Cannot construct a CTC 'HMM' from an empty string")

    nb_states = nb_elements * 2 + 1
    last_nonblank_state = nb_states - 2
    desired = np.full((nb_states, nb_states), np.inf)

    for i in range(nb_states):
        desired[i, i] = 0.0  # we can stay in any state

        if i+1 == nb_states:  # there will be no jumps from the last state
            continue

        desired[i, i+1] = 0.0
        if i % 2 == 1 and i < last_nonblank_state:
            ind_elem = i // 2
            if elements[ind_elem] != elements[ind_elem+1]:
                desired[i, i+2] = 0.0

    return desired


def complete_state_seq(non_blanks: typing.List[int], blank_symbol: int) -> typing.List[int]:
    if blank_symbol in non_blanks:
        raise ValueError(
            "The blank symbol {} is present in the non blank seq {}"
            .format(blank_symbol, non_blanks)
        )

    all_states = np.full(1 + len(non_blanks) * 2, blank_symbol, dtype=int)
    all_states[1::2] = non_blanks
    char_sequence = np.full(1 + len(non_blanks) * 2, -1, dtype=int)
    char_sequence[1::2] = np.arange(len(non_blanks))

    return all_states, char_sequence


def initial_cost(nb_states: int) -> np.ndarray:
    if nb_states < 2:
        raise ValueError(
            "Cannot create initial cost for less than 2 states, got {}".format(nb_states)
        )

    cost = np.full((nb_states, ), np.inf)
    cost[0] = 0.0
    cost[1] = 0.0
    return cost


def final_cost(nb_states: int) -> np.ndarray:
    if nb_states < 2:
        raise ValueError(
            "Cannot create final cost for less than 2 states, got {}".format(nb_states)
        )

    cost = np.full((nb_states, ), np.inf)
    cost[-1] = 0.0
    cost[-2] = 0.0
    return cost


def backtrack(backpointers: np.ndarray, final_state: int) -> typing.List[int]:
    states_from_end = [final_state]

    act_state = final_state
    for i in reversed(range(1, len(backpointers))):
        act_state = backpointers[i, act_state]
        states_from_end.append(act_state)

    return list(reversed(states_from_end))


def expand_logits(array: np.ndarray, seq: typing.List[int]) -> np.ndarray:
    return array[:, seq]


@jit
def compute_update(positions, column_frame, act_cost):
    backpointers = np.zeros(act_cost.shape, np.int32)
    new_cost = np.zeros_like(act_cost)
    new_cost[...] = np.inf

    for j, i in zip(*positions):
        updated_cost = act_cost[j] + column_frame[i]
        if updated_cost < new_cost[i]:
            new_cost[i] = updated_cost
            backpointers[i] = j
    return new_cost, backpointers


def viterbi_align(neg_logits: np.ndarray, A: np.ndarray) -> typing.List[int]:
    nb_states = A.shape[0]
    backpointers = np.full((neg_logits.shape[0], nb_states), -1, dtype=np.int)
    first_frame_cost = initial_cost(nb_states) + neg_logits[0]

    A_positions = np.where(A != np.inf)

    act_cost = first_frame_cost
    for i, frame in enumerate(neg_logits[1:], 1):
        act_cost, backpointers[i] = compute_update(A_positions, frame, act_cost)

    final_frame_cost = act_cost + final_cost(nb_states)

    if np.amin(final_frame_cost) == np.inf:
        raise ValueError("It was not possible to align the states with the logits, best path has cost of np.inf")

    return backtrack(backpointers, np.argmin(final_frame_cost))


def align_text(neg_logprobs, transcription, blank_symbol):
    logit_characters = force_align(neg_logprobs, transcription, blank_symbol, return_seq_positions=True)

    max_probs = (-neg_logprobs).max(axis=-1)

    text_length = transcription.shape[0]

    logit_characters = np.asarray(logit_characters)
    char_positions = np.zeros(text_length, dtype=np.int32)

    for i in range(text_length):
        seq_positions = np.nonzero(logit_characters == i)[0]
        best_pos = np.argmax(max_probs[seq_positions])
        char_positions[i] = seq_positions[best_pos]

    return char_positions
