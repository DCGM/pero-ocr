import numpy as np


def top_k(a, k, reverse=False):
    flat = a.ravel()

    if len(flat) <= k:
        return np.arange(len(a))

    if reverse:
        top_k_inds = np.argpartition(flat, len(flat)-k)[-k:]
    else:
        top_k_inds = np.argpartition(flat, k)[:k]

    return np.unravel_index(top_k_inds, a.shape)
