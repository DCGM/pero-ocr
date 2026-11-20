"""A language model exported by an older pero-ocr/brnolm/torch (test/engines/consistency/data/lm) must still load and
guide decoding the same way."""
import os

import numpy as np
import torch

from pero_ocr.decoding.decoders import BLANK_SYMBOL, CTCPrefixLogRawNumpyDecoder
from pero_ocr.decoding.decoding_itf import construct_lm
from pero_ocr.decoding.lm_wrapper import LMWrapper
from test import tiny_models
from test.conftest import FROZEN_MODELS

CHARS = tiny_models.CHARACTERS
CPU = torch.device('cpu')


def ambiguous_after_a():
    """CTC log-probs reading 'a', then 'c' (0.6) vs 'b' (0.4): the LM decides."""
    n = len(CHARS) + 1
    rows = []
    for spec in [{'a': .98}, {n - 1: .98}, {'c': .6, 'b': .4}, {n - 1: .98}]:
        f = np.full(n, 1e-4)
        for k, p in spec.items():
            f[k if isinstance(k, int) else CHARS.index(k)] = p
        rows.append(np.log(f / f.sum()))
    return np.stack(rows)


def test_frozen_model_still_works(lm_path):
    frozen = LMWrapper(construct_lm(os.path.join(FROZEN_MODELS, 'lm', 'lm.pt')), CHARS, CPU)
    fresh = LMWrapper(construct_lm(lm_path), CHARS, CPU)

    h_frozen = frozen.advance_h0(np.array([CHARS.index('a')]), frozen.initial_h(1))
    h_fresh = fresh.advance_h0(np.array([CHARS.index('a')]), fresh.initial_h(1))
    np.testing.assert_allclose(frozen.log_probs(h_frozen), fresh.log_probs(h_fresh))

    decode = [CTCPrefixLogRawNumpyDecoder(CHARS + [BLANK_SYMBOL], 8, w, 1.0)(ambiguous_after_a()).best_hyp()
              for w in (frozen, fresh)]
    assert decode == ['ab', 'ab']
