"""A language model exported by an older pero-ocr/brnolm/torch (data/lm) must still load through construct_lm(), give
the frozen next-character probabilities, and guide CTC decoding of the frozen frames to the frozen text, with the LM
weights used in production.

The frozen data and how it is compared: frozen_engines.py and README.md in this directory."""
from test.engines.consistency.frozen_engines import check


def test_frozen_output_unchanged():
    check('lm')
