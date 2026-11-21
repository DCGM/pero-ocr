"""The CTC engine (PytorchEngineLineOCR) with a CTC model exported by an older pero-ocr/torch (data/ctc) must still
read the frozen line crops into exactly the frozen text, logits and logit coordinates.

The frozen data and how it is compared: frozen_engines.py and README.md in this directory."""
from test.engines.consistency.frozen_engines import check


def test_frozen_output_unchanged():
    check('ctc')
