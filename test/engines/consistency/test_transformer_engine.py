"""The transformer engine (TransformerEngineLineOCR, exported torchscript) with a model exported by an older
pero-ocr/torch (data/transformer) must still decode the frozen line crops into exactly the frozen text, logits and logit
coordinates, with greedy decoding and with beam search.

The frozen data and how it is compared: frozen_engines.py and README.md in this directory."""
from test.engines.consistency.frozen_engines import check


def test_frozen_output_unchanged():
    check('transformer')
