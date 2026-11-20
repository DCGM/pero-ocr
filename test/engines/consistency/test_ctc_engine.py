"""A CTC model exported by an older pero-ocr/torch (test/engines/consistency/data/ctc) must still load and give the same output."""
import os

import numpy as np

from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR
from test.conftest import FROZEN_MODELS
from test.synthetic import line_image


def test_frozen_model_still_works(ctc_json, cpu):
    """A model exported by an older pero-ocr/torch (test/engines/consistency/data) loads and gives the same output."""
    engine = PytorchEngineLineOCR(ctc_json, cpu)
    frozen = PytorchEngineLineOCR(os.path.join(FROZEN_MODELS, 'ctc', 'ocr_engine.json'), cpu)
    lines = [line_image(n) for n in (0, 1, 4)]
    frozen_out, fresh_out = frozen.process_lines(lines), engine.process_lines(lines)
    assert frozen_out[0] == fresh_out[0] == ['', 'a', 'aaaa']
    for a, b in zip(frozen_out[1], fresh_out[1]):
        np.testing.assert_array_equal(a.toarray(), b.toarray())
