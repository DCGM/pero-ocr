"""A transformer exported by an older pero-ocr/torch (test/engines/consistency/data/transformer) must still load and give the same
output."""
import os

import pytest

from pero_ocr.ocr_engine.transformer_ocr_engine import TransformerEngineLineOCR
from test.conftest import FROZEN_MODELS
from test.synthetic import line_image


@pytest.mark.parametrize('beam_size', [1, 3])
def test_frozen_model_still_works(transformer_json, cpu, beam_size):
    """A model exported by an older pero-ocr/torch (test/engines/consistency/data) loads and gives the same output."""
    frozen = TransformerEngineLineOCR(os.path.join(FROZEN_MODELS, 'transformer', 'ocr_engine.json'), cpu,
                                      beam_size=beam_size)
    fresh = TransformerEngineLineOCR(transformer_json, cpu, beam_size=beam_size)
    lines = [line_image(n) for n in (0, 1, 4)]
    assert frozen.process_lines(lines)[0] == fresh.process_lines(lines)[0] == ['', 'a', 'abcd']
