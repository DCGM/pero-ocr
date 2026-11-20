"""Models exported with an older pero-ocr/torch (test/data/models) must keep loading and working."""
import os

import numpy as np
import pytest

from pero_ocr.layout_engines.cnn_layout_engine import LayoutEngine
from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR
from pero_ocr.ocr_engine.transformer_ocr_engine import TransformerEngineLineOCR
from test.conftest import DATA_DIR
from test.synthetic import line_image, page_image

FROZEN = os.path.join(DATA_DIR, 'models')
LINES = [line_image(n) for n in (0, 1, 4)]


def test_ctc(cpu, ctc_json):
    frozen = PytorchEngineLineOCR(os.path.join(FROZEN, 'ctc', 'ocr_engine.json'), cpu)
    fresh = PytorchEngineLineOCR(ctc_json, cpu)
    frozen_out, fresh_out = frozen.process_lines(LINES), fresh.process_lines(LINES)
    assert frozen_out[0] == fresh_out[0] == ['', 'a', 'aaaa']
    for a, b in zip(frozen_out[1], fresh_out[1]):
        np.testing.assert_array_equal(a.toarray(), b.toarray())


@pytest.mark.parametrize('beam_size', [1, 3])
def test_transformer(cpu, transformer_json, beam_size):
    frozen = TransformerEngineLineOCR(os.path.join(FROZEN, 'transformer', 'ocr_engine.json'), cpu,
                                      beam_size=beam_size)
    fresh = TransformerEngineLineOCR(transformer_json, cpu, beam_size=beam_size)
    assert frozen.process_lines(LINES)[0] == fresh.process_lines(LINES)[0] == ['', 'a', 'abcd']


def test_parsenet(cpu, parsenet_path):
    frozen = LayoutEngine(os.path.join(FROZEN, 'parsenet', 'parsenet.pt'), cpu, downsample=4)
    fresh = LayoutEngine(parsenet_path, cpu, downsample=4)
    page = page_image(3)
    for a, b in zip(frozen.detect(page)[1], fresh.detect(page)[1]):
        np.testing.assert_allclose(a, b)
