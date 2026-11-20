"""A ParseNet exported by an older pero-ocr/torch (test/engines/consistency/data/parsenet) must still load and give the same output."""
import os

import numpy as np

from pero_ocr.layout_engines.cnn_layout_engine import LayoutEngine
from test.conftest import FROZEN_MODELS
from test.synthetic import page_image


def test_frozen_model_still_works(parsenet_path, cpu):
    """A model exported by an older pero-ocr/torch (test/engines/consistency/data) loads and gives the same output."""
    engine = LayoutEngine(parsenet_path, cpu, downsample=4, detection_threshold=0.2, max_mp=5)
    frozen = LayoutEngine(os.path.join(FROZEN_MODELS, 'parsenet', 'parsenet.pt'), cpu, downsample=4,
                          detection_threshold=0.2, max_mp=5)
    page = page_image(3)
    frozen_baselines, fresh_baselines = frozen.detect(page)[1], engine.detect(page)[1]
    assert len(frozen_baselines) == len(fresh_baselines) == 3
    for a, b in zip(frozen_baselines, fresh_baselines):
        np.testing.assert_allclose(a, b)
