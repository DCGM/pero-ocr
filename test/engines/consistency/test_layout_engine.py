"""The ParseNet layout engine (LayoutEngine) with a model exported by an older pero-ocr/torch (data/parsenet) must still
find exactly the frozen regions, baselines and heights on the frozen pages.

The frozen data and how it is compared: frozen_engines.py and README.md in this directory."""
from test.engines.consistency.frozen_engines import check


def test_frozen_output_unchanged():
    check('parsenet')
