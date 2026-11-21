"""The LAYOUT_CNN engine ([LAYOUT_PARSER] METHOD = LAYOUT_CNN: LayoutExtractor with LayoutEngine) with a ParseNet
exported by an older pero-ocr/torch (data/layout_cnn) must still turn the frozen pages into exactly the frozen page
layouts under the configs of the production engines: regions with ids, categories and outlines, and their lines with
ids, baselines, heights and outlines.

The frozen data and how it is compared: frozen_engines.py and README.md in this directory."""
from test.engines.consistency.frozen_engines import check


def test_frozen_output_unchanged():
    check('layout_cnn')
