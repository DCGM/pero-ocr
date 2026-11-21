"""The transformer reading-order sorter (TransformerRegionSorter, REGION_SORTER_TRANSFORMER) with a model exported by an
older pero-ocr/torch (data/region_sorter_transformer) must still put the frozen regions into exactly the frozen order.

The frozen data and how it is compared: frozen_engines.py and README.md in this directory."""
from test.engines.consistency.frozen_engines import check


def test_frozen_output_unchanged():
    check('region_sorter_transformer')
