"""The smart reading-order sorter (SmartRegionSorter, REGION_SORTER_SMART, used by most production engines) has no
model, but its geometry goes through shapely and cv2: it must still put the frozen region layouts into exactly the
frozen order, and return exactly the frozen geometry (it rotates tilted pages to sort them and rotates them back).

The frozen data and how it is compared: frozen_engines.py and README.md in this directory."""
from test.engines.consistency.frozen_engines import check


def test_frozen_output_unchanged():
    check('smart_sorter')
