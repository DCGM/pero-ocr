"""LAYOUT_POSTPROCESSING (LayoutPostprocessor) has no model, but retracing region outlines (a Delaunay triangulation
from scipy, unions and simplification in shapely) depends on those libraries: it must still turn the frozen region
layouts (data/layout_postprocessing) into exactly the frozen region outlines.

The frozen data and how it is compared: frozen_engines.py and README.md in this directory."""
from test.engines.consistency.frozen_engines import check


def test_frozen_output_unchanged():
    check('layout_postprocessing')
