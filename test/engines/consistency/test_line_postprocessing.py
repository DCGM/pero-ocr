"""LINE_POSTPROCESSING (LinePostprocessor, PostprocessingEngine) has no model, but its geometry (rotations, shapely
intersections, polynomial fits) depends on numpy and shapely: it must still turn the frozen region layouts
(data/line_postprocessing) into exactly the frozen baselines, heights and line polygons, when stretching lines to the
region border, by pixels, and when resampling them.

The frozen data and how it is compared: frozen_engines.py and README.md in this directory."""
from test.engines.consistency.frozen_engines import check


def test_frozen_output_unchanged():
    check('line_postprocessing')
