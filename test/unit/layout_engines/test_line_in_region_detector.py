"""detect_lines_in_region (layout_engines/line_in_region_detector.py), the straight lines of DETECT_STRAIGHT_LINES_IN_REGIONS:
the lines follow the direction of the baseline detections in the region, on hand-made detection maps.

The region angle is taken from a row of the eigenvector matrix (the mirrored direction), and the rotation that
follows uses the opposite sign, so the two cancel; these tests pin the result for any rewrite (numpy >= 2.5 returns
complex eigenvectors, which the current code cannot use, #97)."""
import numpy as np
import pytest

from pero_ocr.layout_engines.line_in_region_detector import detect_lines_in_region

SIZE = 400
REGION = np.array([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]], dtype=float)


def maps_with_lines(angle_deg, rows=(120, 200, 280), height=6.0):
    """Detection maps (ascender height, descender height, baselines, ...) with straight baselines through
    (SIZE / 2, row) at angle_deg (positive: going down to the right, image coordinates)."""
    maps = np.zeros((SIZE, SIZE, 5), dtype=np.float32)
    maps[:, :, 0] = maps[:, :, 1] = height
    slope = np.tan(np.radians(angle_deg))
    for x in range(20, SIZE - 20):
        for row in rows:
            y = int(round(row + (x - SIZE / 2) * slope))
            maps[y - 1:y + 2, x, 2] = 1.0
    return maps


@pytest.mark.parametrize('angle', [-30.0, -10.0, 0.0, 5.0, 20.0])
def test_lines_follow_the_detections(angle):
    baselines, heights, polygons = detect_lines_in_region(REGION, maps_with_lines(angle), downsample=1)
    assert len(baselines) == len(heights) == len(polygons) == 3
    for b in baselines:
        found = np.degrees(np.arctan2(b[-1, 1] - b[0, 1], b[-1, 0] - b[0, 0]))
        assert abs(found - angle) < 1.0
    middle = sorted(baselines, key=lambda b: b[:, 1].mean())[1]
    assert abs(np.interp(SIZE / 2, middle[:, 0], middle[:, 1]) - 200) < 3
