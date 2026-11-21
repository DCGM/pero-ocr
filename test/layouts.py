"""Small PageLayouts for engine tests: box regions with straight or given baselines."""
import numpy as np

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine


def rotated(points, angle_deg, origin=(0.0, 0.0)):
    """Points rotated by angle_deg (clockwise on the page, as y grows downwards) around origin."""
    a = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(points, dtype=np.float64) - origin) @ rotation.T + origin


def textline(line_id, baseline, heights=(20.0, 5.0), category=None):
    """A line with the given baseline (list of points) and a polygon following it at the given heights."""
    baseline = np.asarray(baseline, dtype=np.float64)
    direction = baseline[-1] - baseline[0]
    up = np.array([direction[1], -direction[0]]) / np.linalg.norm(direction)  # y grows downwards on the page
    polygon = np.concatenate([baseline + heights[0] * up, (baseline - heights[1] * up)[::-1]])
    return TextLine(id=line_id, baseline=baseline, heights=list(heights), polygon=polygon, category=category)


def straight_line(line_id, x0, x1, y, heights=(20.0, 5.0), category=None):
    return textline(line_id, [[x0, y], [x1, y]], heights, category)


def box_region(region_id, l, t, r, b, lines=(), category=None):
    region = RegionLayout(region_id, np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float64), category=category)
    region.lines = list(lines)
    return region


def page(*regions, page_size=(600, 800)):
    layout = PageLayout(id='p', page_size=page_size)
    layout.regions = list(regions)
    return layout
