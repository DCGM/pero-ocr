"""Merging of two staves joined by a brace into one region of a piano (pianoform) system."""

import numpy as np

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine

PIANOFORM_CATEGORY = 'pianoform'

# A staff belongs to a brace when its vertical center lies within the brace box extended up and down by this fraction
# of the brace height, because a detected brace box is often shorter than the two staves it joins.
BRACE_EXTENSION = 0.3

SYSTEM_PADDING = 20


def merge_pianoform_systems(page_layout: PageLayout, braces: list, page_size, staff_category: str,
                            min_staff_confidence: float) -> PageLayout:
    """Replace each pair of staff regions joined by a brace with one pianoform region.

    Args:
        braces: brace detections as lists starting with x_min, y_min, x_max, y_max, confidence. They are processed
            in the given order and a staff already paired with one brace is not paired again.
        page_size: (height, width) of the page image.
        staff_category: category of the staff regions to pair.
        min_staff_confidence: staves detected with a lower confidence are not paired.
    """
    staves = [region for region in page_layout.regions
              if region.category == staff_category
              and (region.detection_confidence is None or region.detection_confidence >= min_staff_confidence)]
    paired_ids = set()

    for brace in braces:
        brace_left, brace_top, _, brace_bottom, confidence = brace[:5]
        pair = find_staves_for_brace(brace_top, brace_bottom, staves)
        if pair is None:
            continue
        top_stave, bottom_stave = pair
        if top_stave.id in paired_ids or bottom_stave.id in paired_ids:
            continue
        paired_ids.update([top_stave.id, bottom_stave.id])

        box = system_box(top_stave, bottom_stave, brace_left, page_size)
        system = box_region(top_stave.id, box, PIANOFORM_CATEGORY, confidence)
        page_layout.regions[page_layout.regions.index(top_stave)] = system
        page_layout.regions.remove(bottom_stave)

    return page_layout


def find_staves_for_brace(brace_top, brace_bottom, staves):
    """Return the top and bottom staff of a brace, or None unless exactly two staves belong to it.

    More than two staves (e.g. organ systems) are skipped too.
    """
    margin = BRACE_EXTENSION * (brace_bottom - brace_top)
    candidates = [stave for stave in staves
                  if brace_top - margin <= vertical_center(stave) <= brace_bottom + margin]
    if len(candidates) != 2:
        return None
    return sorted(candidates, key=vertical_center)


def system_box(top_stave, bottom_stave, brace_left, page_size):
    """Box around both staves and the brace on their left. Vertically it follows the staves only."""
    page_height, page_width = page_size[:2]
    top_x_min, top_y_min, top_x_max, _ = bounding_box(top_stave)
    bottom_x_min, _, bottom_x_max, bottom_y_max = bounding_box(bottom_stave)

    x_min = max(0, min(brace_left, top_x_min, bottom_x_min) - SYSTEM_PADDING)
    y_min = max(0, top_y_min - SYSTEM_PADDING)
    x_max = min(page_width, max(top_x_max, bottom_x_max) + SYSTEM_PADDING)
    y_max = min(page_height, bottom_y_max + SYSTEM_PADDING)
    return x_min, y_min, x_max, y_max


def box_region(region_id, box, category, confidence):
    """Rectangular region with one line across it."""
    x_min, y_min, x_max, y_max = box
    polygon = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]])
    baseline_y = y_min + (y_max - y_min) / 2
    baseline = np.array([[x_min, baseline_y], [x_max, baseline_y]])
    heights = np.floor(np.array([baseline_y - y_min, y_max - baseline_y]))

    region = RegionLayout(region_id, polygon, category=category, detection_confidence=confidence)
    region.lines.append(TextLine(id=f'{region_id}-l000', index=0, polygon=polygon, baseline=baseline,
                                 heights=heights, category=category))
    return region


def bounding_box(region):
    x_min, y_min = region.polygon.min(axis=0)
    x_max, y_max = region.polygon.max(axis=0)
    return x_min, y_min, x_max, y_max


def vertical_center(region):
    _, y_min, _, y_max = bounding_box(region)
    return (y_min + y_max) / 2
