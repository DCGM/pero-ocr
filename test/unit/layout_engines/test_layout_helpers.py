"""Layout helpers (pero_ocr.layout_engines.layout_helpers) used inside the layout engines: filtering lists by a mask,
the line polygon from a baseline and its heights (baseline_to_textline)."""
import unittest

import numpy as np
import pytest

from pero_ocr.layout_engines import layout_helpers


class FilterLinesTests(unittest.TestCase):
    def test_filter_nothing(self):
        filtered = layout_helpers.filter_list([0, 1, 2], [])
        self.assertEqual(filtered, [0, 1, 2])

    def test_filter_all(self):
        filtered = layout_helpers.filter_list([0, 1, 2], [0, 1, 2])
        self.assertEqual(filtered, [])

    def test_filter_middle(self):
        filtered = layout_helpers.filter_list([0, 1, 2], [1])
        self.assertEqual(filtered, [0, 2])

    def test_repated_value(self):
        filtered = layout_helpers.filter_list([0, 1, 0], [0, 2])
        self.assertEqual(filtered, [1])

    def test_repeated_mask(self):
        filtered = layout_helpers.filter_list([0, 1, 2], [0, 0])
        self.assertEqual(filtered, [1, 2])

    def test_negative_mask(self):
        filtered = layout_helpers.filter_list([0, 1, 2], [-1])
        self.assertEqual(filtered, [0, 1])

    def test_fail_for_too_high(self):
        self.assertRaises(Exception, layout_helpers.filter_list, [0, 1, 0], [3])


def test_textline_around_a_horizontal_baseline():
    polygon = layout_helpers.baseline_to_textline(np.array([[10., 20.], [50., 20.]]), [5, 3])
    assert polygon.tolist() == [[10, 15], [50, 15], [50, 23], [10, 23]]


@pytest.mark.xfail(reason='#96: baseline_to_textline crashes (ValueError, non-broadcastable output operand) on a '
                          'baseline with a single point; such lines exist in annotated data', strict=True)
def test_textline_around_a_single_point_baseline():
    polygon = layout_helpers.baseline_to_textline(np.array([[10., 20.]]), [5, 3])
    assert polygon[:, 1].min() == pytest.approx(15)
    assert polygon[:, 1].max() == pytest.approx(23)
