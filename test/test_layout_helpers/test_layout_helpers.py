import unittest
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
