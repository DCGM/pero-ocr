import unittest
import numpy as np

from collections import Counter

from pero_ocr.decoding.multisort import top_k


class MultisortTests(unittest.TestCase):
    def test_single_elem(self):
        arr = np.asarray([1])
        inds = np.asarray([0])

        self.assertEqual(top_k(arr, k=1), inds)

    def test_unique_solution(self):
        arr = np.asarray([2, 3, 1, 4, 0, 5])
        inds = np.asarray([4])

        self.assertEqual(top_k(arr, k=1), inds)

    def test_simple_2d(self):
        arr = np.asarray([[2, 3, 1], [4, 0, 5]])
        retval = top_k(arr, k=1)

        self.assertEquals(arr[retval], 0)

    def test_k2_2d(self):
        arr = np.asarray([[2, 3, 1], [4, 0, 5]])
        retval = top_k(arr, k=2)

        self.assertEquals(set(arr[retval].tolist()), set([0, 1]))

    def test_k2_2d_reverse(self):
        arr = np.asarray([[2, 3, 1], [4, 0, 5]])
        retval = top_k(arr, k=2, reverse=True)

        self.assertEquals(set(arr[retval].tolist()), set([4, 5]))

    def test_k2_2d_reverse_duplicit_top(self):
        arr = np.asarray([[2, 5, 1], [4, 0, 5]])
        retval = top_k(arr, k=2, reverse=True)

        self.assertEquals(Counter(arr[retval].tolist()), Counter([5, 5]))
