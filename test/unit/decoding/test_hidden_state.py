import unittest

import numpy as np
import torch

from pero_ocr.decoding.lm_wrapper import HiddenState


class HiddenStateTests(unittest.TestCase):
    def test_keeps_value(self):
        in_h = torch.from_numpy(np.asarray([-2.0])).to(dtype=torch.float32)
        h0 = HiddenState(in_h)
        self.assertEqual(h0.prepare_for_torch(), in_h)

    def test_keeps_value_array(self):
        in_h = torch.from_numpy(np.asarray([-2.0, 3.0])).to(dtype=torch.float32)
        h0 = HiddenState(in_h)
        self.assertTrue(torch.equal(h0.prepare_for_torch(), in_h))

    def test_keeps_value_lstm_double_state(self):
        in_h = torch.from_numpy(np.asarray([[-1.0, 3.0], [-2.0, 4.0]])).to(dtype=torch.float32)
        in_c = torch.from_numpy(np.asarray([[-1.0, 3.0], [-2.0, 4.0]])).to(dtype=torch.float32)
        h0 = HiddenState((in_h, in_c))
        self.assertTrue(torch.equal(h0.prepare_for_torch()[0], in_h))
        self.assertTrue(torch.equal(h0.prepare_for_torch()[1], in_c))
