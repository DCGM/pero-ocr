from unittest import TestCase

import numpy as np
from scipy import sparse

from pero_ocr.document_ocr.layout import TextLine
from pero_ocr.ocr_engine.softmax import softmax


class TextLineTests(TestCase):
    def test_simple(self):
        logits = np.array([
            [1.0, -20.0, -19.0],
            [0.1, 0.1, -21.0],
        ])
        probs = softmax(logits, axis=1)
        logits[probs < 0.0001] = 0
        logits = sparse.csc_matrix(logits)

        line = TextLine(logits=logits)
        reconstructed = line.get_dense_logits(-50.0)
        expected = np.array([
            [1.0, -50.0, -50.0],
            [0.1, 0.1, -50.0],
        ])
        self.assertTrue(np.array_equal(reconstructed, expected))
