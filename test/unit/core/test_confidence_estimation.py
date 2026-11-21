"""Confidence estimation from CTC log-probabilities (pero_ocr.core.confidence_estimation): character, word,
transcription and page confidences."""
import numpy as np
import pytest

from pero_ocr.core import confidence_estimation as ce


def test_squeeze():
    assert ce.squeeze([1, 1, 2, 2, 2, 1, 3]) == [1, 2, 1, 3]


def test_group_elements_by_symbols():
    assert ce.group_elements_by_symbols([10, 11, 12, 13], [0, 0, 1, 0]) == [[10, 11], [12], [13]]


def test_get_letter_confidence_takes_best_frame():
    blank = 2
    logits = np.log(np.array([
        [0.9, 0.05, 0.05],
        [0.6, 0.2, 0.2],
        [0.05, 0.05, 0.9],
        [0.1, 0.8, 0.1],
    ]))
    confidences = ce.get_letter_confidence(logits, [0, 0, blank, 1], blank)
    np.testing.assert_allclose(np.exp(confidences), [0.9, 0.8])


def test_page_confidence_quantile():
    assert ce.get_page_confidence_from_transcription_confidences([]) == 0
    assert ce.get_page_confidence_from_transcription_confidences([0.5, 0.5, 0.5]) == pytest.approx(0.5)
    conf = ce.get_page_confidence_from_transcription_confidences([0.1, 0.9, 0.9, 0.9])
    assert 0.1 < conf < 0.9


def test_transcription_confidence_from_characters():
    assert ce.get_transcription_confidence_from_characters(np.array([1.0, 1.0])) == pytest.approx(1.0)
    low = ce.get_transcription_confidence_from_characters(np.array([1.0, 0.2, 1.0]))
    assert 0.2 < low < 1.0


def test_word_confidence_from_characters():
    assert ce.get_word_confidence_from_characters(np.array([0.9, 0.9])) == pytest.approx(0.9, abs=0.01)
