"""Merging the parts of a long line that the OCR engines split (pero_ocr.ocr_engine.line_ocr_engine): finding the
overlap and merging transcriptions and logits."""
import numpy as np
import pytest

from pero_ocr.ocr_engine.line_ocr_engine import find_best_overlap, merge_transcriptions_and_logits


def logits_for(text):
    return np.arange(len(text) * 2, dtype=np.float32).reshape(len(text), 2)


def test_find_best_overlap():
    assert find_best_overlap('hello wor', 'world') == 3
    assert find_best_overlap('abc', 'abc') == 3


def test_merge_with_overlap():
    parts = ['hello wor', 'world peace']
    text, logits = merge_transcriptions_and_logits(parts, [logits_for(p) for p in parts])
    assert text == 'hello world peace'
    assert logits.shape[0] == len(text)


@pytest.mark.xfail(reason='with no overlap, [:-overlap // 2] is [:0], which drops the whole first part',
                   strict=True)
def test_merge_without_overlap():
    parts = ['abc', 'xyz']
    text, _ = merge_transcriptions_and_logits(parts, [logits_for(p) for p in parts])
    assert text == 'abcxyz'


def test_merge_trims_logits_to_transcriptions():
    parts = ['abcd', 'cdef']
    padded = [np.zeros((10, 2), dtype=np.float32) for _ in parts]
    text, logits = merge_transcriptions_and_logits(parts, padded)
    assert logits.shape[0] == len(text)
