"""The line cropper must still cut exactly the same crops as when they were frozen (test/engines/consistency/data/cropper).

The crops are the input of every OCR model, and the models were trained on crops made by the cropper of their time.
A change of a single grey level in some pixels already changes the recognized text on real pages (opencv 5 changed
cv2.remap this way: about 6 % of the pixels by up to 3 grey levels, and 0.1 to 0.5 % of the characters on real
pages), so crops are compared exactly."""
import numpy as np
import pytest

from test.engines.consistency.frozen_crops import crop_case, load_frozen

PAGE, CASES, FROZEN = load_frozen()


def test_data_matches_the_cases():
    expected = set()
    for case in CASES:
        expected |= set(crop_case(PAGE, case))
    assert set(FROZEN) == expected, 'data/cropper is out of date: run test/engines/consistency/data/regenerate.py cropper'


@pytest.mark.parametrize('case', CASES, ids=[c['name'] for c in CASES])
def test_crops_are_unchanged(case):
    for key, value in crop_case(PAGE, case).items():
        frozen = FROZEN[key]
        assert value.shape == frozen.shape, f'{key}: shape {value.shape}, frozen {frozen.shape}'
        if key.endswith('/coords'):
            np.testing.assert_allclose(value, frozen, atol=1e-3, err_msg=key)
            continue
        differing = value != frozen
        assert not differing.any(), (
            f'{key}: {differing.mean():.2%} of the values differ, by up to '
            f'{np.abs(value.astype(int) - frozen).max()} grey levels')
