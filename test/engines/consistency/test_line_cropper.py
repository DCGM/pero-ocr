"""The line cropper (EngineLineCropper) must still cut exactly the frozen crops from the frozen page (data/cropper).

The crops are the input of every OCR model, and the models were trained on crops made by the cropper of their time.
A change of a single grey level in some pixels already changes the recognized text on real pages (opencv 5 changed
cv2.remap this way: about 6 % of the pixels by up to 3 grey levels, and 0.1 to 0.5 % of the characters on real
pages), so crops are compared exactly. The cropper has no model; the general tests of what it can do are in
test/engines/test_line_cropper.py.

The frozen data and how it is compared: frozen_engines.py and README.md in this directory."""
from test.engines.consistency.frozen_engines import check


def test_frozen_output_unchanged():
    check('cropper')
