"""A stand-in for ultralytics.YOLO, for the LAYOUT_YOLO engine (LayoutExtractorYolo, LayoutEngineYolo).

pero-ocr imports ultralytics only when a LAYOUT_YOLO stage is built, and uses a tiny part of it:

    net = YOLO(model_path).to(device)
    result = net(image, conf=threshold, verbose=False[, imgsz=size])[0]
    result.boxes.data   # tensor [n, 6]: x_min, y_min, x_max, y_max, confidence, class id
    result.names        # {class id: category name}

Unlike the tiny torchscript models, the YOLO model file is loaded by ultralytics, not by pero-ocr, so a stand-in model
file would test ultralytics. Instead, the fake replaces the YOLO class itself: it returns fixed DETECTIONS, drops those
below the confidence threshold as YOLO does, and records how pero-ocr called it. The tests then need neither
ultralytics nor a model, and check exactly pero-ocr's side: the config, the call, and how detections become regions.
"""
import sys
import types

import torch

NAMES = {0: 'image', 1: 'stamp', 2: 'page_number'}

# x_min, y_min, x_max, y_max, confidence, class id; on a 1000 x 800 page
DETECTIONS = [
    (100.0, 120.0, 500.0, 400.0, 0.91, 0),  # image
    (550.0, 60.0, 700.0, 160.0, 0.55, 1),  # stamp
    (380.0, 900.0, 420.0, 931.0, 0.80, 2),  # page_number
    (600.0, 600.0, 650.0, 650.0, 0.10, 1),  # a weak stamp, below the usual thresholds
]


class FakeBoxes:
    def __init__(self, data):
        self.data = data


class FakeResult:
    def __init__(self, data, names):
        self.boxes = FakeBoxes(data)
        self.names = names


class FakeYOLO:
    """ultralytics.YOLO: every instance is recorded in ``instances``, every call in ``calls``."""
    instances = []

    def __init__(self, model_path):
        self.model_path = model_path
        self.device = None
        self.calls = []
        FakeYOLO.instances.append(self)

    def to(self, device):
        self.device = device
        return self

    def __call__(self, image, conf=0.25, verbose=True, **kwargs):
        self.calls.append(dict(image=image, conf=conf, verbose=verbose, **kwargs))
        data = torch.tensor([d for d in DETECTIONS if d[4] >= conf], dtype=torch.float32).reshape(-1, 6)
        return [FakeResult(data, dict(NAMES))]


def install(monkeypatch):
    """Make ``import ultralytics`` and ``from ultralytics import YOLO`` give the fake, for one test."""
    FakeYOLO.instances = []
    monkeypatch.setitem(sys.modules, 'ultralytics', types.SimpleNamespace(YOLO=FakeYOLO))
    return FakeYOLO
