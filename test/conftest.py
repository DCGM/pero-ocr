import os

import pytest
import torch

from test import tiny_models

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TEST_DIR)


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason='CUDA is not available')
    for item in items:
        if 'gpu' in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope='session')
def cpu():
    return torch.device('cpu')


@pytest.fixture(scope='session')
def tiny_model_dir(tmp_path_factory):
    """Tiny torchscript stand-ins built from the current code, shared by the whole session."""
    directory = str(tmp_path_factory.mktemp('tiny_models'))
    tiny_models.write_all(directory)
    return directory


@pytest.fixture(scope='session')
def ctc_json(tiny_model_dir):
    return os.path.join(tiny_model_dir, 'ctc', 'ocr_engine.json')


@pytest.fixture(scope='session')
def transformer_json(tiny_model_dir):
    return os.path.join(tiny_model_dir, 'transformer', 'ocr_engine.json')


@pytest.fixture(scope='session')
def parsenet_path(tiny_model_dir):
    return os.path.join(tiny_model_dir, 'parsenet', 'parsenet.pt')


@pytest.fixture(scope='session')
def lm_path(tiny_model_dir):
    return os.path.join(tiny_model_dir, 'lm', 'lm.pt')


@pytest.fixture(scope='session')
def region_sorter_path(tiny_model_dir):
    return os.path.join(tiny_model_dir, 'region_sorter', 'model-cuda.pt')


@pytest.fixture
def fake_yolo(monkeypatch):
    """ultralytics.YOLO replaced by test/fake_yolo.FakeYOLO for one test; works without ultralytics installed."""
    from test import fake_yolo
    return fake_yolo.install(monkeypatch)
