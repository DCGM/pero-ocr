import json
import os
import platform
import subprocess

import numpy
import pytest
import torch

from test import tiny_models

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_MODELS = os.path.join(TEST_DIR, 'engines', 'consistency', 'data')
REPO_DIR = os.path.dirname(TEST_DIR)


def pytest_addoption(parser):
    parser.addoption('--update-golden-io', action='store_true', default=False,
                     help='Rewrite the golden IO files in test/io/consistency/data from the current code instead of comparing.')
    parser.addoption('--update-frozen-models', action='store_true', default=False,
                     help='Rewrite the frozen tiny models in test/engines/consistency/data from the current code. Only after a '
                          'deliberate break of the model format: this resets the backward-compatibility check.')


def write_generated_with(directory):
    """Record which code and library versions produced the committed test data in ``directory``."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()

    info = {
        'git_commit': git('rev-parse', '--short', 'HEAD'),
        'pero_ocr_uncommitted_changes': bool(git('status', '--porcelain', '--', 'pero_ocr')),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': numpy.__version__,
    }
    with open(os.path.join(directory, 'generated_with.json'), 'w') as f:
        json.dump(info, f, indent=1)
        f.write('\n')


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason='CUDA is not available')
    for item in items:
        if 'gpu' in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope='session')
def update_golden_io(request):
    return request.config.getoption('--update-golden-io')


@pytest.fixture(scope='session')
def update_frozen_models(request):
    return request.config.getoption('--update-frozen-models')


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
