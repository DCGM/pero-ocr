import pytest

from test import tiny_models
from test.conftest import FROZEN_MODELS, write_generated_with


@pytest.fixture(scope='session', autouse=True)
def regenerate_frozen_models(update_frozen_models):
    """With --update-frozen-models, rewrite the frozen tiny models before the engine tests use them."""
    if update_frozen_models:
        tiny_models.write_all(FROZEN_MODELS)
        write_generated_with(FROZEN_MODELS)
