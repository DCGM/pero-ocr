import pytest

from test.conftest import write_generated_with
from test.io.consistency.golden import GOLDEN_DIR


@pytest.fixture(scope='session', autouse=True)
def record_golden_provenance(update_golden):
    """With --update-golden, record which code and versions produced the golden files."""
    yield
    if update_golden:
        write_generated_with(GOLDEN_DIR)
