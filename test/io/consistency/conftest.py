import pytest

from test.conftest import write_generated_with
from test.io.consistency.golden_io import GOLDEN_IO_DIR


@pytest.fixture(scope='session', autouse=True)
def record_golden_io_provenance(update_golden_io):
    """With --update-golden-io, record which code and versions produced the golden IO files."""
    yield
    if update_golden_io:
        write_generated_with(GOLDEN_IO_DIR)
