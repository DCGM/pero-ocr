"""PAGE XML export must keep producing the committed golden IO files in test/io/consistency/data/page_xml."""
import pytest

from test.io.consistency.golden_io import check_dir_matches_table, check_golden_io, names_in


@pytest.mark.parametrize('name', names_in('page_xml'))
def test_unchanged(name):
    check_golden_io(name)


def test_data_dir_matches_the_table():
    check_dir_matches_table('page_xml')
