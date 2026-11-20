"""Golden-file helpers: compare exporter output with the committed files in data/, or rewrite them with --update-golden."""
import difflib
import os

import lxml.etree as ET
import numpy as np
import pytest

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MASKED = 'MASKED'
# elements whose text changes on every run (timestamps)
VOLATILE_ELEMENTS = {'Created', 'LastChange', 'processingDateTime'}


def normalize_xml(xml: str) -> str:
    parser = ET.XMLParser(remove_blank_text=True)
    root = ET.fromstring(xml.encode('utf-8'), parser)
    for el in root.iter():
        if isinstance(el.tag, str) and ET.QName(el).localname in VOLATILE_ELEMENTS:
            el.text = MASKED
    return ET.tostring(root, pretty_print=True, encoding='utf-8', xml_declaration=True).decode('utf-8')


def golden_path(name: str) -> str:
    return os.path.join(GOLDEN_DIR, name)


def check_golden_xml(name: str, actual_xml: str, update: bool):
    path = golden_path(name)
    actual = normalize_xml(actual_xml)
    if update:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(actual)
        return
    if not os.path.exists(path):
        pytest.fail(f'Golden file {path} does not exist, create it with: pytest test/io/consistency --update-golden')
    with open(path, encoding='utf-8') as f:
        expected = normalize_xml(f.read())
    if actual != expected:
        diff = ''.join(difflib.unified_diff(expected.splitlines(True), actual.splitlines(True),
                                            fromfile=f'data/{name}', tofile='current output'))
        pytest.fail(f'Output differs from data/{name}. If the change is intended, rerun with --update-golden '
                    f'and commit the updated file.\n{diff}')


META_KEYS = {'line_characters', 'logit_coords'}


def assert_logit_dicts_equal(actual: dict, expected: dict):
    assert set(actual) == set(expected)
    assert actual['line_characters'] == expected['line_characters']
    assert actual['logit_coords'] == expected['logit_coords']
    for key in set(expected) - META_KEYS:
        a, e = actual[key], expected[key]
        if e is None:
            assert a is None, key
            continue
        assert type(a) is type(e), key
        assert a.dtype == e.dtype, key
        assert a.shape == e.shape, key
        np.testing.assert_array_equal(a.toarray(), e.toarray())
