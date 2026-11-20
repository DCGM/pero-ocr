"""Golden-file helpers: compare exporter output with committed reference files, or rewrite them with --update-golden."""
import difflib
import os

import lxml.etree as ET
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
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(actual)
        return
    if not os.path.exists(path):
        pytest.fail(f'Golden file {path} does not exist, create it with: pytest --update-golden')
    with open(path, encoding='utf-8') as f:
        expected = normalize_xml(f.read())
    if actual != expected:
        diff = ''.join(difflib.unified_diff(expected.splitlines(True), actual.splitlines(True),
                                            fromfile=f'golden/{name}', tofile='current output'))
        pytest.fail(f'Output differs from golden/{name}. If the change is intended, rerun with --update-golden '
                    f'and commit the updated file.\n{diff}')
