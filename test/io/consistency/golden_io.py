"""Golden IO files: which files exist in data/, how pero-ocr produces each of them, and how they are compared.

GOLDEN_IO_FILES is the single list of golden IO files. data/regenerate.py writes every entry, the consistency tests
compare every entry, so the two cannot drift apart.
"""
import difflib
import os
import pickle
import shutil
import tempfile

import lxml.etree as ET
import numpy as np
import pytest

from pero_ocr.core.layout import ALTOVersion, PAGEVersion, create_ocr_processing_element
from test.io.reference_layout import build_edge_layout, build_empty_layout, build_reference_layout

GOLDEN_IO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FORMAT_DIRS = ['page_xml', 'alto', 'logits']
MASKED = 'MASKED'
# elements whose text changes on every run (timestamps)
VOLATILE_ELEMENTS = {'Created', 'LastChange', 'processingDateTime'}


# --- how each file is produced ------------------------------------------------------------------------------------

def _page_xml(build, **kwargs):
    return lambda: build().to_pagexml_string(**kwargs)


def _alto(build, **kwargs):
    return lambda: build().to_altoxml_string(**kwargs)


def _alto_with_page_uuid_and_processing_element():
    processing = create_ocr_processing_element(id='IdOcrCustom', software_creator_str='Creator',
                                               software_name_str='Name', software_version_str='v9.9.9',
                                               processing_datetime='2026-01-01T00:00:00')
    return build_reference_layout().to_altoxml_string(ocr_processing_element=processing,
                                                      page_uuid='0f1e2d3c-page-uuid')


def _logits_file(build):
    """save_logits(), the variant parse_folder.py writes (pickle protocol 4)."""
    def produce():
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'page.logits')
            build().save_logits(path, missing_line_logits_ok=True)
            with open(path, 'rb') as f:
                return f.read()
    return produce


def _logits_bytes(build):
    """save_logits_bytes(), pickled with HIGHEST_PROTOCOL."""
    return lambda: build().save_logits_bytes(missing_line_logits_ok=True)


GOLDEN_IO_FILES = {
    'page_xml/reference_page.2019.xml': _page_xml(build_reference_layout),
    'page_xml/reference_page.2013.xml': _page_xml(build_reference_layout, version=PAGEVersion.PAGE_2013_07_15),
    'page_xml/reference_page.2019.validate_id.xml': _page_xml(build_reference_layout, validate_id=True),
    'page_xml/edge_page.2019.xml': _page_xml(build_edge_layout),
    'page_xml/edge_page.2013.xml': _page_xml(build_edge_layout, version=PAGEVersion.PAGE_2013_07_15),
    'page_xml/empty_page.2019.xml': _page_xml(build_empty_layout),
    'alto/reference_page.v2.xml': _alto(build_reference_layout),
    'alto/reference_page.v4.xml': _alto(build_reference_layout, version=ALTOVersion.ALTO_v4_4),
    'alto/reference_page.v2.page_uuid.xml': _alto_with_page_uuid_and_processing_element,
    'alto/edge_page.v2.xml': _alto(build_edge_layout),
    'alto/edge_page.v4.xml': _alto(build_edge_layout, version=ALTOVersion.ALTO_v4_4),
    'alto/empty_page.v2.xml': _alto(build_empty_layout),
    'logits/reference_page.logits': _logits_file(build_reference_layout),
    'logits/reference_page.bytes.logits': _logits_bytes(build_reference_layout),
    'logits/edge_page.bytes.logits': _logits_bytes(build_edge_layout),
}


def names_in(format_dir):
    return sorted(name for name in GOLDEN_IO_FILES if name.startswith(format_dir + '/'))


def golden_io_path(name: str) -> str:
    return os.path.join(GOLDEN_IO_DIR, name)


def produce(name: str):
    """The current content of a golden IO file: normalized XML text, or pickle bytes."""
    content = GOLDEN_IO_FILES[name]()
    return normalize_xml(content) if name.endswith('.xml') else content


# --- writing ------------------------------------------------------------------------------------------------------

def write_all(directory: str = GOLDEN_IO_DIR) -> list:
    """Recreate all golden IO files in ``directory`` from the current code; files not in the table are removed."""
    for format_dir in FORMAT_DIRS:
        shutil.rmtree(os.path.join(directory, format_dir), ignore_errors=True)
    written = []
    for name in GOLDEN_IO_FILES:
        path = os.path.join(directory, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        content = produce(name)
        if isinstance(content, str):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(path, 'wb') as f:
                f.write(content)
        written.append(path)
    return written


# --- comparing ----------------------------------------------------------------------------------------------------

REGENERATE = 'python test/io/consistency/data/regenerate.py'


def normalize_xml(xml: str) -> str:
    parser = ET.XMLParser(remove_blank_text=True)
    root = ET.fromstring(xml.encode('utf-8'), parser)
    for el in root.iter():
        if isinstance(el.tag, str) and ET.QName(el).localname in VOLATILE_ELEMENTS:
            el.text = MASKED
    return ET.tostring(root, pretty_print=True, encoding='utf-8', xml_declaration=True).decode('utf-8')


def check_golden_io(name: str):
    """Fail if the current code no longer produces the committed golden IO file ``name``."""
    path = golden_io_path(name)
    if not os.path.exists(path):
        pytest.fail(f'data/{name} does not exist, create it with: {REGENERATE}')
    actual = produce(name)
    if name.endswith('.xml'):
        with open(path, encoding='utf-8') as f:
            expected = normalize_xml(f.read())
        if actual != expected:
            diff = ''.join(difflib.unified_diff(expected.splitlines(True), actual.splitlines(True),
                                                fromfile=f'data/{name}', tofile='current output'))
            pytest.fail(f'Output differs from data/{name}. If the change is intended, run {REGENERATE} '
                        f'and commit the updated files.\n{diff}')
    else:
        with open(path, 'rb') as f:
            expected = f.read()
        assert actual[:2] == expected[:2], f'pickle protocol changed: {actual[:2]!r} vs data/{name} {expected[:2]!r}'
        assert_logit_dicts_equal(pickle.loads(actual), pickle.loads(expected))


def check_dir_matches_table(format_dir: str):
    """Fail if data/<format_dir> holds files that are not in GOLDEN_IO_FILES, or misses some that are."""
    on_disk = sorted(f'{format_dir}/{f}' for f in os.listdir(os.path.join(GOLDEN_IO_DIR, format_dir)))
    assert on_disk == names_in(format_dir), f'data/{format_dir} is out of sync with GOLDEN_IO_FILES, run {REGENERATE}'


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
