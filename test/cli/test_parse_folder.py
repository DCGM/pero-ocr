"""End-to-end runs of user_scripts/parse_folder.py with tiny models on CPU.

parse_folder.py catches per-page exceptions and still exits 0, so the tests check the produced files, not only
the exit code."""
import os
import pickle
import subprocess
import sys

import cv2
import lxml.etree as ET
import pytest

from pero_ocr.core.layout import PageLayout
from test import tiny_models
from test.conftest import REPO_DIR
from test.io.consistency.golden import golden_path
from test.synthetic import page_image

PARSE_FOLDER = os.path.join(REPO_DIR, 'user_scripts', 'parse_folder.py')
PAGES = {'page_a': 3, 'page_b': 2}

ENGINE_CONFIG = """
[PAGE_PARSER]
RUN_LAYOUT_PARSER = {layout}
RUN_LINE_CROPPER = yes
RUN_OCR = yes
RUN_DECODER = {decoder}

[PARSE_FOLDER]
LOGGING_LEVEL = WARNING

[LAYOUT_PARSER_1]
METHOD = LAYOUT_CNN
MODEL_PATH = ./layout/parsenet.pt
USE_CPU = yes
DOWNSAMPLE = 4
DETECTION_THRESHOLD = 0.2
MAX_MEGAPIXELS = 5
DETECT_REGIONS = yes
DETECT_LINES = yes
DETECT_STRAIGHT_LINES_IN_REGIONS = no
MERGE_LINES = no
ADJUST_HEIGHTS = no
ADJUST_BASELINES = no
MULTI_ORIENTATION = no

[LAYOUT_PARSER_2]
METHOD = REGION_SORTER_SMART

[LINE_CROPPER]
INTERP = 2
LINE_SCALE = 1.0
LINE_HEIGHT = 40

[OCR]
METHOD = {method}
OCR_JSON = ./ocr/ocr_engine.json
USE_CPU = yes

[DECODER]
TYPE = FAST-LOG-RAW
BEAM_SIZE = 4
LM_SCALE = 0.0
USE_CPU = yes
"""


def write_engine(directory, transformer=False, layout=True, decoder=False):
    os.makedirs(directory, exist_ok=True)
    tiny_models.write_parsenet(os.path.join(directory, 'layout'))
    if transformer:
        tiny_models.write_transformer(os.path.join(directory, 'ocr'))
    else:
        tiny_models.write_ctc(os.path.join(directory, 'ocr'))
    config = os.path.join(directory, 'config.ini')
    with open(config, 'w') as f:
        f.write(ENGINE_CONFIG.format(method='pytorch_ocr-transformer' if transformer else 'pytorch_ocr',
                                     layout='yes' if layout else 'no', decoder='yes' if decoder else 'no'))
    return config


def write_images(directory):
    os.makedirs(directory, exist_ok=True)
    for name, n_lines in PAGES.items():
        cv2.imwrite(os.path.join(directory, name + '.jpg'), page_image(n_lines))
    return directory


def run(*args, cwd=None):
    result = subprocess.run([sys.executable, PARSE_FOLDER, '--device', 'cpu', *args], cwd=cwd,
                            capture_output=True, text=True, timeout=600,
                            env={**os.environ, 'PYTHONPATH': REPO_DIR})
    return result


def signature(xml_path):
    """{element path: attribute names seen on it}: the shape of the document, not its values."""
    root = ET.parse(xml_path).getroot()
    sig = {}

    def walk(el, path):
        p = path + '/' + ET.QName(el).localname
        sig.setdefault(p, set()).update(el.attrib)
        for child in el:
            if isinstance(child.tag, str):
                walk(child, p)
    walk(root, '')
    return sig


def assert_shape_within(produced, golden):
    assert set(produced) <= set(golden), set(produced) - set(golden)
    for path, attrs in produced.items():
        assert attrs <= golden[path], (path, attrs - golden[path])


@pytest.fixture(scope='module')
def full_run(tmp_path_factory):
    base = tmp_path_factory.mktemp('parse_folder')
    config = write_engine(str(base / 'engine'))
    images = write_images(str(base / 'images'))
    out = {k: str(base / k) for k in ['xml', 'alto', 'logits', 'render', 'lines']}
    out['transcriptions'] = str(base / 'transcriptions.txt')
    result = run('-c', config, '-i', images, '--output-xml-path', out['xml'], '--output-alto-path', out['alto'],
                 '--output-logit-path', out['logits'], '--output-render-path', out['render'],
                 '--output-line-path', out['lines'], '--output-transcriptions-file-path', out['transcriptions'])
    return base, config, images, out, result


pytestmark = pytest.mark.slow


def test_exit_code_and_no_page_errors(full_run):
    *_, result = full_run
    assert result.returncode == 0, result.stderr
    assert 'ERROR: Failed to process file' not in result.stdout, result.stdout


def test_one_output_of_each_kind_per_page(full_run):
    _, _, _, out, _ = full_run
    assert sorted(os.listdir(out['xml'])) == ['page_a.xml', 'page_b.xml']
    assert sorted(os.listdir(out['alto'])) == ['page_a.xml', 'page_b.xml']
    assert sorted(os.listdir(out['logits'])) == ['page_a.logits', 'page_b.logits']
    assert sorted(os.listdir(out['render'])) == ['page_a.jpg', 'page_b.jpg']
    assert len(os.listdir(out['lines'])) == sum(PAGES.values())


def test_page_xml_content(full_run):
    _, _, _, out, _ = full_run
    for name, n_lines in PAGES.items():
        layout = PageLayout(file=os.path.join(out['xml'], name + '.xml'))
        assert layout.id == name
        assert layout.page_size == (600, 800)
        lines = list(layout.lines_iterator())
        assert len(lines) == n_lines
        assert all(line.transcription == 'a' for line in lines)
        assert all(line.transcription_confidence is not None for line in lines)


def test_page_xml_has_the_golden_shape(full_run):
    """Same elements and attributes as the golden reference (minus features the run does not produce)."""
    _, _, _, out, _ = full_run
    assert_shape_within(signature(os.path.join(out['xml'], 'page_a.xml')),
                        signature(golden_path('page_xml/reference_page.2019.xml')))


def test_alto_has_the_golden_shape(full_run):
    _, _, _, out, _ = full_run
    assert_shape_within(signature(os.path.join(out['alto'], 'page_a.xml')),
                        signature(golden_path('alto/reference_page.v2.xml')))


def test_logits_match_page_xml(full_run):
    _, _, _, out, _ = full_run
    layout = PageLayout(file=os.path.join(out['xml'], 'page_a.xml'))
    with open(os.path.join(out['logits'], 'page_a.logits'), 'rb') as f:
        logits = pickle.load(f)
    line_ids = {line.id for line in layout.lines_iterator()}
    assert set(logits) == line_ids | {'line_characters', 'logit_coords'}
    assert all(logits['line_characters'][i] == tiny_models.CHARACTERS + ['​'] for i in line_ids)


def test_transcriptions_file(full_run):
    _, _, _, out, _ = full_run
    with open(out['transcriptions']) as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]
    assert len(lines) == sum(PAGES.values())
    for line in lines:
        key, text = line.split(' ', 1)
        assert key.startswith(('page_a-', 'page_b-')) and key.endswith('.jpg')
        assert text == 'a'


def test_skip_processed(full_run, tmp_path):
    _, config, images, out, _ = full_run
    xml_dir = tmp_path / 'xml'
    xml_dir.mkdir()
    (xml_dir / 'page_a.xml').write_bytes(open(os.path.join(out['xml'], 'page_a.xml'), 'rb').read())
    before = os.path.getmtime(xml_dir / 'page_a.xml')
    result = run('-c', config, '-i', images, '--output-xml-path', str(xml_dir), '-s')
    assert result.returncode == 0, result.stderr
    assert os.path.getmtime(xml_dir / 'page_a.xml') == before
    assert sorted(os.listdir(xml_dir)) == ['page_a.xml', 'page_b.xml']


@pytest.mark.xfail(reason='parse_folder.py divides by len(ids_to_process) for the average time, which is 0 when '
                          'every page was already processed', strict=True)
def test_skip_processed_when_everything_is_done(full_run):
    _, config, images, out, _ = full_run
    result = run('-c', config, '-i', images, '--output-xml-path', out['xml'], '-s')
    assert result.returncode == 0, result.stderr


def test_ocr_on_input_page_xml(full_run, tmp_path):
    """Layout from PAGE XML (-x), OCR only, with the transformer engine."""
    _, _, images, out, _ = full_run
    config = write_engine(str(tmp_path / 'engine'), transformer=True, layout=False)
    result = run('-c', config, '-i', images, '-x', out['xml'], '--output-xml-path', str(tmp_path / 'xml'))
    assert result.returncode == 0, result.stderr
    for name, n_lines in PAGES.items():
        source = PageLayout(file=os.path.join(out['xml'], name + '.xml'))
        layout = PageLayout(file=str(tmp_path / 'xml' / (name + '.xml')))
        assert [line.id for line in layout.lines_iterator()] == [line.id for line in source.lines_iterator()]
        assert [line.transcription for line in layout.lines_iterator()] == ['a'] * n_lines


def test_decoder_on_input_logits(full_run, tmp_path):
    """Re-decode stored logits (-x + --input-logit-path) without running the OCR again."""
    _, _, images, out, _ = full_run
    engine = tmp_path / 'engine'
    write_engine(str(engine), layout=False, decoder=True)
    config = engine / 'config.ini'
    config.write_text(config.read_text().replace('RUN_LINE_CROPPER = yes', 'RUN_LINE_CROPPER = no')
                      .replace('RUN_OCR = yes', 'RUN_OCR = no'))
    result = run('-c', str(config), '-i', images, '-x', out['xml'], '--input-logit-path', out['logits'],
                 '--output-xml-path', str(tmp_path / 'xml'))
    assert result.returncode == 0, result.stderr
    layout = PageLayout(file=str(tmp_path / 'xml' / 'page_a.xml'))
    assert [line.transcription for line in layout.lines_iterator()] == ['a'] * PAGES['page_a']


def test_unreadable_image_does_not_stop_the_run(tmp_path):
    config = write_engine(str(tmp_path / 'engine'))
    images = write_images(str(tmp_path / 'images'))
    (tmp_path / 'images' / 'broken.jpg').write_bytes(b'not an image')
    result = run('-c', config, '-i', images, '--output-xml-path', str(tmp_path / 'xml'))
    assert result.returncode == 0
    assert 'ERROR: Failed to process file broken.' in result.stdout
    assert sorted(os.listdir(tmp_path / 'xml')) == ['page_a.xml', 'page_b.xml']


def test_missing_config(tmp_path):
    result = run('-c', str(tmp_path / 'nope.ini'), '-i', str(tmp_path))
    assert result.returncode != 0
    assert 'Config file does not exist' in result.stdout
