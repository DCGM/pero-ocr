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
from test.io.consistency.golden_io import golden_io_path
import numpy as np

from test.synthetic import draw_text

PARSE_FOLDER = os.path.join(REPO_DIR, 'user_scripts', 'parse_folder.py')
PAGES = {'page_a': ['bad cab', 'dab', 'cd'], 'page_b': ['abc', 'b d']}  # the text written on each page

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
LM_SCALE = {lm_scale}
USE_CPU = yes
{lm_line}"""


def write_engine(directory, transformer=False, layout=True, decoder=False, lm=False):
    os.makedirs(directory, exist_ok=True)
    tiny_models.write_parsenet(os.path.join(directory, 'layout'))
    if transformer:
        tiny_models.write_transformer(os.path.join(directory, 'ocr'))
    else:
        tiny_models.write_ctc(os.path.join(directory, 'ocr'))
    if lm:
        tiny_models.write_lm(os.path.join(directory, 'language_model'))
    config = os.path.join(directory, 'config.ini')
    with open(config, 'w') as f:
        f.write(ENGINE_CONFIG.format(method='pytorch_ocr-transformer' if transformer else 'pytorch_ocr',
                                     layout='yes' if layout else 'no', decoder='yes' if decoder else 'no',
                                     lm_scale='0.4' if lm else '0.0',
                                     lm_line='LM = ./language_model/lm.pt\nINSERTION_BONUS = 1.0\n' if lm else ''))
    return config


def write_images(directory):
    """The PAGES, written for the tiny models, as lossless png (jpeg would shift the colours that encode the text)."""
    os.makedirs(directory, exist_ok=True)
    for name, lines in PAGES.items():
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        for i, text in enumerate(lines):
            draw_text(image, 100, 100 + 80 * i, 130 + 80 * i, text)
        cv2.imwrite(os.path.join(directory, name + '.png'), image)
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


def assert_shape_within(produced, golden_io):
    assert set(produced) <= set(golden_io), set(produced) - set(golden_io)
    for path, attrs in produced.items():
        assert attrs <= golden_io[path], (path, attrs - golden_io[path])


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
    assert len(os.listdir(out['lines'])) == sum(len(lines) for lines in PAGES.values())


def test_page_xml_content(full_run):
    _, _, _, out, _ = full_run
    for name, text in PAGES.items():
        layout = PageLayout(file=os.path.join(out['xml'], name + '.xml'))
        assert layout.id == name
        assert layout.page_size == (600, 800)
        assert [line.transcription for line in layout.lines_iterator()] == text
        assert all(line.transcription_confidence > 0.99 for line in layout.lines_iterator())


def test_page_xml_has_the_golden_io_shape(full_run):
    """Same elements and attributes as the golden IO reference (minus features the run does not produce)."""
    _, _, _, out, _ = full_run
    assert_shape_within(signature(os.path.join(out['xml'], 'page_a.xml')),
                        signature(golden_io_path('page_xml/reference_page.2019.xml')))


def test_alto_has_the_golden_io_shape(full_run):
    _, _, _, out, _ = full_run
    assert_shape_within(signature(os.path.join(out['alto'], 'page_a.xml')),
                        signature(golden_io_path('alto/reference_page.v2.xml')))


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
    keys, texts = zip(*(line.split(' ', 1) for line in lines))
    assert all(key.startswith(('page_a-', 'page_b-')) and key.endswith('.jpg') for key in keys)
    assert sorted(texts) == sorted(text for page in PAGES.values() for text in page)


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
    for name, text in PAGES.items():
        source = PageLayout(file=os.path.join(out['xml'], name + '.xml'))
        layout = PageLayout(file=str(tmp_path / 'xml' / (name + '.xml')))
        assert [line.id for line in layout.lines_iterator()] == [line.id for line in source.lines_iterator()]
        assert [line.transcription for line in layout.lines_iterator()] == text


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
    assert [line.transcription for line in layout.lines_iterator()] == PAGES['page_a']


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


def test_full_run_with_language_model(tmp_path):
    """The shape of the production LM job: layout + CTC OCR + FAST-LOG-RAW decoding with a torchscript LM."""
    config = write_engine(str(tmp_path / 'engine'), decoder=True, lm=True)
    images = write_images(str(tmp_path / 'images'))
    result = run('-c', config, '-i', images, '--output-xml-path', str(tmp_path / 'xml'),
                 '--output-alto-path', str(tmp_path / 'alto'), '--output-logit-path', str(tmp_path / 'logits'))
    assert result.returncode == 0, result.stderr
    assert 'ERROR: Failed to process file' not in result.stdout, result.stdout
    for name, text in PAGES.items():
        layout = PageLayout(file=str(tmp_path / 'xml' / (name + '.xml')))
        assert [line.transcription for line in layout.lines_iterator()] == text
        assert (tmp_path / 'alto' / (name + '.xml')).exists()
        assert (tmp_path / 'logits' / (name + '.logits')).exists()


# --- outputs --------------------------------------------------------------------------------------------------------

def alto_lines(path):
    """[(line box, [(word, hpos, width)])] of an ALTO file, in document order."""
    root = ET.parse(path).getroot()
    lines = []
    for text_line in root.iter('{*}TextLine'):
        box = [float(text_line.get(k)) for k in ('HPOS', 'VPOS', 'WIDTH', 'HEIGHT')]
        words = [(s.get('CONTENT'), float(s.get('HPOS')), float(s.get('WIDTH'))) for s in text_line.iter('{*}String')]
        lines.append((box, words))
    return lines


def test_alto_words_lie_where_they_are_written(full_run):
    """ALTO word boxes come from aligning the text to the CTC logits and mapping them back to the page: every word
    in its line, in order, starting where it was written (draw_text: 24 px characters, 12 px gaps, from x = 100)."""
    _, _, _, out, _ = full_run
    lines = alto_lines(os.path.join(out['alto'], 'page_a.xml'))
    assert [[word for word, _, _ in words] for _, words in lines] == [text.split() for text in PAGES['page_a']]
    for (left, _, width, _), words in lines:
        starts = [hpos for _, hpos, _ in words]
        assert starts == sorted(starts)
        assert all(left - 2 <= hpos and hpos + w <= left + width + 2 for _, hpos, w in words)
    (_, words), = [line for line in lines if len(line[1]) == 2]  # 'bad cab': 'cab' is the 5th character
    assert abs(words[1][1] - (100 + 4 * 36)) <= 12


def test_line_export_to_lmdb(full_run, tmp_path):
    """--output-line-path with 'lmdb' in the name writes an LMDB of jpeg line crops keyed '<page>-<line id>.jpg'."""
    import lmdb
    _, config, images, _, _ = full_run
    path = str(tmp_path / 'lines.lmdb')
    assert run('-c', config, '-i', images, '--output-line-path', path).returncode == 0
    with lmdb.open(path, readonly=True, lock=False).begin() as txn:
        records = {key.decode(): cv2.imdecode(np.frombuffer(value, np.uint8), 1) for key, value in txn.cursor()}
    assert len(records) == sum(len(lines) for lines in PAGES.values())
    assert all(key.startswith(('page_a-', 'page_b-')) and img.shape[0] == 40 for key, img in records.items())


@pytest.mark.xfail(reason='parse_folder.py copies the store_true flags --output-render-order and '
                          '--output-render-category into the config even when they are not given (default False, '
                          'checked with "is not None"), so OUTPUT_RENDER_ORDER = yes in the config never takes effect',
                   strict=True)
def test_render_options_from_the_config(full_run, tmp_path):
    _, config, images, _, _ = full_run
    assert run('-c', config, '-i', images, '--output-render-path', str(tmp_path / 'flag'),
               '--output-render-order').returncode == 0
    render_order = os.path.join(os.path.dirname(config), 'render_order.ini')  # next to the models it refers to
    with open(config) as source, open(render_order, 'w') as f:
        f.write(source.read().replace('LOGGING_LEVEL = WARNING', 'LOGGING_LEVEL = WARNING\nOUTPUT_RENDER_ORDER = yes'))
    assert run('-c', render_order, '-i', images, '--output-render-path', str(tmp_path / 'config')).returncode == 0
    flag = cv2.imread(str(tmp_path / 'flag' / 'page_a.jpg'))
    configured = cv2.imread(str(tmp_path / 'config' / 'page_a.jpg'))
    assert np.array_equal(flag, configured)


def cropper_only_config(directory, categories=None):
    """A config that only crops lines (e.g. to export training lines from existing PAGE XML)."""
    path = os.path.join(directory, 'crop.ini')
    with open(path, 'w') as f:
        f.write('[PAGE_PARSER]\nRUN_LINE_CROPPER = yes\n\n[LINE_CROPPER]\nINTERP = 2\nLINE_SCALE = 1.0\nLINE_HEIGHT = 40\n'
                + (f'CATEGORIES = ["{categories}"]\n' if categories else ''))
    return path


def test_line_cropping_in_two_processes(full_run, tmp_path):
    """--process-count works for pipelines without a model (#96: with a model it crashes)."""
    _, _, images, out, _ = full_run
    result = run('-c', cropper_only_config(str(tmp_path)), '-i', images, '-x', out['xml'],
                 '--output-line-path', str(tmp_path / 'lines'), '--process-count', '2')
    assert result.returncode == 0, result.stderr
    assert len(os.listdir(tmp_path / 'lines')) == sum(len(lines) for lines in PAGES.values())


@pytest.mark.xfail(reason='--output-line-path writes line.crop of every line, which is None for lines the cropper '
                          'left out (its CATEGORIES), so the page fails after its PAGE XML was written', strict=True)
def test_line_export_skips_lines_without_a_crop(full_run, tmp_path):
    _, _, images, out, _ = full_run
    result = run('-c', cropper_only_config(str(tmp_path), categories='title'), '-i', images, '-x', out['xml'],
                 '--output-line-path', str(tmp_path / 'lines'))
    assert result.returncode == 0, result.stderr
    assert 'ERROR: Failed to process file' not in result.stdout, result.stdout
    assert os.listdir(tmp_path / 'lines') == []


# --- inputs ---------------------------------------------------------------------------------------------------------

def test_redecoding_stored_logits_without_images(full_run, tmp_path):
    """The re-decoding job: PAGE XML and logits in (-x, --input-logit-path), no images, decoder only."""
    _, _, _, out, _ = full_run
    engine = tmp_path / 'engine'
    write_engine(str(engine), layout=False, decoder=True)
    config = engine / 'config.ini'
    config.write_text(config.read_text().replace('RUN_LINE_CROPPER = yes', 'RUN_LINE_CROPPER = no')
                      .replace('RUN_OCR = yes', 'RUN_OCR = no'))
    result = run('-c', str(config), '-x', out['xml'], '--input-logit-path', out['logits'],
                 '--output-xml-path', str(tmp_path / 'xml'))
    assert result.returncode == 0, result.stderr
    assert 'ERROR: Failed to process file' not in result.stdout, result.stdout
    for name, text in PAGES.items():
        layout = PageLayout(file=str(tmp_path / 'xml' / (name + '.xml')))
        assert [line.transcription for line in layout.lines_iterator()] == text


@pytest.mark.xfail(reason='PageOCR.provides_ctc_logits is true for the transformer engine, so parse_folder.py does '
                          'not refuse ALTO export from transformer logits, whose word positions are wrong',
                   strict=True)
def test_transformer_engine_refuses_alto_without_ctc_logits(full_run, tmp_path):
    _, _, images, _, _ = full_run
    config = write_engine(str(tmp_path / 'engine'), transformer=True)
    result = run('-c', config, '-i', images, '--output-alto-path', str(tmp_path / 'alto'))
    assert result.returncode == 2
