"""PageParser: the pipeline logic above the engines. Which stages a config builds and runs, in which order, how numbered
sections pair line croppers with OCRs, what PageParser itself does with the result (confidences, filtering), and full
runs of production-like pipelines.

What each stage does with its own config options is tested with the stage in test/engines."""
import configparser

import numpy as np
import pytest

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr import page_parser as pp
from pero_ocr.layout_engines.naive_sorter import NaiveRegionSorter
from pero_ocr.layout_engines.smart_sorter import SmartRegionSorter
from pero_ocr.layout_engines.transformer_sorter import TransformerRegionSorter
from test import tiny_models
from test.configs import section
from test.synthetic import page_image

PAGE = page_image(3)


def make_config(text: str, **paths) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_string(text.format(**paths))
    return config


LAYOUT = """
[LAYOUT_PARSER]
METHOD = LAYOUT_CNN
MODEL_PATH = {parsenet}
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
"""

CROPPER = """
[LINE_CROPPER{suffix}]
INTERP = 2
LINE_SCALE = 1.0
LINE_HEIGHT = 40
"""

OCR = """
[OCR{suffix}]
METHOD = {method}
OCR_JSON = {json}
USE_CPU = yes
"""

LM_DECODER = """
[DECODER]
TYPE = FAST-LOG-RAW
BEAM_SIZE = 8
LM = ./lm/lm.pt
LM_SCALE = 0.4
INSERTION_BONUS = 1.0
USE_CPU = yes
"""


def full_config(parsenet, ocr_json, method='pytorch_ocr', extra='', page_parser=''):
    text = ('[PAGE_PARSER]\nRUN_LAYOUT_PARSER = yes\nRUN_LINE_CROPPER = yes\nRUN_OCR = yes\n' + page_parser + LAYOUT +
            CROPPER.replace('{suffix}', '') + OCR.replace('{suffix}', '').replace('{method}', method) + extra)
    return make_config(text, parsenet=parsenet, json=ocr_json)


def new_page():
    return PageLayout(id='page', page_size=PAGE.shape[:2])


def texts(layout):
    return [line.transcription for line in layout.lines_iterator()]


@pytest.fixture(scope='module')
def ctc_parser(parsenet_path, ctc_json, cpu):
    return pp.PageParser(full_config(parsenet_path, ctc_json), device=cpu)


# --- which stages run ---------------------------------------------------------------------------------------------

def test_empty_config_raises(cpu):
    with pytest.raises(ValueError):
        pp.PageParser(configparser.ConfigParser(), device=cpu)


def test_no_stages(cpu):
    parser = pp.PageParser(make_config('[PAGE_PARSER]\nRUN_LAYOUT_PARSER = no\n'), device=cpu)
    layout = parser.process_page(PAGE, new_page())
    assert layout.regions == []
    assert not parser.provides_ctc_logits


def test_run_flags_switch_configured_stages_off(parsenet_path, ctc_json, cpu):
    """Sections that are present but whose RUN_* flag is off are neither built nor run."""
    config = full_config(parsenet_path, ctc_json, extra=LM_DECODER)
    for flag in ('RUN_LAYOUT_PARSER', 'RUN_LINE_CROPPER', 'RUN_OCR'):
        config['PAGE_PARSER'][flag] = 'no'
    parser = pp.PageParser(config, device=cpu)
    assert (parser.layout_parsers, parser.line_croppers, parser.ocrs, parser.decoder) == ({}, {}, {}, None)
    assert parser.process_page(PAGE, new_page()).regions == []


def test_full_pipeline_ctc(ctc_parser):
    layout = ctc_parser.process_page(PAGE, new_page())
    lines = list(layout.lines_iterator())
    assert len(lines) == 3
    for line in lines:
        assert line.transcription == 'a'
        assert line.crop.shape[0] == 40
        assert line.logits is not None and line.characters is not None
        assert 0 <= line.transcription_confidence <= 1
    assert layout.confidence is not None
    assert ctc_parser.provides_ctc_logits


def test_full_pipeline_transformer(parsenet_path, transformer_json, cpu):
    parser = pp.PageParser(full_config(parsenet_path, transformer_json, method='pytorch_ocr-transformer'), device=cpu)
    layout = parser.process_page(PAGE, new_page())
    assert texts(layout) == ['a'] * 3


def test_ocr_on_existing_layout(cpu, ctc_json):
    """OCR-only config (no layout parser) on a layout loaded from PAGE XML."""
    text = '[PAGE_PARSER]\nRUN_LINE_CROPPER = yes\nRUN_OCR = yes\n' + CROPPER.replace('{suffix}', '') + \
           OCR.replace('{suffix}', '').replace('{method}', 'pytorch_ocr')
    parser = pp.PageParser(make_config(text, json=ctc_json), device=cpu)
    layout = new_page()
    region = RegionLayout('r1', np.array([[0, 0], [800, 0], [800, 600], [0, 600]]))
    region.lines = [TextLine(id='r1-l1', baseline=np.array([[100.0, 140.0], [700.0, 140.0]]), heights=[40.0, 10.0],
                             polygon=np.array([[100, 100], [700, 100], [700, 150], [100, 150]]))]
    layout.regions = [region]
    layout = parser.process_page(PAGE, layout)
    assert texts(layout) == ['a']


# --- sections -----------------------------------------------------------------------------------------------------

def test_numbered_sections_pair_croppers_with_ocrs(cpu, ctc_json, tmp_path):
    """LINE_CROPPER_<n> and OCR_<n> run together, section number by section number: here every category of lines
    gets its own cropper and OCR model."""
    embed_json = tiny_models.write_ctc(str(tmp_path), embed_num=4, extra={'embed_id': 2})
    text = '[PAGE_PARSER]\nRUN_LINE_CROPPER = yes\nRUN_OCR = yes\n'
    for n, (json_path, category) in enumerate([(ctc_json, 'text'), (embed_json, 'title')]):
        text += CROPPER.replace('{suffix}', f'_{n}') + f'CATEGORIES = ["{category}"]\n'
        text += OCR.replace('{suffix}', f'_{n}').replace('{method}', 'pytorch_ocr').replace('{json}', json_path) + \
            f'CATEGORIES = ["{category}"]\n'
    parser = pp.PageParser(make_config(text), device=cpu)
    assert sorted(parser.ocrs) == ['0', '1']

    layout = new_page()
    region = RegionLayout('r1', np.array([[0, 0], [800, 0], [800, 600], [0, 600]]))
    for i, (top, category) in enumerate([(100, 'text'), (200, 'title')]):
        y = top + 40.0
        region.lines.append(TextLine(id=f'r1-l{i}', category=category,
                                     baseline=np.array([[100.0, y], [700.0, y]]), heights=[40.0, 10.0],
                                     polygon=np.array([[100, top], [700, top], [700, y + 10], [100, y + 10]])))
    layout.regions = [region]
    layout = parser.process_page(PAGE, layout)
    assert texts(layout) == ['a', 'c']


def test_section_names(cpu):
    """[X] and [X_<n>] are all built; [X] gets the key '-1', the others their number."""
    config = make_config('[PAGE_PARSER]\nRUN_LINE_CROPPER = yes\n' + ''.join(
        CROPPER.replace('{suffix}', s) for s in ['', '_2', '_10', '_1']))
    parser = pp.PageParser(config, device=cpu)
    assert set(parser.line_croppers) == {'-1', '1', '2', '10'}


def ordered_layout_config(parsenet_path, first, second):
    """LAYOUT_PARSER_<first> makes one whole-page region, LAYOUT_PARSER_<second> detects lines into existing regions:
    only in this order do the lines end up on the page."""
    whole_page = f'\n[LAYOUT_PARSER_{first}]\nMETHOD = REGION_WHOLE_PAGE\n'
    lines = LAYOUT.replace('[LAYOUT_PARSER]', f'[LAYOUT_PARSER_{second}]').replace('DETECT_REGIONS = yes',
                                                                                  'DETECT_REGIONS = no')
    return make_config('[PAGE_PARSER]\nRUN_LAYOUT_PARSER = yes\n' + lines + whole_page, parsenet=parsenet_path)


def test_sections_run_in_order(parsenet_path, cpu):
    parser = pp.PageParser(ordered_layout_config(parsenet_path, 1, 2), device=cpu)
    assert len(list(parser.process_page(PAGE, new_page()).lines_iterator())) == 3


@pytest.mark.xfail(reason='PageParser.process_page sorts sections by their string keys, so LAYOUT_PARSER_10 runs '
                          'before LAYOUT_PARSER_2 (likewise LINE_CROPPER_<n> and OCR_<n>)', strict=True)
def test_sections_run_in_numeric_order(parsenet_path, cpu):
    parser = pp.PageParser(ordered_layout_config(parsenet_path, 2, 10), device=cpu)
    assert len(list(parser.process_page(PAGE, new_page()).lines_iterator())) == 3


def test_relative_paths_resolve_against_config_path(tmp_path, cpu):
    """PageParser hands its config_path to every stage, so the model paths in an engine config are relative to it."""
    model_dir = tmp_path / 'engine'
    tiny_models.write_ctc(str(model_dir / 'ocr'))
    tiny_models.write_parsenet(str(model_dir / 'layout'))
    tiny_models.write_lm(str(model_dir / 'lm'))
    config = full_config('layout/parsenet.pt', 'ocr/ocr_engine.json', extra=LM_DECODER)
    config['PAGE_PARSER']['RUN_DECODER'] = 'yes'
    parser = pp.PageParser(config, device=cpu, config_path=str(model_dir))
    assert texts(parser.process_page(PAGE, new_page())) == ['a'] * 3


# --- the layout parser methods ------------------------------------------------------------------------------------

@pytest.mark.parametrize('method, engine_class', [
    ('REGION_WHOLE_PAGE', pp.WholePageRegion),
    ('LAYOUT_CNN', pp.LayoutExtractor),
    ('LAYOUT_YOLO', pp.LayoutExtractorYolo),
    ('LINE_POSTPROCESSING', pp.LinePostprocessor),
    ('LAYOUT_POSTPROCESSING', pp.LayoutPostprocessor),
    ('REGION_SORTER_NAIVE', NaiveRegionSorter),
    ('REGION_SORTER_SMART', SmartRegionSorter),
    ('REGION_SORTER_TRANSFORMER', TransformerRegionSorter),
])
def test_layout_parser_methods(method, engine_class, parsenet_path, region_sorter_path, fake_yolo, cpu):
    cnn = make_config(LAYOUT, parsenet=parsenet_path)['LAYOUT_PARSER']
    options = {
        'LAYOUT_CNN': {key.upper(): value for key, value in cnn.items()},
        'LAYOUT_YOLO': dict(MODEL_PATH='yolo.pt', DETECTION_THRESHOLD=0.3, USE_CPU='yes'),
        'LINE_POSTPROCESSING': dict(STRETCH_LINES=0, RESAMPLE_LINES='no', HEIGHTS_FROM_REGIONS='no'),
        'LAYOUT_POSTPROCESSING': dict(RETRACE_REGIONS='no'),
        'REGION_SORTER_TRANSFORMER': dict(MODEL_PATH=region_sorter_path, MAX_BBOX_COUNT=32, IMAGE_WIDTH=64,
                                          IMAGE_HEIGHT=64, USE_CPU='yes'),
    }.get(method, {})
    options['METHOD'] = method
    assert isinstance(pp.layout_parser_factory(section(**options), cpu), engine_class)


def test_unknown_layout_method(cpu):
    with pytest.raises(ValueError, match='Unknown layout parser method'):
        pp.layout_parser_factory(section(METHOD='NOPE'), cpu)


# --- what PageParser does after the stages ------------------------------------------------------------------------

def test_decoder_runs_on_the_ocr_output(parsenet_path, ctc_json, cpu, tmp_path):
    """RUN_DECODER: the decoder re-decodes the OCR logits after all OCR sections, then confidences are recomputed."""
    tiny_models.write_lm(str(tmp_path / 'lm'))
    config = full_config(parsenet_path, ctc_json, extra=LM_DECODER)
    config['PAGE_PARSER']['RUN_DECODER'] = 'yes'
    parser = pp.PageParser(config, device=cpu, config_path=str(tmp_path))
    layout = parser.process_page(PAGE, new_page())
    assert texts(layout) == ['a'] * 3
    assert (parser.decoder.lines_examined, parser.decoder.lines_decoded) == (3, 3)
    assert all(line.transcription_confidence is not None for line in layout.lines_iterator())


@pytest.mark.parametrize('threshold, kept', [(0, 3), (0.5, 3), (1.1, 0)])
def test_filter_confident_lines_threshold(parsenet_path, ctc_json, cpu, threshold, kept):
    """FILTER_CONFIDENT_LINES_THRESHOLD removes lines with a confidence at or below it; 0 (and the default -1) keeps
    all lines."""
    config = full_config(parsenet_path, ctc_json, page_parser=f'FILTER_CONFIDENT_LINES_THRESHOLD = {threshold}\n')
    layout = pp.PageParser(config, device=cpu).process_page(PAGE, new_page())
    assert len(list(layout.lines_iterator())) == kept


def test_lines_without_confidence_are_never_filtered(cpu):
    """Lines that no OCR read (no confidence) stay, whatever the threshold."""
    parser = pp.PageParser(make_config('[PAGE_PARSER]\nFILTER_CONFIDENT_LINES_THRESHOLD = 0.9\n'), device=cpu)
    layout = new_page()
    region = RegionLayout('r1', np.array([[0, 0], [800, 0], [800, 600], [0, 600]]))
    region.lines = [TextLine(id='r1-l1', transcription='from PAGE XML')]
    layout.regions = [region]
    assert texts(parser.process_page(PAGE, layout)) == ['from PAGE XML']


# --- production-like pipelines ------------------------------------------------------------------------------------

def test_production_shape_with_language_model(parsenet_path, ctc_json, cpu, tmp_path):
    """The shape of the production LM engines: ParseNet, the smart region sorter, line cropper, CTC OCR and
    FAST-LOG-RAW decoding with a torchscript LM for the less confident lines."""
    tiny_models.write_lm(str(tmp_path / 'lm'))
    sorter = '\n[LAYOUT_PARSER_2]\nMETHOD = REGION_SORTER_SMART\n'
    decoder = LM_DECODER + 'CONFIDENCE_THRESHOLD = 0.66\nCARRY_H_OVER = yes\n'
    config = full_config(parsenet_path, ctc_json, extra=sorter + decoder)
    config['PAGE_PARSER']['RUN_DECODER'] = 'yes'
    parser = pp.PageParser(config, device=cpu, config_path=str(tmp_path))
    layout = parser.process_page(PAGE, new_page())
    assert texts(layout) == ['a'] * 3
    assert parser.decoder.lines_examined == 3 and layout.confidence is not None


def test_layout_cnn_followed_by_transformer_region_sorter(parsenet_path, region_sorter_path, cpu):
    """ParseNet finds regions and lines, the transformer sorter orders the regions (the layout.order engine)."""
    sorter = (f'\n[LAYOUT_PARSER_2]\nMETHOD = REGION_SORTER_TRANSFORMER\nMODEL_PATH = {region_sorter_path}\n'
              f'MAX_BBOX_COUNT = 32\nIMAGE_WIDTH = 64\nIMAGE_HEIGHT = 64\nUSE_CPU = yes\n')
    text = '[PAGE_PARSER]\nRUN_LAYOUT_PARSER = yes\n' + LAYOUT.replace('[LAYOUT_PARSER]', '[LAYOUT_PARSER_1]') + sorter
    parser = pp.PageParser(make_config(text, parsenet=parsenet_path), device=cpu)
    layout = parser.process_page(PAGE, new_page())
    assert len(layout.regions) >= 1 and len(list(layout.lines_iterator())) == 3


def test_layout_cnn_followed_by_yolo(parsenet_path, fake_yolo, cpu):
    """ParseNet finds the text, YOLO adds non-text regions (ultralytics replaced by test/fake_yolo.py)."""
    yolo = '\n[LAYOUT_PARSER_2]\nMETHOD = LAYOUT_YOLO\nMODEL_PATH = yolo.pt\nDETECTION_THRESHOLD = 0.3\nUSE_CPU = yes\n'
    text = '[PAGE_PARSER]\nRUN_LAYOUT_PARSER = yes\n' + LAYOUT.replace('[LAYOUT_PARSER]', '[LAYOUT_PARSER_1]') + yolo
    parser = pp.PageParser(make_config(text, parsenet=parsenet_path), device=cpu)
    layout = parser.process_page(PAGE, new_page())
    text_regions = [r for r in layout.regions if r.category in (None, 'text')]
    assert sum(len(r.lines) for r in text_regions) == 3
    assert [r.category for r in layout.regions[len(text_regions):]] == ['image', 'stamp', 'page_number']
    assert len({r.id for r in layout.regions}) == len(layout.regions)
