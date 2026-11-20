"""PageParser wiring: which stages run for a config and how they hand the PageLayout to each other."""
import configparser
import os

import numpy as np
import pytest

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr import page_parser as pp
from test import tiny_models
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


def full_config(parsenet, ocr_json, method='pytorch_ocr', extra=''):
    text = ('[PAGE_PARSER]\nRUN_LAYOUT_PARSER = yes\nRUN_LINE_CROPPER = yes\nRUN_OCR = yes\n' + LAYOUT +
            CROPPER.replace('{suffix}', '') + OCR.replace('{suffix}', '').replace('{method}', method) + extra)
    return make_config(text, parsenet=parsenet, json=ocr_json)


def new_page():
    return PageLayout(id='page', page_size=PAGE.shape[:2])


@pytest.fixture(scope='module')
def ctc_parser(parsenet_path, ctc_json, cpu):
    return pp.PageParser(full_config(parsenet_path, ctc_json), device=cpu)


def test_empty_config_raises(cpu):
    with pytest.raises(ValueError):
        pp.PageParser(configparser.ConfigParser(), device=cpu)


def test_no_stages(cpu):
    parser = pp.PageParser(make_config('[PAGE_PARSER]\nRUN_LAYOUT_PARSER = no\n'), device=cpu)
    layout = parser.process_page(PAGE, new_page())
    assert layout.regions == []
    assert not parser.provides_ctc_logits


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
    assert [line.transcription for line in layout.lines_iterator()] == ['a'] * 3


def test_ids_follow_region_line_scheme(ctc_parser):
    layout = ctc_parser.process_page(PAGE, new_page())
    for region in layout.regions:
        assert region.id.startswith('r')
        for line in region.lines:
            assert line.id.startswith(region.id + '-l')


def test_ocr_on_existing_layout(cpu, ctc_json):
    """OCR-only config (RUN_LAYOUT_PARSER = no) on a layout loaded from PAGE XML."""
    text = '[PAGE_PARSER]\nRUN_LINE_CROPPER = yes\nRUN_OCR = yes\n' + CROPPER.replace('{suffix}', '') + \
           OCR.replace('{suffix}', '').replace('{method}', 'pytorch_ocr')
    parser = pp.PageParser(make_config(text, json=ctc_json), device=cpu)
    layout = new_page()
    region = RegionLayout('r1', np.array([[0, 0], [800, 0], [800, 600], [0, 600]]))
    region.lines = [TextLine(id='r1-l1', baseline=np.array([[100.0, 140.0], [700.0, 140.0]]), heights=[40.0, 10.0],
                             polygon=np.array([[100, 100], [700, 100], [700, 150], [100, 150]]))]
    layout.regions = [region]
    layout = parser.process_page(PAGE, layout)
    assert layout.regions[0].lines[0].transcription == 'a'


def test_numbered_sections_with_categories(cpu, ctc_json, tmp_path):
    """OCR_<n> sections process only lines of their CATEGORIES, each with its own LINE_CROPPER_<n>."""
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
    assert [line.transcription for line in layout.lines_iterator()] == ['a', 'c']


@pytest.mark.parametrize('decoder', ['TYPE = GREEDY', 'TYPE = FAST-LOG-RAW\nBEAM_SIZE = 4\nLM_SCALE = 0.0'])
def test_decoder_without_lm(parsenet_path, ctc_json, cpu, decoder):
    config = full_config(parsenet_path, ctc_json, extra=f'\n[DECODER]\n{decoder}\nUSE_CPU = yes\n')
    config['PAGE_PARSER']['RUN_DECODER'] = 'yes'
    parser = pp.PageParser(config, device=cpu, config_path='')
    layout = parser.process_page(PAGE, new_page())
    assert [line.transcription for line in layout.lines_iterator()] == ['a'] * 3
    assert parser.decoder.lines_decoded == 3


def test_filter_confident_lines(ctc_parser):
    layout = ctc_parser.process_page(PAGE, new_page())
    ctc_parser.filter_confident_lines_threshold = 1.1
    try:
        layout = ctc_parser.filter_confident_lines(layout)
    finally:
        ctc_parser.filter_confident_lines_threshold = -1
    assert list(layout.lines_iterator()) == []


@pytest.mark.parametrize('runs, atomic, expected', [
    ([1, 1], 'yes', ['ALPHA', 'ALPHA']),
    ([1, 2], 'yes', ['a', 'aa']),  # 'aa' is not in the table: nothing on the page is substituted
    ([1, 2], 'no', ['ALPHA', 'aa']),
])
def test_output_substitution(cpu, tmp_path, runs, atomic, expected):
    from test.synthetic import line_image
    json_path = tiny_models.write_ctc(str(tmp_path), extra={'output_substitution_table': {'a': 'ALPHA'}})
    text = '[PAGE_PARSER]\nRUN_OCR = yes\n' + OCR.replace('{suffix}', '').replace('{method}', 'pytorch_ocr') + \
        f'SUBSTITUTE_OUTPUT_ATOMIC = {atomic}\n'
    parser = pp.PageParser(make_config(text, json=json_path), device=cpu)
    layout = new_page()
    region = RegionLayout('r1', np.array([[0, 0], [10, 0], [10, 10]]))
    region.lines = [TextLine(id=f'l{i}', crop=line_image(n)) for i, n in enumerate(runs)]
    layout.regions = [region]
    layout = parser.process_page(None, layout)
    assert [line.transcription for line in layout.lines_iterator()] == expected


def test_missing_crop_raises(cpu, ctc_json):
    text = '[PAGE_PARSER]\nRUN_OCR = yes\n' + OCR.replace('{suffix}', '').replace('{method}', 'pytorch_ocr')
    parser = pp.PageParser(make_config(text, json=ctc_json), device=cpu)
    layout = new_page()
    region = RegionLayout('r1', np.array([[0, 0], [10, 0], [10, 10]]))
    region.lines = [TextLine(id='l1')]
    layout.regions = [region]
    with pytest.raises(Exception, match='Missing crop'):
        parser.process_page(None, layout)


def test_relative_paths_resolve_against_config_path(tmp_path, cpu):
    model_dir = tmp_path / 'engine'
    tiny_models.write_ctc(str(model_dir / 'ocr'))
    tiny_models.write_parsenet(str(model_dir / 'layout'))
    config = full_config('layout/parsenet.pt', 'ocr/ocr_engine.json')
    parser = pp.PageParser(config, device=cpu, config_path=str(model_dir))
    assert len(list(parser.process_page(PAGE, new_page()).lines_iterator())) == 3


@pytest.mark.parametrize('method', [
    'REGION_WHOLE_PAGE',
    pytest.param('REGION_SORTER_NAIVE', marks=pytest.mark.xfail(
        reason='NaiveRegionSorter.sort_regions clusters columns by y_min instead of x_min', strict=True)),
    'REGION_SORTER_SMART',
    'LAYOUT_POSTPROCESSING',
])
def test_cpu_only_layout_parsers(method, cpu):
    config = make_config(f'[S]\nMETHOD = {method}\nRETRACE_REGIONS = yes\n')
    parser = pp.layout_parser_factory(config['S'], cpu)
    layout = new_page()
    for i, x in enumerate([500, 20]):
        region = RegionLayout(f'r{i}', np.array([[x, 50], [x + 200, 50], [x + 200, 400], [x, 400]]))
        region.lines = [TextLine(id=f'r{i}-l0', baseline=np.array([[x + 10.0, 100.0], [x + 190.0, 100.0]]),
                                 heights=[20.0, 5.0],
                                 polygon=np.array([[x + 10, 80], [x + 190, 80], [x + 190, 105], [x + 10, 105]]))]
        layout.regions.append(region)
    layout = parser.process_page(PAGE, layout)
    if method == 'REGION_WHOLE_PAGE':
        assert [r.id for r in layout.regions] == ['r1']
    elif method.startswith('REGION_SORTER'):
        assert [r.id for r in layout.regions] == ['r1', 'r0']  # left column first
    else:
        assert len(layout.regions) == 2


def test_unknown_layout_method(cpu):
    config = make_config('[S]\nMETHOD = NOPE\n')
    with pytest.raises(ValueError, match='Unknown layout parser method'):
        pp.layout_parser_factory(config['S'], cpu)


def test_line_postprocessing(cpu):
    config = make_config('[S]\nMETHOD = LINE_POSTPROCESSING\nSTRETCH_LINES = max\nRESAMPLE_LINES = yes\n'
                         'HEIGHTS_FROM_REGIONS = no\n')
    parser = pp.layout_parser_factory(config['S'], cpu)
    layout = new_page()
    region = RegionLayout('r1', np.array([[50, 50], [750, 50], [750, 200], [50, 200]]))
    region.lines = [TextLine(id='r1-l0', baseline=np.array([[100.0, 100.0], [300.0, 100.0]]), heights=[20.0, 5.0],
                             polygon=np.array([[100, 80], [300, 80], [300, 105], [100, 105]]))]
    layout.regions = [region]
    layout = parser.process_page(PAGE, layout)
    baseline = layout.regions[0].lines[0].baseline
    assert baseline[:, 0].max() - baseline[:, 0].min() > 200  # stretched to the region


def test_init_config_sections_order(cpu):
    config = make_config('[PAGE_PARSER]\nRUN_LINE_CROPPER = yes\n' + ''.join(
        CROPPER.replace('{suffix}', s) for s in ['', '_2', '_10', '_1']))
    parser = pp.PageParser(config, device=cpu)
    assert set(parser.line_croppers) == {'-1', '1', '2', '10'}


LM_DECODER = """
[DECODER]
TYPE = FAST-LOG-RAW
BEAM_SIZE = 8
LM = ./lm/lm.pt
LM_SCALE = 0.4
INSERTION_BONUS = 1.0
USE_CPU = yes
"""


@pytest.mark.parametrize('extra, decoded', [
    ('', 3),                                 # without a threshold every line is decoded
    ('CARRY_H_OVER = yes\n', 3),
    ('CONFIDENCE_THRESHOLD = 0.66\n', 0),    # the tiny OCR is sure of every line, so none needs decoding
])
def test_decoder_with_language_model(parsenet_path, ctc_json, cpu, tmp_path, extra, decoded):
    """The shape of the production LM config: layout + CTC OCR + FAST-LOG-RAW with a torchscript LM."""
    from pero_ocr.decoding.lm_wrapper import LMWrapper
    tiny_models.write_lm(str(tmp_path / 'lm'))
    config = full_config(parsenet_path, ctc_json, extra=LM_DECODER + extra)
    config['PAGE_PARSER']['RUN_DECODER'] = 'yes'
    parser = pp.PageParser(config, device=cpu, config_path=str(tmp_path))
    assert isinstance(parser.decoder.decoder._lm, LMWrapper)
    layout = parser.process_page(PAGE, new_page())
    assert [line.transcription for line in layout.lines_iterator()] == ['a'] * 3
    assert (parser.decoder.lines_examined, parser.decoder.lines_decoded) == (3, decoded)
