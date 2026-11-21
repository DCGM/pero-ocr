"""PageParser: the pipeline logic above the engines. Which stages a config builds and runs, in which order, how numbered
sections pair line croppers with OCRs, what PageParser itself does with the result (confidences, filtering), and full
runs of production-like pipelines.

Which engine a config section builds and what the stage around it does are tested in test/stages; what each engine does
with its own options in test/engines."""
import configparser
import json

import numpy as np
import pytest

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr import page_parser as pp
from test import tiny_models
from test.synthetic import draw_text, page_image

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


# --- contracts between the stages ---------------------------------------------------------------------------------
# Each engine is frozen and tested on its own input; these check that what one stage hands over is what the next one
# can use.

def ocr_only_config(*ocrs, cropper_height=40, decoder=''):
    """[LINE_CROPPER] for all lines, then the given OCR sections (suffix, json, method, categories), and a decoder."""
    text = '[PAGE_PARSER]\nRUN_LINE_CROPPER = yes\nRUN_OCR = yes\n' + ('RUN_DECODER = yes\n' if decoder else '')
    text += CROPPER.replace('{suffix}', '').replace('LINE_HEIGHT = 40', f'LINE_HEIGHT = {cropper_height}')
    for suffix, json_path, method, categories in ocrs:
        text += OCR.replace('{suffix}', suffix).replace('{method}', method).replace('{json}', json_path)
        text += f'CATEGORIES = {json.dumps(categories)}\n' if categories else ''
    return make_config(text + decoder)


def written_page(*lines):
    """A page with the given (text, category) lines written for the tiny models, and its layout with those lines."""
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    layout = new_page()
    region = RegionLayout('r1', np.array([[0, 0], [800, 0], [800, 600], [0, 600]]))
    for i, (text, category) in enumerate(lines):
        top, bottom = 100 + 80 * i, 130 + 80 * i
        right = draw_text(image, 100, top, bottom, text)
        region.lines.append(TextLine(id=f'r1-l{i}', category=category, heights=[30.0, 8.0],
                                     baseline=np.array([[100.0, bottom], [float(right), bottom]]),
                                     polygon=np.array([[100, top], [right, top], [right, bottom + 8],
                                                       [100, bottom + 8]])))
    layout.regions = [region]
    return image, layout


def test_page_of_written_text(cpu, ctc_json):
    image, layout = written_page(('bad cab', 'text'), ('dab', 'text'))
    parser = pp.PageParser(ocr_only_config(('', ctc_json, 'pytorch_ocr', None)), device=cpu)
    assert texts(parser.process_page(image, layout)) == ['bad cab', 'dab']


@pytest.mark.xfail(reason='BaseEngineLineOCR.process_lines means to reject crops of the wrong height, but the check is '
                          'inverted and the ValueError is never raised, so a [LINE_CROPPER] LINE_HEIGHT that does not '
                          "match the OCR model's line_px_height fails with a numpy broadcast error",
                   raises=AssertionError, strict=True)
def test_crop_height_that_does_not_match_the_ocr_model(cpu, ctc_json):
    image, layout = written_page(('bad', 'text'))
    parser = pp.PageParser(ocr_only_config(('', ctc_json, 'pytorch_ocr', None), cropper_height=32), device=cpu)
    with pytest.raises(ValueError, match='Line height needs to be 40'):
        parser.process_page(image, layout)


@pytest.mark.xfail(reason='page_decoder_factory decodes every line with the characters of [OCR] OCR_JSON instead of '
                          'the characters of the OCR that read the line, so lines read by another OCR section with '
                          'another alphabet are decoded into the wrong letters', strict=True)
def test_decoder_uses_the_alphabet_of_the_ocr_that_read_the_line(cpu, ctc_json, tmp_path):
    """[OCR] reads the text lines, [OCR_1], a model with another alphabet, the titles; a greedy decoder over both must
    give back what each OCR read."""
    other_alphabet = tiny_models.write_ctc(str(tmp_path), characters=tiny_models.CHARACTERS[::-1])
    image, layout = written_page(('bad', 'text'), ('bad', 'title'))
    ocrs = [('', ctc_json, 'pytorch_ocr', ['text']), ('_1', other_alphabet, 'pytorch_ocr', ['title'])]
    read = texts(pp.PageParser(ocr_only_config(*ocrs), device=cpu).process_page(image, layout))
    assert read[0] != read[1]  # the same picture, read in two alphabets
    decoder = '\n[DECODER]\nTYPE = GREEDY\nUSE_CPU = yes\nCATEGORIES = ["text", "title"]\n'
    image, layout = written_page(('bad', 'text'), ('bad', 'title'))
    decoded = texts(pp.PageParser(ocr_only_config(*ocrs, decoder=decoder), device=cpu).process_page(image, layout))
    assert decoded == read


@pytest.mark.xfail(reason='PageOCR.provides_ctc_logits is true for the transformer engine too, whose logits are per '
                          'decoding step, not per CTC frame; parse_folder.py relies on it for the ALTO word positions '
                          'and the logits export', strict=True)
def test_transformer_only_engine_provides_no_ctc_logits(cpu, transformer_json):
    parser = pp.PageParser(ocr_only_config(('', transformer_json, 'pytorch_ocr-transformer', None)), device=cpu)
    assert not parser.provides_ctc_logits


def test_pages_in_a_row_come_out_as_if_processed_alone(parsenet_path, ctc_json, cpu, tmp_path):
    """One PageParser processes page after page (pero-worker, the API client, parse_folder.py): without CARRY_H_OVER,
    a page must come out the same whatever was processed before it. (Pages of very different text sizes do not, as
    the layout engine keeps its adapted downsample: #96, Weird behaviour, tested in test_layout_engine.py.)"""
    tiny_models.write_lm(str(tmp_path / 'lm'))
    config = full_config(parsenet_path, ctc_json, extra=LM_DECODER + 'CONFIDENCE_THRESHOLD = 0.66\n')
    config['PAGE_PARSER']['RUN_DECODER'] = 'yes'

    def result(layout):
        return [(line.id, line.transcription, round(line.transcription_confidence, 6),
                 tuple(np.round(line.baseline.ravel(), 3))) for line in layout.lines_iterator()]

    pages = [text_page(doubt=0.49), text_page(doubt=0.0), text_page(doubt=0.6)]
    alone = [result(pp.PageParser(config, device=cpu, config_path=str(tmp_path)).process_page(page, new_page()))
             for page in pages]
    parser = pp.PageParser(config, device=cpu, config_path=str(tmp_path))
    in_a_row = [result(parser.process_page(page, new_page())) for page in pages]
    assert in_a_row == alone
    assert [texts for texts in ([line[1] for line in page] for page in alone)] == \
        [['abcd', 'ab', 'bad cab'], ['abcd', 'ac', 'bad cab'], ['abcd', 'ab', 'bad cab']]


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

def text_page(doubt):
    """Three lines of text for the tiny models; the 'c' of the second line is drawn with the given doubt (a 'b' is
    almost as likely)."""
    page = np.zeros((600, 800, 3), dtype=np.uint8)
    draw_text(page, 100, 100, 130, 'abcd')
    draw_text(page, 100, 170, 200, 'ac', [0.0, doubt])
    draw_text(page, 100, 240, 270, 'bad cab')
    return page


@pytest.mark.parametrize('run_decoder, texts_out, decoded', [
    ('no', ['abcd', 'ac', 'bad cab'], None),  # the OCR alone reads the unsure character as 'c'
    ('yes', ['abcd', 'ab', 'bad cab'], 1),  # only the unsure line is decoded, and the LM prefers 'b' after 'a'
])
def test_production_shape_with_language_model(parsenet_path, ctc_json, cpu, tmp_path, run_decoder, texts_out, decoded):
    """The shape of the production LM engines: ParseNet, the smart region sorter, line cropper, CTC OCR and
    FAST-LOG-RAW decoding with a torchscript LM (LM_SCALE 0.4) for the lines below CONFIDENCE_THRESHOLD 0.66."""
    tiny_models.write_lm(str(tmp_path / 'lm'))
    sorter = '\n[LAYOUT_PARSER_2]\nMETHOD = REGION_SORTER_SMART\n'
    decoder = LM_DECODER + 'CONFIDENCE_THRESHOLD = 0.66\nCARRY_H_OVER = yes\n'
    config = full_config(parsenet_path, ctc_json, extra=sorter + decoder)
    config['PAGE_PARSER']['RUN_DECODER'] = run_decoder
    parser = pp.PageParser(config, device=cpu, config_path=str(tmp_path))
    layout = parser.process_page(text_page(doubt=0.49), new_page())
    assert texts(layout) == texts_out
    confidences = [line.transcription_confidence for line in layout.lines_iterator()]
    assert confidences[0] > 0.99 and confidences[1] < 0.66 and confidences[2] > 0.99
    if decoded is not None:
        assert (parser.decoder.lines_examined, parser.decoder.lines_decoded) == (3, decoded)


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
