"""The OCR stage (PageOCR in pero_ocr.document_ocr.page_parser) as a black box: an [OCR] config section and a
PageLayout with line crops in, the lines transcribed. The OCR engines underneath are tested in test_ctc_engine.py and
test_transformer_engine.py.

The tiny OCR models (test/tiny_models.py) read every bright block of a crop as one character: the CTC model always
'a', the transformer 'a', 'b', 'c', ... in turn."""
import os

import numpy as np
import pytest
import torch

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr.page_parser import ocr_factory
from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR
from pero_ocr.ocr_engine.transformer_ocr_engine import TransformerEngineLineOCR
from test import tiny_models
from test.configs import section
from test.synthetic import line_image


def make(ocr_json, device=torch.device('cpu'), config_path='', **options):
    return ocr_factory(section(OCR_JSON=ocr_json, USE_CPU='yes', **options), device, config_path=config_path)


def page(*lines):
    layout = PageLayout(id='p', page_size=(600, 800))
    region = RegionLayout('r1', np.array([[0, 0], [800, 0], [800, 600], [0, 600]]))
    region.lines = list(lines)
    layout.regions = [region]
    return layout


def line(n_runs, line_id=None, category=None, **kwargs):
    return TextLine(id=line_id or f'l{n_runs}', crop=line_image(n_runs), category=category, **kwargs)


def texts(layout):
    return [line.transcription for line in layout.lines_iterator()]


# --- engines and results ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('method, engine_class', [
    (None, PytorchEngineLineOCR),
    ('pytorch_ocr', PytorchEngineLineOCR),
    ('pytorch_ocr-transformer', TransformerEngineLineOCR),
])
def test_method_selects_the_engine(ctc_json, transformer_json, method, engine_class):
    options = {} if method is None else {'METHOD': method}
    ocr = make(transformer_json if engine_class is TransformerEngineLineOCR else ctc_json, **options)
    assert isinstance(ocr.ocr_engine, engine_class) and ocr.provides_ctc_logits
    # the tiny CTC model reads every block as 'a', the tiny transformer reads 'a', 'b', 'c', ...
    expected = ['a', 'abc'] if engine_class is TransformerEngineLineOCR else ['a', 'aaa']
    assert texts(ocr.process_page(None, page(line(1), line(3)))) == expected


def test_lines_get_logits_characters_and_confidences(ctc_json):
    layout = make(ctc_json).process_page(None, page(line(2)))
    result, = layout.lines_iterator()
    assert result.transcription == 'aa'
    assert result.logits.shape[1] == len(result.characters) == len(tiny_models.CHARACTERS) + 1
    assert result.logit_coords is not None
    assert 0 < result.transcription_confidence <= 1 and len(result.character_confidences) == 2


def test_page_without_lines(ctc_json):
    assert make(ctc_json).process_page(None, page()).regions[0].lines == []


def test_missing_crop_raises(ctc_json):
    with pytest.raises(Exception, match='Missing crop in line l1'):
        make(ctc_json).process_page(None, page(TextLine(id='l1')))


def test_categories(ctc_json):
    """Only lines of the CATEGORIES are read; lines without a category pass every filter."""
    layout = make(ctc_json, CATEGORIES=['title']).process_page(
        None, page(line(1, 'text', 'text'), line(2, 'title', 'title'), line(3, 'none')))
    assert texts(layout) == [None, 'aa', 'aaa']


def test_lines_of_other_categories_need_no_crop(ctc_json):
    layout = make(ctc_json, CATEGORIES=['title']).process_page(
        None, page(TextLine(id='text', category='text'), line(2, 'title', 'title')))
    assert texts(layout) == [None, 'aa']


# --- output substitution ------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def substituting_json(tmp_path_factory):
    return tiny_models.write_ctc(str(tmp_path_factory.mktemp('substitution')),
                                 extra={'output_substitution_table': {'a': 'ALPHA'}})


@pytest.mark.parametrize('runs, options, expected', [
    ([1, 1], {}, ['ALPHA', 'ALPHA']),
    ([1, 2], {}, ['a', 'aa']),  # atomic by default: 'aa' is not in the table, so nothing on the page is substituted
    ([1, 2], {'SUBSTITUTE_OUTPUT_ATOMIC': 'no'}, ['ALPHA', 'aa']),
    ([1, 1], {'SUBSTITUTE_OUTPUT': 'no'}, ['a', 'a']),
])
def test_output_substitution(substituting_json, runs, options, expected):
    lines = [line(n, f'l{i}') for i, n in enumerate(runs)]
    layout = make(substituting_json, **options).process_page(None, page(*lines))
    assert texts(layout) == expected


# --- keeping better transcriptions --------------------------------------------------------------------------------

def test_update_transcription_by_confidence(ctc_json):
    """With UPDATE_TRANSCRIPTION_BY_CONFIDENCE, a line keeps its transcription when it is more confident than the new
    one, e.g. when a second OCR section reads lines that a first one already read."""
    ocr = make(ctc_json, UPDATE_TRANSCRIPTION_BY_CONFIDENCE='yes')
    new_confidence = make(ctc_json).process_page(None, page(line(1))).regions[0].lines[0].transcription_confidence
    sure = line(1, 'sure', transcription='kept', transcription_confidence=1.0)
    unsure = line(1, 'unsure', transcription='replaced', transcription_confidence=new_confidence / 2)
    unknown = line(1, 'unknown', transcription='replaced')
    assert texts(ocr.process_page(None, page(sure, unsure, unknown))) == ['kept', 'a', 'a']
    assert texts(make(ctc_json).process_page(None, page(line(1, transcription='old', transcription_confidence=1.0)))) \
        == ['a']  # without the option, the new transcription always wins


# --- config -------------------------------------------------------------------------------------------------------

def test_beam_search_settings_reach_the_transformer(transformer_json):
    ocr = make(transformer_json, METHOD='pytorch_ocr-transformer', BEAM_SIZE=3, BEAM_MERGING='cn')
    assert (ocr.ocr_engine.beam_size, ocr.ocr_engine.beam_merging) == (3, 'cn')
    assert texts(ocr.process_page(None, page(line(2)))) == ['ab']


def test_ocr_json_relative_to_the_config(ctc_json):
    ocr = make(os.path.basename(ctc_json), config_path=os.path.dirname(ctc_json))
    assert texts(ocr.process_page(None, page(line(1)))) == ['a']


def test_use_cpu_overrides_the_device(ctc_json):
    assert make(ctc_json, device=torch.device('cuda')).device == torch.device('cpu')
