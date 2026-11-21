"""The [DECODER] stage (page_decoder_factory and PageDecoder in pero_ocr.document_ocr.page_parser): the [DECODER]
config section (plus [OCR] OCR_JSON for the character set) and a page whose lines carry OCR logits in, the lines
re-decoded. The stage picks the decoder by TYPE, loads the LM, and does what is the same for every decoder: which lines
are decoded (CATEGORIES, CONFIDENCE_THRESHOLD), carrying the LM state over line ends (CARRY_H_OVER), failing lines,
the summary. The decoders and the language model themselves are tested in test/engines/test_decoders.py and
test/engines/test_lm.py.

The lines carry hand-made CTC frames (test/synthetic.py); the tiny LM prefers 'b' after 'a' and keeps its state over
line ends."""
import os

import numpy as np
import pytest
import torch
from scipy import sparse

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.decoding.decoders import BLANK_SYMBOL, GreedyDecoder
from pero_ocr.document_ocr.page_parser import PageDecoder, page_decoder_factory
from test import tiny_models
from test.configs import sections
from test.synthetic import A, AMBIGUOUS_AFTER_A, BLANK, C_OR_B, ctc_frames

CHARS = tiny_models.CHARACTERS
CPU = torch.device('cpu')
B_OR_C_AT_LINE_START = ctc_frames(C_OR_B, BLANK)


def make(ctc_json, device=CPU, config_path='', **options):
    return page_decoder_factory(sections(OCR={'OCR_JSON': ctc_json}, DECODER={'USE_CPU': 'yes', **options}), device,
                                config_path=config_path)


LM_DECODER = dict(TYPE='FAST-LOG-RAW', BEAM_SIZE=8, LM_SCALE=1.0)


def page_with_lines(*line_logits, categories=None):
    """A page whose lines carry the given log-prob frames as (sparse) logits, and the transcription '?'."""
    page = PageLayout(id='p', page_size=(100, 100))
    region = RegionLayout('r1', np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
    for i, logits in enumerate(line_logits):
        region.lines.append(TextLine(
            id=f'r1-l{i}', transcription='?', category=(categories or ['text'] * len(line_logits))[i],
            logits=None if logits is None else sparse.csc_matrix(logits.astype(np.float32)),
            characters=CHARS + [BLANK_SYMBOL]))
    page.regions = [region]
    return page


def texts(page):
    return [line.transcription for line in page.lines_iterator()]


# --- decoders from config -----------------------------------------------------------------------------------------

def test_greedy(ctc_json):
    decoder = make(ctc_json, TYPE='GREEDY')
    assert isinstance(decoder, PageDecoder)
    assert isinstance(decoder.decoder, GreedyDecoder)
    assert texts(decoder.process_page(page_with_lines(AMBIGUOUS_AFTER_A))) == ['ac']


def test_beam_search_without_lm(ctc_json):
    decoder = make(ctc_json, TYPE='FAST-LOG-RAW', BEAM_SIZE=4, LM_SCALE=0.0)
    assert decoder.decoder._lm is None
    assert texts(decoder.process_page(page_with_lines(AMBIGUOUS_AFTER_A))) == ['ac']


def test_beam_search_with_lm_relative_to_the_config(ctc_json, lm_path):
    """The production LM setup: FAST-LOG-RAW with a torchscript LM; the LM overrides a visually ambiguous 'c'."""
    decoder = make(ctc_json, config_path=os.path.dirname(os.path.dirname(lm_path)),
                   LM=os.path.join('lm', os.path.basename(lm_path)), **LM_DECODER)
    assert texts(decoder.process_page(page_with_lines(AMBIGUOUS_AFTER_A))) == ['ab']


def test_characters_come_from_the_ocr_json(tmp_path):
    """The decoder takes the OCR model's characters from [OCR] OCR_JSON: the same logits columns mean other letters
    for a model with another alphabet."""
    reversed_json = tiny_models.write_ctc(str(tmp_path), characters=CHARS[::-1])
    decoder = make(reversed_json, TYPE='GREEDY')
    assert texts(decoder.process_page(page_with_lines(AMBIGUOUS_AFTER_A))) == [CHARS[::-1][CHARS.index('a')] +
                                                                              CHARS[::-1][CHARS.index('c')]]


@pytest.mark.xfail(reason="page_decoder_factory reads the character set from config['OCR'], so a config with numbered "
                          "OCR sections fails with KeyError: 'OCR'", raises=KeyError, strict=True)
def test_character_set_from_a_numbered_ocr_section(ctc_json):
    decoder = page_decoder_factory(sections(OCR_1={'OCR_JSON': ctc_json}, DECODER={'USE_CPU': 'yes', 'TYPE': 'GREEDY'}),
                                   CPU)
    assert texts(decoder.process_page(page_with_lines(AMBIGUOUS_AFTER_A))) == ['ac']


def test_unknown_type(ctc_json):
    with pytest.raises(ValueError, match="Unknown decoder type: 'NOPE'"):
        make(ctc_json, TYPE='NOPE')


def test_use_cpu_overrides_the_device(ctc_json, lm_path):
    decoder = make(ctc_json, device=torch.device('cuda'), LM=lm_path, **LM_DECODER)
    assert decoder.decoder._lm._lm_device == CPU


# --- which lines are decoded --------------------------------------------------------------------------------------

@pytest.mark.parametrize('options, decoded', [
    ({}, 2),  # without a threshold every line is decoded
    ({'CONFIDENCE_THRESHOLD': 0.9}, 1),  # the clear line keeps its transcription
])
def test_confidence_threshold(ctc_json, lm_path, options, decoded):
    decoder = make(ctc_json, LM=lm_path, **LM_DECODER, **options)
    page = decoder.process_page(page_with_lines(ctc_frames(A, BLANK), AMBIGUOUS_AFTER_A))
    assert texts(page) == (['?', 'ab'] if decoded == 1 else ['a', 'ab'])
    assert (decoder.lines_examined, decoder.lines_decoded) == (2, decoded)


def test_categories(ctc_json):
    """Only lines of the CATEGORIES are decoded, 'text' by default; lines without a category are always decoded."""
    def decode(**options):
        decoder = make(ctc_json, TYPE='GREEDY', **options)
        return texts(decoder.process_page(page_with_lines(*[ctc_frames(A)] * 3, categories=['text', 'title', None])))

    assert decode() == ['a', '?', 'a']
    assert decode(CATEGORIES=['title']) == ['?', 'a', 'a']


@pytest.mark.parametrize('carry_h_over, expected', [('no', ['a', 'c']), ('yes', ['a', 'b'])])
def test_carry_h_over_lets_the_previous_line_guide_the_next(ctc_json, lm_path, carry_h_over, expected):
    decoder = make(ctc_json, LM=lm_path, CARRY_H_OVER=carry_h_over, **LM_DECODER)
    assert texts(decoder.process_page(page_with_lines(ctc_frames(A, BLANK), B_OR_C_AT_LINE_START))) == expected


def test_carry_h_over_continues_on_the_next_page(ctc_json, lm_path):
    """The state carries over to the next page processed, too: the caller decides the order of the pages
    (parse_folder.py sorts the images by name; in production, the services above pero-ocr)."""
    decoder = make(ctc_json, LM=lm_path, CARRY_H_OVER='yes', **LM_DECODER)
    decoder.process_page(page_with_lines(ctc_frames(A, BLANK)))
    assert texts(decoder.process_page(page_with_lines(B_OR_C_AT_LINE_START))) == ['b']


# --- around the decoder --------------------------------------------------------------------------------------------

def test_confident_lines_still_guide_the_next_line(ctc_json, lm_path):
    """A line skipped as confident still passes its transcription on to the next line with CARRY_H_OVER."""
    decoder = make(ctc_json, LM=lm_path, CONFIDENCE_THRESHOLD=0.9, CARRY_H_OVER='yes', **LM_DECODER)
    page = page_with_lines(ctc_frames(A, BLANK), B_OR_C_AT_LINE_START)
    page.regions[0].lines[0].transcription = 'a'
    assert texts(decoder.process_page(page)) == ['a', 'b']


def test_a_failing_line_does_not_stop_the_page(ctc_json, lm_path):
    decoder = make(ctc_json, LM=lm_path, **LM_DECODER)
    assert texts(decoder.process_page(page_with_lines(None, AMBIGUOUS_AFTER_A))) == ['?', 'ab']


def test_summary(ctc_json, lm_path):
    decoder = make(ctc_json, LM=lm_path, CONFIDENCE_THRESHOLD=0.9, **LM_DECODER)
    assert decoder.decoding_summary() == 'This PageDecoder has not processed a single line yet'
    decoder.process_page(page_with_lines(ctc_frames(A, BLANK)))
    assert decoder.decoding_summary() == 'Processed 1 lines, but none required actual decoding'
    decoder.process_page(page_with_lines(AMBIGUOUS_AFTER_A))
    assert decoder.decoding_summary().startswith('Ran on 2, decoded 1 lines (50.0 %)')
