from __future__ import annotations
import sys
import os
import re
import logging
from typing import List

import music21 as music

from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.layout_engines.layout_helpers import split_page_layout_by_categories
from pero_ocr.music.music_structures import Measure
from pero_ocr.music.output_translator import OutputTranslator as Translator


class MusicPageExporter:
    """Take pageLayout XML exported from pero-ocr with transcriptions and re-construct page of musical notation.

    For CLI usage see user_scripts/music_exporter.py
    """

    def __init__(self, input_xml_path: str = '', input_transcription_files: List[str] = None,
                 translator_path: str = None, output_folder: str = 'output_page', export_midi: bool = False,
                 export_musicxml: bool = False, categories: List = None, verbose: bool = False):
        self.translator_path = translator_path
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]  \t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='[%(levelname)-s]\t- %(message)s')
        self.verbose = verbose

        if input_xml_path and not os.path.isfile(input_xml_path):
            logging.error('No input file of this path was found')
        self.input_xml_path = input_xml_path

        self.input_transcription_files = input_transcription_files if input_transcription_files else []

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.output_folder = output_folder
        self.export_midi = export_midi
        self.export_musicxml = export_musicxml

        self.translator = Translator(filename=self.translator_path) if translator_path else None
        self.categories = categories if categories else ['Notový zápis']

    def __call__(self, page_layout=None) -> None:
        if self.input_transcription_files:
            MusicLinesExporter(input_files=self.input_transcription_files, output_folder=self.output_folder,
                               translator=self.translator, verbose=self.verbose)()
        if page_layout:
            self.process_page(page_layout)

        if self.input_xml_path:
            input_page_layout = PageLayout(file=self.input_xml_path)
            self.export_page_layout(input_page_layout)

    def process_page(self, page_layout: PageLayout) -> None:
        self.export_page_layout(page_layout, page_layout.id)

    def export_page_layout(self, page_layout: PageLayout, file_id: str = None) -> None:
        if self.export_musicxml or self.export_midi:
            page_layout, _ = split_page_layout_by_categories(page_layout, self.categories)
            parts = self.regions_to_parts(
                page_layout.regions)
            if not parts:
                return

            music_parts = []
            for part in parts:
                music_parts.append(part.encode_to_music21())

            # Finalize score creation
            metadata = music.metadata.Metadata()
            metadata.title = metadata.composer = ''
            score = music.stream.Score([metadata] + music_parts)

            if self.export_musicxml:
                output_file = self.get_output_file(file_id, extension='musicxml')
                xml = music21_to_musicxml(score)
                write_to_file(output_file, xml)

            if self.export_midi:
                self.export_to_midi(score, parts, file_id)

    def get_output_file(self, file_id: str = None, extension: str = 'musicxml') -> str:
        base = self.get_output_file_base(file_id)
        return f'{base}.{extension}'

    def get_output_file_base(self, file_id: str = None) -> str:
        if not file_id:
            file_id = os.path.basename(self.input_xml_path)
            if not file_id:
                file_id = 'output'
        name, *_ = re.split(r'\.', file_id)
        return os.path.join(self.output_folder, f'{name}')

    def export_to_midi(self, score, parts, file_id: str = None):
        # Export whole score to midi
        output_file = self.get_output_file(file_id, extension='mid')
        score.write("midi", output_file)

        for part in parts:
            base = self.get_output_file_base(file_id)
            part.export_to_midi(base)

    def regions_to_parts(self, regions: List[RegionLayout]) -> List[Part]:
        """Takes a list of regions and splits them to parts."""
        max_parts = max(
            [len(region.get_lines_of_category(self.categories)) for region in regions],
            default=0
        )
        if max_parts == 0:
            print('Warning: No music lines found in page.')
            return []

        parts = [Part(self.translator) for _ in range(max_parts)]
        for region in regions:
            for part, line in zip(parts, region.get_lines_of_category(self.categories)):
                part.add_textline(line)

        return parts


class MusicLinesExporter:
    """Takes text files with transcriptions as individual lines and exports musicxml file for each one"""
    def __init__(self, input_files: List[str] = None, output_folder: str = 'output_musicxml',
                 translator: Translator = None, verbose: bool = False):
        self.translator = translator
        self.output_folder = output_folder

        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]  \t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='[%(levelname)-s]\t- %(message)s')

        logging.debug('Hello World! (from ReverseConverter)')

        self.input_files = MusicLinesExporter.get_input_files(input_files)
        MusicLinesExporter.prepare_output_folder(output_folder)

    def __call__(self):
        if not self.input_files:
            logging.error('No input files provided. Exiting...')
            sys.exit(1)

        # For every file, convert it to MusicXML
        for input_file_name in self.input_files:
            logging.info(f'Reading file {input_file_name}')
            lines = MusicLinesExporter.read_file_lines(input_file_name)

            for i, line in enumerate(lines):
                match = re.fullmatch(r'([a-zA-Z0-9_\-]+)[a-zA-Z0-9_\.]+\s+([0-9]+\s+)?\"([\S\s]+)\"', line)

                if not match:
                    logging.debug(f'NOT MATCHING PATTERN. Skipping line {i} in file {input_file_name}: '
                                  f'({line[:min(50, len(line))]}...)')
                    continue

                stave_id = match.group(1)
                labels = match.group(3)
                if self.translator is not None:
                    labels = self.translator.translate_line(labels)
                output_file_name = os.path.join(self.output_folder, f'{stave_id}.musicxml')

                parsed_labels = semantic_line_to_music21_score(labels)
                if not isinstance(parsed_labels, music.stream.Stream):
                    logging.error(f'Labels could not be parsed. Skipping line {i} in file {input_file_name}: '
                                  f'({line[:min(50, len(line))]}...)')
                    continue

                logging.info(f'Parsing successfully completed.')
                # parsed_labels.show()  # Show parsed labels in some visual program (MuseScore by default)

                xml = music21_to_musicxml(parsed_labels)
                write_to_file(output_file_name, xml)

    @staticmethod
    def prepare_output_folder(output_folder: str):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    @staticmethod
    def get_input_files(input_files: List[str] = None):
        existing_files = []

        if not input_files:
            return []

        for input_file in input_files:
            if os.path.isfile(input_file):
                existing_files.append(input_file)

        return existing_files

    @staticmethod
    def read_file_lines(input_file: str) -> List[str]:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        if not lines:
            logging.warning(f'File {input_file} is empty!')

        return [line for line in lines if line]


class Part:
    """Represent musical part (part of notation for one instrument/section)"""

    def __init__(self, translator: Translator = None):
        self.translator = translator

        self.repr_music21 = music.stream.Part([music.instrument.Piano()])  # Default instrument is piano
        self.labels: List[str] = []
        self.textlines: List[TextLineWrapper] = []
        self.measures: List[Measure] = []  # List of measures in internal representation, NOT music21

    def add_textline(self, line: TextLine) -> None:
        labels = line.transcription
        if self.translator is not None:
            labels = self.translator.translate_line(labels)
        self.labels.append(labels)

        new_measures = parse_semantic_to_measures(labels)

        # Delete first clef symbol of first measure in line if same as last clef in previous line
        if len(self.measures) and new_measures[0].get_start_clef() == self.measures[-1].last_clef:
            new_measures[0].delete_clef_symbol()

        new_measures_encoded = encode_measures(new_measures, len(self.measures) + 1)
        new_measures_encoded_without_measure_ids = encode_measures(new_measures)

        self.measures += new_measures
        self.repr_music21.append(new_measures_encoded)

        self.textlines.append(TextLineWrapper(line, new_measures_encoded_without_measure_ids))

    def encode_to_music21(self) -> music.stream.Part:
        if self.repr_music21 is None:
            logging.info('Part empty')

        return self.repr_music21

    def export_to_midi(self, file_base: str):
        for text_line in self.textlines:
            text_line.export_midi(file_base)


class TextLineWrapper:
    """Class to wrap one TextLine for easier export etc."""
    def __init__(self, text_line: TextLine, measures: List[music.stream.Measure]):
        self.text_line = text_line
        self.repr_music21 = music.stream.Part([music.instrument.Piano()] + measures)

    def export_midi(self, file_base: str = 'out'):
        # do not export line, if it has no notes
        if not any([note for note in self.repr_music21.flatten().notes]):
            return

        filename = f'{file_base}_{self.text_line.id}.mid'

        xml = music21_to_musicxml(self.repr_music21)
        parsed_xml = music.converter.parse(xml)
        parsed_xml.write('mid', filename)


def parse_semantic_to_measures(labels: str) -> List[Measure]:
    """Convert line of semantic labels to list of measures.

    Args:
        labels (str): one line of labels in semantic format without any prefixes.
    """
    labels = labels.strip('"')

    measures_labels = re.split(r'barline', labels)

    stripped_measures_labels = []
    for measure_label in measures_labels:
        stripped = measure_label.strip().strip('+').strip()
        if stripped:
            stripped_measures_labels.append(stripped)

    measures = [Measure(measure_label) for measure_label in stripped_measures_labels if measure_label]

    previous_measure_key = music.key.Key()  # C Major as a default key (without accidentals)
    for measure in measures:
        previous_measure_key = measure.get_key(previous_measure_key)

    measures[0].new_system = True

    previous_measure_last_clef = measures[0].get_last_clef()
    for measure in measures[1:]:
        previous_measure_last_clef = measure.get_last_clef(previous_measure_last_clef)

    return measures


def encode_measures(measures: List, measure_id_start_from: int = 1) -> List[Measure]:
    """Get list of measures and encode them to music21 encoded measures."""
    logging.debug('-------------------------------- -------------- --------------------------------')
    logging.debug('-------------------------------- START ENCODING --------------------------------')
    logging.debug('-------------------------------- -------------- --------------------------------')

    measures_encoded = []
    for measure_id, measure in enumerate(measures):
        measures_encoded.append(measure.encode_to_music21())
        measures_encoded[-1].number = measure_id_start_from + measure_id

    return measures_encoded


def semantic_line_to_music21_score(labels: str) -> music.stream.Score:
    """Get semantic line of labels, Return stream encoded in music21 score format."""
    measures = parse_semantic_to_measures(labels)
    measures_encoded = encode_measures(measures)

    # stream = music.stream.Score(music.stream.Part([music.instrument.Piano()] + measures_encoded))
    metadata = music.metadata.Metadata()
    metadata.title = metadata.composer = ''
    stream = music.stream.Score([metadata, music.stream.Part([music.instrument.Piano()] + measures_encoded)])

    return stream


def music21_to_musicxml(music_object):
    out_bytes = music.musicxml.m21ToXml.GeneralObjectExporter(music_object).parse()
    out_str = out_bytes.decode('utf-8')
    return out_str.strip()


def write_to_file(output_file_name, xml):
    with open(output_file_name, 'w', encoding='utf-8') as f:
        f.write(xml)

    logging.info(f'File {output_file_name} successfully written.')
