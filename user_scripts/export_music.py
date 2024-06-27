#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""Script to take output of pero-ocr with musical transcriptions and export it to musicxml and MIDI formats.

INPUTS:
- PageLayout
    - INPUT options:
        - PageLayout object using `ExportMusicPage.__call__()` method
        - XML PageLayout (exported directly from pero-ocr engine) using `--input-xml-path` argument
    - Represents one whole page of musical notation transcribed by pero-ocr engine
    - OUTPUT options:
        - One musicxml file for the page
        - MIDI file for page and for individual lines (named according to IDs in PageLayout)
- Text files with individual transcriptions and their IDs on each line using `--input-transcription-files` argument.
    - e.g.: 2370961.png ">2 + kGM + E2W E3q. + |"
            1300435.png "=4 + kDM + G3z + F3z + |"
            ...
    - OUTPUTS one musicxml file for each line with names corresponding to IDs in each line

Author: Vojtěch Vlach
Contact: xvlach22@vutbr.cz
"""

import sys
import argparse
import time

from pero_ocr.music.music_exporter import MusicPageExporter


def parseargs():
    print(' '.join(sys.argv))
    print('----------------------------------------------------------------------')
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input-xml-path", type=str, default='',
        help="Path to input XML file with exported PageLayout.")
    parser.add_argument(
        '-f', '--input-transcription-files', nargs='*', default=None,
        help='Input files with sequences as lines with IDs at the beginning.')
    parser.add_argument(
        "-t", "--translator-path", type=str, default=None,
        help="JSON File containing translation dictionary from shorter encoding (exported by model) to longest "
             "Check if needed by seeing start of any line in the transcription."
             "(e.g. SSemantic (model output): >2 + kGM + B3z + C4z + |..."
             "      Semantic (stored in XML): clef-G2 + keySignature-GM + note-B3_eighth + note-C4_eighth + barline...")
    parser.add_argument(
        "-o", "--output-folder", default='output_page',
        help="Set output file with extension. Output format is JSON")
    parser.add_argument(
        "-m", "--export-midi", action='store_true',
        help=("Enable exporting midi file to output_folder."
              "Exports whole file and individual lines with names corresponding to them TextLine IDs."))
    parser.add_argument(
        "-M", "--export-musicxml", action='store_true',
        help=("Enable exporting musicxml file to output_folder."
              "Exports whole file as one MusicXML."))
    parser.add_argument(
        '-v', "--verbose", action='store_true', default=False,
        help="Enable verbose logging.")

    return parser.parse_args()


def main():
    """Main function for simple testing"""
    args = parseargs()

    start = time.time()
    MusicPageExporter(
        input_xml_path=args.input_xml_path,
        input_transcription_files=args.input_transcription_files,
        translator_path=args.translator_path,
        output_folder=args.output_folder,
        export_midi=args.export_midi,
        export_musicxml=args.export_musicxml,
        verbose=args.verbose)()

    end = time.time()
    print(f'Total time: {end - start:.2f} s')


if __name__ == "__main__":
    main()
