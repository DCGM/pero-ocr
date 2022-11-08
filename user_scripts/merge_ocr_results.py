#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import argparse
import traceback
import sys

from pero_ocr.document_ocr.layout import PageLayout
from pero_ocr.confidence_estimation import get_line_confidence
from pero_ocr.document_ocr.arabic_helper import ArabicHelper


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Merge results of multiple OCR engines together by picking the most cinfident transcription '
                    'for each text line. The tool takes multiple directories, where each should contain Page XML '
                    'files and corresponding logit files. The file names in each directory must be the same.'
                    'Text lines and their IDs must be the same in each directory.')
    parser.add_argument('--output-path', required=True, help='Store here the merged Page XML and logit files.')
    parser.add_argument('--filter-list', help='Only process ID in this file')

    parser.add_argument('input_paths', metavar='input_paths', type=str, nargs='+',
                        help='List of directories with OCR outputs to merge.')
    parser.add_argument('--min-confidence', type=float, default=0,
                        help='Remove text lines with confidence lower than this value.')
    parser.add_argument('--fix-arabic-order', action='store_true', help='Export correct sequential order of arabic text in Page XML format.')
    args = parser.parse_args()
    return args


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_confidences(line):
    if line.transcription is not None and line.transcription != "":
        char_map = dict([(c, i) for i, c in enumerate(line.characters)])
        c_idx = np.asarray([char_map[c] for c in line.transcription])
        try:
            confidences = get_line_confidence(line, c_idx)
        except ValueError:
            print('ERROR: Known error in get_line_confidence() - Please, fix it. Logit slice has zero length.')
            confidences = np.ones(len(line.transcription)) * 0.5
        return confidences
    return np.asarray([])


def merge_layouts(page_layouts):
    merged_layout = page_layouts[0]
    all_lines = [layout.lines_iterator() for layout in page_layouts]

    for lines in zip(*all_lines):
        merged_line = lines[0]

        for line in lines:
            if line.id != merged_line.id:
                print(f'ERROR: Line ID is not matching for layout id {merged_layout.id}.')
                exit(-1)

        best_confidence = 0
        for line in lines:
            line_confidences = get_confidences(line)
            if line_confidences.size > 0:
                line_confidence = line_confidences.mean()
            else:
                line_confidence = -10

            if line_confidence > best_confidence:
                best_confidence = line_confidence
                merged_line.transcription = line.transcription
                merged_line.logits = line.logits
                merged_line.characters = line.characters
                merged_line.transcription_confidence = line_confidence


def main():
    args = parse_arguments()

    create_dir_if_not_exists(args.output_path)

    input_paths = args.input_paths

    files_to_process = [f for f in os.listdir(input_paths[0]) if os.path.splitext(f)[1].lower() == '.xml']

    print('input_paths', input_paths)

    if args.filter_list:
        with open(args.filter_list) as f:
            ids_to_process = f.read().split()

        files_to_process = [f for f in files_to_process if os.path.splitext(f)[0] in ids_to_process]

    # arabic_helper = ArabicHelper()

    for xml_file_name in files_to_process:
        print(xml_file_name)
        input_layouts = []
        for input_path in input_paths:
            try:
                page_layout = PageLayout(file=os.path.join(input_path, xml_file_name))
                page_layout.load_logits(os.path.join(input_path, os.path.splitext(xml_file_name)[0] + '.logits'))
                input_layouts.append(page_layout)
            except KeyboardInterrupt:
                traceback.print_exc()
                print('Terminated by user.')
                sys.exit()
            except Exception as e:
                print(f'ERROR: Failed to load Page XML or .logit file "{xml_file_name}" from "{input_path}".')
                print(e)
                traceback.print_exc()

        merge_layouts(input_layouts)
        merged_layout = input_layouts[0]

        if args.min_confidence > 0:
            for region in merged_layout.regions:
                region.lines = \
                    [l for l in region.lines if l.transcription_confidence and l.transcription_confidence > args.min_confidence]

        if args.fix_arabic_order:
            for line in merged_layout.lines_iterator():
                if arabic_helper.is_arabic_line(line.transcription):
                    line.transcription = arabic_helper.label_form_to_string(line.transcription)

        merged_layout.to_pagexml(os.path.join(args.output_path, xml_file_name))
        merged_layout.save_logits(os.path.join(args.output_path, os.path.splitext(xml_file_name)[0] + '.logits'))


if __name__ == "__main__":
    main()

