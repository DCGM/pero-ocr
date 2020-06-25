#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import Levenshtein

from pero_ocr.document_ocr.layout import PageLayout


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', help='Folder with page xmls whose CER will be computed', required=True)
    parser.add_argument('--ref', help='Folder with reference page xml', required=True)
    args = parser.parse_args()
    return args


def read_page_xml(path):
    try:
        page_layout = PageLayout(file=path)
    except:
        print(f'Warning: unable to load page xml "{path}"')
        return None
    return page_layout


def compare_page_layouts(hyp_fn, ref_fn):
    hyp_page = read_page_xml(hyp_fn)
    ref_page = read_page_xml(ref_fn)
    if hyp_page is None or ref_page is None:
        return None

    hyp_lines = dict([(line.id, line.transcription) for line in hyp_page.lines_iterator()])
    ref_lines = dict([(line.id, line.transcription) for line in ref_page.lines_iterator()])

    char_sum = 0
    char_dist = 0
    line_ids = set(hyp_lines.keys()) | set(ref_lines.keys())
    for line_id in line_ids:
        if line_id not in hyp_lines:
            print(f'Warning: Line "{line_id}" missing in "{hyp_fn}"')
        if line_id not in ref_lines:
            print(f'Warning: Line "{line_id}" missing in "{ref_fn}"')

        char_sum += len(ref_lines[line_id])
        char_dist += Levenshtein.distance(ref_lines[line_id], hyp_lines[line_id])

    return char_sum, char_dist


def main():
    args = parse_arguments()

    xml_to_process = set([f for f in os.listdir(args.ref) if os.path.splitext(f)[1] == '.xml'])
    xml_to_process |= set([f for f in os.listdir(args.hyp) if os.path.splitext(f)[1] == '.xml'])

    total_char_sum = 0
    total_char_dist = 0
    for xml_file in xml_to_process:
        result = compare_page_layouts(os.path.join(args.hyp, xml_file), os.path.join(args.ref, xml_file))
        if result is not None:
            char_sum, char_dist = result
            print('Result:', xml_file, char_sum, char_dist, char_dist / (char_sum + 1))
            total_char_sum += char_sum
            total_char_dist += char_dist

    print('Result: FINAL', total_char_sum, total_char_dist, total_char_dist / (total_char_sum))


if __name__ == "__main__":
    main()
