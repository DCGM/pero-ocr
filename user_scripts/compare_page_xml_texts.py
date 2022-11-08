#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import Levenshtein
import sys

from pero_ocr.document_ocr.layout import PageLayout


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--print-all', action='store_true', help='Report CER per page xml')
    parser.add_argument('--hyp', help='Folder with page xmls whose CER will be computed', required=True)
    parser.add_argument('--ref', help='Folder with reference page xml', required=True)
    args = parser.parse_args()
    return args


def read_page_xml(path):
    try:
        page_layout = PageLayout(file=path)
    except Exception:
        print(f'Warning: unable to load page xml "{path}"')
        return None
    return page_layout


def compare_page_layouts(hyp_fn, ref_fn):
    hyp_page = read_page_xml(hyp_fn)
    ref_page = read_page_xml(ref_fn)
    if hyp_page is None or ref_page is None:
        return None

    hyp_lines = {line.id: line.transcription for line in hyp_page.lines_iterator()}
    ref_lines = {line.id: line.transcription for line in ref_page.lines_iterator()}

    char_sum = 0
    char_dist = 0
    line_ids = set(hyp_lines.keys()) | set(ref_lines.keys())
    for line_id in line_ids:
        if line_id not in hyp_lines:
            sys.stderr.write(f'Warning: Line "{line_id}" missing in "{hyp_fn}"\n')
            continue
        if line_id not in ref_lines:
            # sys.stderr.write(f'Warning: Line "{line_id}" missing in "{ref_fn}"\n')
            continue

        char_sum += len(ref_lines[line_id])
        char_dist += Levenshtein.distance(ref_lines[line_id].strip(), hyp_lines[line_id].strip())

    return char_sum, char_dist


def print_result(name, nb_errors, ref_len):
    if ref_len > 0:
        print(f'{name} {100.0*nb_errors/ref_len:.2f} % CER [ {nb_errors} / {ref_len} ]')
    else:
        print(f'{name} N/A % CER [ {nb_errors} / {ref_len} ]')


def main():
    args = parse_arguments()

    xml_to_process = set(f for f in os.listdir(args.ref) if os.path.splitext(f)[1] == '.xml')
    xml_to_process |= set(f for f in os.listdir(args.hyp) if os.path.splitext(f)[1] == '.xml')

    total_char_sum = 0
    total_char_dist = 0
    for xml_file in xml_to_process:
        result = compare_page_layouts(os.path.join(args.hyp, xml_file), os.path.join(args.ref, xml_file))
        if result is not None:
            char_sum, char_dist = result
            if args.print_all:
                print_result(xml_file, char_dist, char_sum)
            total_char_sum += char_sum
            total_char_dist += char_dist

    print_result('summary', total_char_dist, total_char_sum)


if __name__ == "__main__":
    main()
