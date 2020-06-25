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


def compare_page_layouts(page_file_1, page_file_2):
    page1 = read_page_xml(page_file_1)
    page2 = read_page_xml(page_file_2)
    if page1 is None or page2 is None:
        return None

    lines1 = dict([(line.id, line.transcription) for line in page1.lines_iterator()])
    lines2 = dict([(line.id, line.transcription) for line in page2.lines_iterator()])

    char_sum = 0
    char_dist = 0
    line_ids = set(lines1.keys()) | set(lines2.keys())
    for line_id in line_ids:
        if line_id not in lines1:
            print(f'Warning: Line "{line_id}" missing in "{page_file_1}"')
        if line_id not in lines2:
            print(f'Warning: Line "{line_id}" missing in "{page_file_2}"')

        char_sum += len(lines2[line_id])
        char_dist += Levenshtein.distance(lines1[line_id], lines2[line_id])

    return char_sum, char_dist


def main():
    # initialize some parameters
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
