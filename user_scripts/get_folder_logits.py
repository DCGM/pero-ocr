#!/usr/bin/env python3

import argparse
import os
import pickle

from pero_ocr.ocr_engine import line_ocr_engine as ocr
from pero_ocr.line_images_io import read_images


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--ocr-json', help='Path to OCR config', required=True)
    parser.add_argument('-i', '--input', help='Folder with lines as images', required=True)
    parser.add_argument('-o', '--output', help='Where to put the logits', required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    ocr_json = args.ocr_json

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings on loading models
    ocr_engine = ocr.EngineLineOCR(ocr_json, gpu_id=0)

    lines, names = read_images(args.input)
    _, logits = ocr_engine.process_lines(lines)

    complete_data = {'names': names, 'logits': logits}

    with open(args.output, 'wb') as f:
        pickle.dump(complete_data, f)


if __name__ == "__main__":
    main()
