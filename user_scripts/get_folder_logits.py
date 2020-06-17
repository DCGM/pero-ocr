#!/usr/bin/env python3

import argparse
import os
import pickle

from pero_ocr.line_images_io import read_images


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--ocr-json', help='Path to OCR config', required=True)
    parser.add_argument('-i', '--input', help='Folder with lines as images', required=True)
    parser.add_argument('-o', '--output', help='Where to put the logits', required=True)
    parser.add_argument('-m', '--model-type', default='Pytorch', choices=['TF', 'Pytorch'])

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    ocr_json = args.ocr_json

    if args.model_type == 'TF':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings on loading models
        from pero_ocr.ocr_engine.line_ocr_engine import EngineLineOCR as OCREngine
    elif args.model_type == 'Pytorch':
        from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR as OCREngine

    ocr_engine = OCREngine(ocr_json, gpu_id=0)

    lines, names = read_images(args.input)
    _, logits = ocr_engine.process_lines(lines)

    complete_data = {'names': names, 'logits': logits}

    with open(args.output, 'wb') as f:
        pickle.dump(complete_data, f)


if __name__ == "__main__":
    main()
