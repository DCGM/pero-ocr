#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import configparser
import argparse
import cv2
import logging
import logging.handlers
import re
from typing import Set, List, Optional
import traceback
import sys
import time
from multiprocessing import Pool
import random
import Levenshtein

from safe_gpu import safe_gpu

from pero_ocr import utils  # noqa: F401 -- there is code executed upon import here.
from pero_ocr.document_ocr.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser

from user_scripts.parse_folder import setup_logging
from user_scripts.parse_folder import get_value_or_none
from user_scripts.parse_folder import create_dir_if_not_exists
from user_scripts.parse_folder import Computator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to input config file.')
    parser.add_argument('-i', '--input-image-path', help='')
    parser.add_argument('-x', '--input-xml-path', help='')
    parser.add_argument('--max-lines', type=int, help='')
    parser.add_argument('--set-gpu', action='store_true', help='Sets visible CUDA device to first unused GPU.')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    config = configparser.ConfigParser()
    config.read(args.config)
    config['PARSE_FOLDER']['INPUT_IMAGE_PATH'] = args.input_image_path
    config['PARSE_FOLDER']['INPUT_XML_PATH'] = args.input_xml_path

    setup_logging(config['PARSE_FOLDER'])
    logger = logging.getLogger()

    if args.set_gpu:
        gpu_owner = safe_gpu.GPUOwner(logger=logger)  # noqa: F841

    page_parser = PageParser(config, config_path=os.path.dirname(args.config))

    logger.info(f'Reading images from {args.input_image_path}.')

    images, page_layouts = load_images_and_page_layouts(args.input_image_path, args.input_xml_path)
    line_crops, gts = get_line_crops_and_transcriptions(page_parser, images, page_layouts, args.max_lines)
    print(len(line_crops), len(gts))
    #for line, trans in zip(line_crops_to_process, transcriptions):
    #    print(line, trans)

    t_start = time.time()

    for embed_id in range(page_parser.ocr.ocr_engine.embed_num):
        page_parser.ocr.ocr_engine.embed_id = embed_id

        t1 = time.time()

        transcriptions, _, _ = page_parser.ocr.ocr_engine.process_lines(line_crops)
        ref_char_sum = 0
        ref_gt_char_dist = 0
        for gt, trans in zip(gts, transcriptions):
            ref_char_sum += len(gt)
            ref_gt_char_dist += Levenshtein.distance(gt, trans)
        if ref_char_sum > 0:
            print(f'{embed_id + 1}/{page_parser.ocr.ocr_engine.embed_num} {100.0 * ref_gt_char_dist / ref_char_sum:.2f} % CER [ {ref_gt_char_dist} / {ref_char_sum} ] Time: {time.time() - t1:.2f}')
        else:
            print(f'{embed_id + 1}/{page_parser.ocr.ocr_engine.embed_num} N/A % CER [ {ref_gt_char_dist} / {ref_char_sum} ] Time: {time.time() - t1:.2f}')

    print(f'PROCESSING TIME {(time.time() - t_start)}')


def get_line_crops_and_transcriptions(page_parser, images, page_layouts, max_lines):
    num_lines = 0
    for page_layout in page_layouts:
        for region in page_layout.regions:
            num_lines += len(region.lines)

    lines_to_keep = list(range(num_lines))
    random.shuffle(lines_to_keep)
    lines_to_keep = lines_to_keep[:max_lines]

    line_index = 0
    for page_layout in page_layouts:
        for region in page_layout.regions:
            new_lines = []
            for line in region.lines:
                if line_index in lines_to_keep:
                    new_lines.append(line)
                line_index += 1
            region.lines = new_lines

    for i in range(len(page_layouts)):
        page_layouts[i] = page_parser.line_cropper.process_page(images[i], page_layouts[i])

    line_crops = []
    transcriptions = []
    for page_layout in page_layouts:
        for region in page_layout.regions:
            for line in region.lines:
                line_crops.append(line.crop)
                transcriptions.append(line.transcription)

    return line_crops, transcriptions


def load_images_and_page_layouts(input_image_path, input_xml_path):
    ignored_extensions = ['', '.xml', '.logits']
    images_to_process = [f for f in os.listdir(input_image_path) if
                         os.path.splitext(f)[1].lower() not in ignored_extensions]
    images_to_process = sorted(images_to_process)
    ids_to_process = [os.path.splitext(os.path.basename(file))[0] for file in images_to_process]
    images = []
    page_layouts = []
    for index, (file_id, image_file_name) in enumerate(zip(ids_to_process, images_to_process)):
        image = cv2.imread(os.path.join(input_image_path, image_file_name), 1)
        if image is None:
            raise Exception(f'Unable to read image "{os.path.join(input_image_path, image_file_name)}"')
        images.append(image)
        page_layout = PageLayout(file=os.path.join(input_xml_path, file_id + '.xml'))
        page_layouts.append(page_layout)

    return images, page_layouts


if __name__ == "__main__":
    main()
