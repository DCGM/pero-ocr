#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import configparser
import argparse
import cv2
import re
from typing import Set, List, Optional
import traceback
import sys
import time
import subprocess

from pero_ocr import utils
from pero_ocr.document_ocr.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to input config file.')
    parser.add_argument('-s', '--skip-processed', action='store_true', required=False,
                        help='If set, already processed files are skipped.')
    parser.add_argument('-i', '--input-image-path', help='')
    parser.add_argument('-x', '--input-xml-path', help='')
    parser.add_argument('--output-xml-path', help='')
    parser.add_argument('--output-render-path', help='')
    parser.add_argument('--output-line-path', help='')
    parser.add_argument('--output-logit-path', help='')
    parser.add_argument('--output-alto-path', help='')
    parser.add_argument('--set-gpu', action='store_true', help='Sets visible CUDA device to first unused GPU.')
    args = parser.parse_args()
    return args


def get_value_or_none(config, section, key):
    if config.has_option(section, key):
        value = config[section][key]
    else:
        value = None
    return value


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_already_processed_files_in_directory(directory: Optional[str]) -> Set[str]:
    already_processed = set()

    if directory is not None:
        file_pattern = r"(.+?)(\.logits|\.xml|\.jpg)"
        regex = re.compile(file_pattern)

        for file in os.listdir(directory):
            matched = regex.match(file)
            if matched:
                already_processed.add(matched.groups()[0])

    return already_processed


def load_already_processed_files(directories: List[Optional[str]]) -> Set[str]:
    already_processed = set()
    first = True

    for directory in directories:
        if directory is not None:
            files = load_already_processed_files_in_directory(directory)

            if first:
                already_processed = files
                first = False
            else:
                already_processed = already_processed.intersection(files)

    return already_processed


class LMDB_writer(object):
    def __init__(self, path):
        import lmdb
        gb100 = 100000000000
        self.env_out = lmdb.open(path, map_size=gb100)
        self.data_size = 0
        self.file = open('dataset.ann', 'w')

    def __call__(self, page_layout: PageLayout, file_id):
        with self.env_out.begin(write=True) as txn_out:
            c_out = txn_out.cursor()
            all_lines = list(page_layout.lines_iterator())
            all_lines = sorted(all_lines, key=lambda x: x.id)
            for line in all_lines:
                if line.transcription:
                    key = f'{file_id}-{line.id}.jpg'
                    img = cv2.imencode('.jpg', line.crop.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
                    print(key, line.transcription, file=self.file)
                    self.data_size += len(img)
                    c_out.put(key.encode(), img)


def main():
    # initialize some parameters
    args = parse_arguments()
    config_path = args.config
    skip_already_processed_files = args.skip_processed

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings on loading models

    if args.set_gpu:
        utils.setGPU()

    if not os.path.isfile(config_path):
        print(f'ERROR: Config file does not exist: "{config_path}".')
        exit(-1)

    config = configparser.ConfigParser()
    config.read(config_path)

    if args.input_image_path is not None:
        config['PARSE_FOLDER']['INPUT_IMAGE_PATH'] = args.input_image_path
    if args.input_xml_path is not None:
        config['PARSE_FOLDER']['INPUT_XML_PATH'] = args.input_xml_path
    if args.output_xml_path is not None:
        config['PARSE_FOLDER']['OUTPUT_XML_PATH'] = args.output_xml_path
    if args.output_render_path is not None:
        config['PARSE_FOLDER']['OUTPUT_RENDER_PATH'] = args.output_render_path
    if args.output_line_path is not None:
        config['PARSE_FOLDER']['OUTPUT_LINE_PATH'] = args.output_line_path
    if args.output_logit_path is not None:
        config['PARSE_FOLDER']['OUTPUT_LOGIT_PATH'] = args.output_logit_path
    if args.output_alto_path is not None:
        config['PARSE_FOLDER']['OUTPUT_ALTO_PATH'] = args.output_alto_path

    page_parser = PageParser(config, config_path=os.path.dirname(config_path))

    input_image_path = get_value_or_none(config, 'PARSE_FOLDER', 'INPUT_IMAGE_PATH')
    input_xml_path = get_value_or_none(config, 'PARSE_FOLDER', 'INPUT_XML_PATH')
    input_logit_path = get_value_or_none(config, 'PARSE_FOLDER', 'INPUT_LOGIT_PATH')

    output_render_path = get_value_or_none(config, 'PARSE_FOLDER', 'OUTPUT_RENDER_PATH')
    output_line_path = get_value_or_none(config, 'PARSE_FOLDER', 'OUTPUT_LINE_PATH')
    output_xml_path = get_value_or_none(config, 'PARSE_FOLDER', 'OUTPUT_XML_PATH')
    output_logit_path = get_value_or_none(config, 'PARSE_FOLDER', 'OUTPUT_LOGIT_PATH')
    output_alto_path = get_value_or_none(config, 'PARSE_FOLDER', 'OUTPUT_ALTO_PATH')

    if output_line_path is not None and 'lmdb' in output_line_path:
        lmdb_writer = LMDB_writer(output_line_path)
    else:
        lmdb_writer = None

    if output_render_path is not None:
        create_dir_if_not_exists(output_render_path)
    if output_line_path is not None:
        create_dir_if_not_exists(output_line_path)
    if output_xml_path is not None:
        create_dir_if_not_exists(output_xml_path)
    if output_logit_path is not None:
        create_dir_if_not_exists(output_logit_path)
    if output_alto_path is not None:
        create_dir_if_not_exists(output_alto_path)

    if input_logit_path is not None and input_xml_path is None:
        input_logit_path = None
        print('Warning: Logit path specified and Page XML path not specified. Logits will be ignored.')

    if input_image_path is not None:
        print(f'Reading images from {input_image_path}.')
        images_to_process = [f for f in os.listdir(input_image_path) if
                             os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.tif']]
        ids_to_process = [os.path.splitext(os.path.basename(file))[0] for file in images_to_process]
    elif input_xml_path is not None:
        print(f'Reading page xml from {input_xml_path}')
        xml_to_process = [f for f in os.listdir(input_xml_path) if
                          os.path.splitext(f)[1] == '.xml']
        images_to_process = [None] * len(xml_to_process)
        ids_to_process = [os.path.splitext(os.path.basename(file))[0] for file in xml_to_process]
    else:
        raise Exception(
            f'Either INPUT_IMAGE_PATH or INPUT_XML_PATH has to be specified. Both are missing in {config_path}.')

    if skip_already_processed_files:
        # Files already processed are skipped. File is considered as already processed when file with appropriate
        # extension is found in all required output directories. If any of the output paths is set to 'None'
        # (i.e. the output is not required) than this directory is omitted.
        already_processed_files = load_already_processed_files([output_xml_path, output_logit_path, output_render_path])
        if len(already_processed_files) > 0:
            print(f"Already processed {len(already_processed_files)} file(s).")

            images_to_process = [image for id, image in zip(ids_to_process, images_to_process) if id not in already_processed_files]
            ids_to_process = [id for id in ids_to_process if id not in already_processed_files]

    for index, (file_id, image_file_name) in enumerate(zip(ids_to_process, images_to_process)):
        print("Processing {file_id}".format(file_id=file_id))
        t1 = time.time()
        try:
            if input_image_path is not None:
                image = cv2.imread(os.path.join(input_image_path, image_file_name), 1)
                if image is None:
                    raise Exception(f'Unable to read image "{os.path.join(input_image_path, image_file_name)}"')
            else:
                image = None

            if input_xml_path:
                page_layout = PageLayout(file=os.path.join(input_xml_path, file_id + '.xml'))
            else:
                page_layout = PageLayout(id=file_id, page_size=(image.shape[0], image.shape[1]))

            if input_logit_path is not None:
                page_layout.load_logits(os.path.join(input_logit_path, file_id + '.logits'))

            page_layout = page_parser.process_page(image, page_layout)

            if output_xml_path is not None:
                page_layout.to_pagexml(os.path.join(output_xml_path, file_id + '.xml'))

            if output_render_path is not None:
                page_layout.render_to_image(image)
                cv2.imwrite(os.path.join(output_render_path, file_id + '.jpg'), image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            if output_logit_path is not None:
                page_layout.save_logits(os.path.join(output_logit_path, file_id + '.logits'))

            if output_alto_path is not None:
                page_layout.to_altoxml(os.path.join(output_alto_path, file_id + '.xml'))

            if output_line_path is not None:
                if lmdb_writer:
                    lmdb_writer(page_layout, file_id)
                else:
                    for region in page_layout.regions:
                        for line in region.lines:
                            cv2.imwrite(
                                os.path.join(output_line_path, f'{file_id}-{line.id}.jpg'),
                                line.crop.astype(np.uint8),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 98])

        except KeyboardInterrupt:
            traceback.print_exc()
            print('Terminated by user.')
            sys.exit()
        except Exception as e:
            print(f'ERROR: Failed to process file {file_id}.')
            print(e)
            traceback.print_exc()
        print("DONE {current}/{total} ({percentage:.2f} %) [id: {file_id}] Time:{time:.2f}".format(
            current=index+1, total=len(ids_to_process), percentage=(index+1)/len(ids_to_process) * 100,
            file_id=file_id, time=time.time() - t1))


if __name__ == "__main__":
    main()
