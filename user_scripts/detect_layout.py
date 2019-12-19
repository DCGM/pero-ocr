#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import configparser
import argparse
import shutil
import traceback

import pero_ocr.document_ocr.IO_utils as io
import pero_ocr.document_ocr.parser_utils as parser
import pero_ocr.document_ocr.layout as layout
import pero_ocr.document_ocr.paragraph_engine as paragraphs


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to input config file', required=True)

    args = parser.parse_args()

    return(args)


def main():

    # initialize some parameters
    args = parse_arguments()
    config_path = args.config

    config = configparser.ConfigParser()
    config.read(config_path)
    input_path = config['PATHS']['INPUT']
    output_path = config['PATHS']['OUTPUT']
    layout_model = config['SETTINGS']['LAYOUT_MODEL']
    use_cpu = config['SETTINGS'].getboolean('USE_CPU')
    save_renders = config['SETTINGS'].getboolean('SAVE_RENDERS')
    downsample = config['SETTINGS'].getint('DOWNSAMPLE')

    if not os.path.exists(os.path.join(output_path, 'page')):
        os.makedirs(os.path.join(output_path, 'page'))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings on loading models
    paragraph_engine = paragraphs.EngineParagraphDetector(layout_model, use_cpu=use_cpu)

    files_to_process = [f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    for file in files_to_process:
        image_fullsize, image = io.load_and_preprocess(input_path, file, 0, downsample)
        file = os.path.splitext(file)[0]
        paragraph_coords = paragraph_engine.detect(image)
        page_layout = []
        for paragraph_id, paragraph_coord in enumerate(paragraph_coords, 1):
            page_layout.append(layout.RegionLayout('r_{}'.format(paragraph_id), downsample * np.asarray(paragraph_coord)))

        xml_string = io.layout_to_xml(file, page_layout, image_fullsize.shape[:2])
        with open(os.path.join(output_path, 'page', '{}.xml'.format(file)), 'w') as xml_file:
            xml_file.write(xml_string.decode())
        if save_renders:
            io.save_render(page_layout, output_path, file, image_fullsize)

if __name__ == "__main__":
    main()
