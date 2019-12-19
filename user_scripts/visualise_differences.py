#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import subprocess

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from pero_ocr.line_images_io import read_images
from pero_ocr.transcription_io import load_transcriptions


def get_font_fn(font):
    args = ['fc-match', '--format=%{file}', font]
    font_fn = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    return font_fn


class TextDrawer:
    def __init__(self, font):
        self.font = font

    def draw_line(self, text, width):
        height = 32

        img = PIL.Image.new('RGB', (width, height), (255, 255, 255))
        drawer = PIL.ImageDraw.Draw(img)
        drawer.text((10, 10), text, font=self.font, fill=(0, 0, 0))

        return np.asarray(img)


def process_line(scan, drawer, hyp_1, hyp_2):
    hyp_1_img = drawer.draw_line(hyp_1, np.asarray(scan).shape[1])
    hyp_2_img = drawer.draw_line(hyp_2, np.asarray(scan).shape[1])
    return np.concatenate([scan, hyp_1_img, hyp_2_img])


def images_dict(folder):
    lines, names = read_images(folder)

    return {name: line for name, line in zip(names, lines)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lines_folder')
    parser.add_argument('hyps1')
    parser.add_argument('hyps2')
    parser.add_argument('out_folder')
    parser.add_argument('font', default='DejaVu', nargs='?')
    args = parser.parse_args()

    font_fn = get_font_fn(args.font)
    font = PIL.ImageFont.truetype(font_fn, 16)
    drawer = TextDrawer(font)

    images = images_dict(args.lines_folder)
    hyps_1 = load_transcriptions(args.hyps1)
    hyps_2 = load_transcriptions(args.hyps2)

    for key in images:
        if hyps_1[key] == hyps_2[key]:
            continue

        full_img = process_line(images[key], drawer, hyps_1[key], hyps_2[key])
        matplotlib.image.imsave('{}/{}.png'.format(args.out_folder, key), full_img)


if __name__ == '__main__':
    main()
