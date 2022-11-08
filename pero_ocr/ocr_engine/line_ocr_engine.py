# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import cv2
import numpy as np
from os.path import isabs, realpath, join, dirname
from scipy import sparse

from .softmax import softmax


class BaseEngineLineOCR(object):
    def __init__(self, json_def, device, batch_size=8):
        with open(json_def, 'r', encoding='utf8') as f:
            self.config = json.load(f)

        self.line_px_height = self.config['line_px_height']
        self.line_vertical_scale = self.config['line_vertical_scale']

        if isabs(self.config['checkpoint']):
            self.checkpoint = self.config['checkpoint']
        else:
            self.checkpoint = realpath(join(dirname(json_def), self.config['checkpoint']))

        self.characters = tuple(self.config['characters'])
        self.net_name = self.config['net_name']
        if "embed_num" in self.config:
            self.embed_num = int(self.config["embed_num"])
        else:
            self.embed_num = None
        if "embed_id" in self.config:
            if self.config["embed_id"] != "mean":
                self.embed_id = int(self.config["embed_id"])
            else:
                self.embed_id = "mean"
        else:
            self.embed_id = None

        self.device = device

        self.batch_size = batch_size

        self.line_padding_px = 32
        self.max_input_horizontal_pixels = 480 * batch_size

    def process_lines(self, lines, sparse_logits=True, tight_crop_logits=False, no_logits=False):
        """Runs ocr network on multiple lines.
        Args:
            lines (iterable): contains cropped lines as numpy arrays.

        Returns:
            transcripts (list of strings): contains UTF-8 line transcripts
            logits (list of sparse matrices): character logits for lines
        """

        # check line crops for correct shape
        for line in lines:
            if line.shape[0] == self.line_px_height:
                ValueError("Line height needs to be {} for this ocr network and is {} instead.".format(self.line_px_height, line.shape[0]))
            if line.shape[2] == 3:
                ValueError("Line crops need three color channes, but this one has {}.".format(line.shape[2]))

        all_transcriptions = [None]*len(lines)
        all_logits = [None]*len(lines)
        all_logit_coords = [None]*len(lines)

        #  process all lines ordered by their length
        line_ids = [x for x, y in sorted(enumerate(lines), key=lambda x: -x[1].shape[1])]
        while line_ids:
            max_width = lines[line_ids[0]].shape[1]
            max_width = int(np.ceil(max_width / 32.0) * 32)
            batch_size = max(1, self.max_input_horizontal_pixels // max_width)

            batch_line_ids = line_ids[:batch_size]
            line_ids = line_ids[batch_size:]

            batch_data = np.zeros(
                [len(batch_line_ids), self.line_px_height, max_width + 2*self.line_padding_px, 3], dtype=np.uint8)
            for data, ids in zip(batch_data, batch_line_ids):
                data[:, self.line_padding_px:self.line_padding_px+lines[ids].shape[1], :] = lines[ids]

            if batch_data.shape[2] > self.max_input_horizontal_pixels:
                print(f'WARNING: Line too long for OCR engine. Cropping from {batch_data.shape[2]} px down to {self.max_input_horizontal_pixels}.')
                batch_data = batch_data[:, :, :self.max_input_horizontal_pixels]

            out_transcriptions, out_logits = self.run_ocr(batch_data)

            if no_logits:
                for ids, transcription in zip(batch_line_ids, out_transcriptions):
                    all_transcriptions[ids] = transcription
            else:
                for ids, transcription, line_logits in zip(batch_line_ids, out_transcriptions, out_logits):
                    all_transcriptions[ids] = transcription

                    if tight_crop_logits:
                        line_logits = line_logits[
                                      int(self.line_padding_px // self.net_subsampling):int(
                                         (self.line_padding_px + lines[ids].shape[1]) // self.net_subsampling)]
                        all_logit_coords[ids] = [None, None]
                        #else:
                    #    line_logits = line_logits[
                    #              int(self.line_padding_px // self.net_subsampling - 2):int(
                    #                  lines[ids].shape[1] // self.net_subsampling + 8)]
                    else:
                        all_logit_coords[ids] = [
                            int(self.line_padding_px // self.net_subsampling),
                            int((self.line_padding_px + lines[ids].shape[1]) // self.net_subsampling)]
                    if sparse_logits:
                        line_probs = softmax(line_logits, axis=1)
                        line_logits[line_probs < 0.0001] = 0
                        line_logits = sparse.csc_matrix(line_logits)
                    all_logits[ids] = line_logits

        return all_transcriptions, all_logits, all_logit_coords
