# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import numpy as np
from os.path import isabs, realpath, join, dirname
from scipy import sparse
import torch

from .softmax import softmax

from pero_ocr.sequence_alignment import levenshtein_distance
from pero_ocr.music.output_translator import OutputTranslator


class BaseEngineLineOCR(object):
    def __init__(self, json_def, device, batch_size=8, model_type="ctc", substitute_output_atomic: bool = True):
        with open(json_def, 'r', encoding='utf8') as f:
            self.config = json.load(f)

        self.line_px_height = self.config['line_px_height']
        self.line_vertical_scale = self.config['line_vertical_scale']

        if isabs(self.config['checkpoint']):
            self.checkpoint = self.config['checkpoint']
        else:
            self.checkpoint = realpath(join(dirname(json_def), self.config['checkpoint']))

        self.characters = tuple(self.config['characters'])

        self.output_substitution = None
        if 'output_substitution_table' in self.config:
            self.output_substitution = OutputTranslator(dictionary=self.config['output_substitution_table'],
                                                        atomic=substitute_output_atomic)

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

        self.max_line_width = 1e10  # if the max_line_width is large enough, lines are not split into multiple parts when processed by transformers
        if "max_line_width" in self.config:
            self.max_line_width = int(self.config["max_line_width"])

        self.model_type = model_type

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

            if self.model_type == "transformer":
                max_width = min(max_width, self.max_line_width + 2 * self.line_padding_px)

            batch_size = max(1, self.max_input_horizontal_pixels // max_width)

            batch_line_ids = line_ids[:batch_size]
            line_ids = line_ids[batch_size:]

            batch_images = [lines[line_id] for line_id in batch_line_ids]
            batch_image_spans = []

            if self.model_type == "transformer":
                overlap = self.max_line_width // 4

                new_batch_images = []
                for i, image in enumerate(batch_images):
                    if image.shape[1] > self.max_line_width:
                        image_parts = []
                        
                        start = 0
                        end = self.max_line_width

                        while end < image.shape[1]:
                            image_parts.append(image[:, start:end, :])
                            start += self.max_line_width - overlap
                            end += self.max_line_width - overlap

                        image_parts.append(image[:, start:end, :])
                        new_batch_images += image_parts
                        batch_image_spans.append(len(image_parts))

                    else:
                        new_batch_images.append(image)
                        batch_image_spans.append(1)

                batch_images = new_batch_images

            batch_data = np.zeros([len(batch_images), self.line_px_height, max_width + 2*self.line_padding_px, 3], dtype=np.uint8)            
            for data, image in zip(batch_data, batch_images):
                data[:, self.line_padding_px:self.line_padding_px+image.shape[1], :] = image

            if batch_data.shape[2] > self.max_input_horizontal_pixels:
                print(f'WARNING: Line too long for OCR engine. Cropping from {batch_data.shape[2]} px down to {self.max_input_horizontal_pixels}.')
                batch_data = batch_data[:, :, :self.max_input_horizontal_pixels]

            out_transcriptions, out_logits = self.run_ocr(batch_data)

            if self.model_type == "transformer":
                merged_transcriptions = []
                merged_logits = []
                start = 0
                for span in batch_image_spans:
                    merged_line_transcription, merged_line_logits = merge_transcriptions_and_logits(out_transcriptions[start:start+span], out_logits[start:start+span])
                    merged_transcriptions.append(merged_line_transcription)
                    merged_logits.append(merged_line_logits)
                    start += span

                out_transcriptions = merged_transcriptions
                out_logits = merged_logits

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
                    elif self.model_type == "ctc":
                        all_logit_coords[ids] = [
                            int(self.line_padding_px // self.net_subsampling),
                            int((self.line_padding_px + lines[ids].shape[1]) // self.net_subsampling)]

                    elif self.model_type == "transformer":
                        all_logit_coords[ids] = [0, len(transcription)]

                    if sparse_logits:
                        line_probs = softmax(line_logits, axis=1)
                        line_logits[line_probs < 0.0001] = 0
                        line_logits = sparse.csc_matrix(line_logits)
                    all_logits[ids] = line_logits

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return all_transcriptions, all_logits, all_logit_coords


def merge_transcriptions_and_logits(transcription_parts, logits_parts):
    logits_parts_shrinked = []
    for transcription, logits in zip(transcription_parts, logits_parts):
        logits_parts_shrinked.append(logits[:len(transcription)])

    result_transcription = transcription_parts[0]
    result_logits = logits_parts_shrinked[0]

    for transcription, logits in zip(transcription_parts[1:], logits_parts_shrinked[1:]):
        overlap = find_best_overlap(result_transcription, transcription)
        result_transcription = result_transcription[:-overlap // 2] + transcription[overlap // 2:]
        result_logits = np.concatenate([result_logits[:-overlap // 2], logits[overlap // 2:]], axis=0)

    return result_transcription, result_logits


def find_best_overlap(text1, text2):
    max_overlap = min(len(text1), len(text2))

    best_cer = 1
    best_overlap = 0

    for i in range(1, max_overlap+1):
        s1 = text1[-i:]
        s2 = text2[:i]
        cer = levenshtein_distance(list(s1), list(s2)) / len(s1)

        if cer < best_cer:
            best_cer = cer
            best_overlap = i

    return best_overlap
