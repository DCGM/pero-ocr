# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import cv2
import numpy as np
from os.path import isabs, realpath, join,dirname
from scipy import sparse
import tensorflow as tf

from .CTC_nets import build_eval_net, line_nets
from .softmax import softmax


class EngineLineOCR(object):
    def __init__(self, json_def, gpu_id=0, batch_size=8):
        with open(json_def, 'r', encoding='utf8') as f:
            self.config = json.load(f)

        self.line_px_height = self.config['line_px_height']
        self.line_vertical_scale = self.config['line_vertical_scale']

        if isabs(self.config['checkpoint']):
            self.checkpoint = self.config['checkpoint']
        else:
            self.checkpoint = realpath(join(dirname(json_def), self.config['checkpoint']))

        self.characters = self.config['characters']
        self.net_name = self.config['net_name']
        self.gpu_id = gpu_id

        self.batch_size = batch_size
        self.data_shape = [self.batch_size, self.line_px_height, None, 3]

        self.net_graph = tf.Graph()
        tf.reset_default_graph()
        with self.net_graph.as_default():
            net = line_nets[self.net_name]
            (saver, input_data, _, seq_len, logits, logits_t, decoded, _) = build_eval_net(
                [self.batch_size, self.line_px_height, None, 3], len(self.characters), net)

        self.net_subsampling = 1
        self.out_decoded = decoded
        self.out_logits = logits
        self.in_seq_len = seq_len
        self.saver = saver
        self.input_data = input_data
        self.line_padding_px = 32

        if gpu_id is None:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto(device_count={'GPU': 1})
            config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.net_graph, config=config)
        self.saver.restore(self.session, self.checkpoint)
        self.data_shape[2] = 128
        out_logits, = self.session.run(
            [self.out_logits],
            feed_dict={self.input_data: np.zeros(self.data_shape, dtype=np.uint8)}
        )
        self.net_subsampling = self.data_shape[2] / out_logits.shape[1]
        self.data_shape[2] = None

    def process_lines(self, lines):
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

        #  process all lines ordered by their length
        line_ids = [x for x, y in sorted(enumerate(lines), key=lambda x: x[1].shape[1])]
        while line_ids:
            batch_line_ids = line_ids[:self.batch_size]
            line_ids = line_ids[self.batch_size:]
            max_width = 0
            for ids in batch_line_ids:
                max_width = max(max_width, lines[ids].shape[1])

            max_width = int(np.ceil(max_width / 32.0) * 32)

            batch_data = np.zeros(
                [self.batch_size, self.line_px_height, max_width + 2*self.line_padding_px, 3], dtype=np.uint8)
            for data, ids in zip(batch_data, batch_line_ids):
                data[:, self.line_padding_px:self.line_padding_px+lines[ids].shape[1], :] = lines[ids]

            seq_lengths = np.ones([self.batch_size], dtype=np.int32) * batch_data.shape[2] / self.net_subsampling

            out_decoded, out_logits = self.session.run(
                [self.out_decoded, self.out_logits],
                feed_dict={self.input_data: batch_data, self.in_seq_len: seq_lengths})

            out_decoded = out_decoded[0]
            for i, ids in enumerate(batch_line_ids):
                pos, = np.nonzero(out_decoded.indices[:, 0] == i)
                tmp_string = ''
                if pos.size:
                    for val in out_decoded.values[pos]:
                        tmp_string += self.characters[val]
                all_transcriptions[ids] = tmp_string

            for ids, line_logits in zip(batch_line_ids, out_logits):
                line_logits = line_logits[
                    int(self.line_padding_px // self.net_subsampling - 2):int(lines[ids].shape[1] // self.net_subsampling + 8)]
                line_probs = softmax(line_logits, axis=1)
                line_logits[line_probs < 0.0001] = 0
                line_logits = sparse.csc_matrix(line_logits)
                all_logits[ids] = line_logits

        return all_transcriptions, all_logits


def test_line_ocr(line_list, ocr_engine_json):
    ocr_engine = EngineLineOCR(ocr_engine_json, gpu_id=1)

    lines = []
    for line in line_list:
        line_img = cv2.imread(line, 1)
        if line_img is None:
            raise ValueError('Error: Could not read image "{}"'.format(line))
        lines.append(line_img)

    transcriptions, logits = ocr_engine.process_lines(lines)

    for transcription, line in zip(transcriptions, lines):
        print(transcription)
        cv2.imshow('out', line)
        if cv2.waitKey() == 27:
            return


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ocr-engine', required=True, help='JSON file with line ocr engine definition.')
    parser.add_argument('--line-list', required=True, help='File containing list of line images.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()

    with open(args.line_list, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    test_line_ocr(lines, args.ocr_engine)
