# -*- coding: utf-8 -*-
import shutil
import time
import os
import argparse
import re
import pickle
import json

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class EngineRepairCNN(object):
    def __init__(self, json_path, use_cpu=False):
        with open(json_path, 'r', encoding='utf8') as f:
            config = json.load(f)
        with open(config['chars_path'], 'rb') as handle:
            rb = pickle.load(handle)
        self.from_char = rb['from_char']
        self.to_char = rb['to_char']
        self.chars = rb['chars']
        self.max_labels = config['max_labels']
        self.height = config['height']
        self.max_width = config['max_width']

        repair_model = config['repair_model']
        inpainting_model = config['inpainting_model']

        if repair_model:
            tf.reset_default_graph()
            saver = tf.train.import_meta_graph(repair_model + '.meta')
            if use_cpu:
                tf_config = tf.ConfigProto(device_count={'GPU': 0})
            else:
                tf_config = tf.ConfigProto(device_count={'GPU': 1})
                tf_config.gpu_options.allow_growth = True
            self.repair_session = tf.Session(config=tf_config)
            saver.restore(self.repair_session, repair_model)

        if inpainting_model:
            tf.reset_default_graph()
            saver = tf.train.import_meta_graph(inpainting_model + '.meta')
            if use_cpu:
                tf_config = tf.ConfigProto(device_count={'GPU': 0})
            else:
                tf_config = tf.ConfigProto(device_count={'GPU': 1})
                tf_config.gpu_options.allow_growth = True
            self.inpainting_session = tf.Session(config=tf_config)
            saver.restore(self.inpainting_session, inpainting_model)

    def repair_line(self, line, transcription):
        labels = self.transcriptions_to_labels([transcription])[0]
        labels = self.clip_or_pad_labels(labels).astype(np.int32)

        line = self.clip_or_pad_image(line)
        line = line[np.newaxis, :, :, :] / 255.

        feed_dict = {'inference_content:0': line,
                     'inference_style:0': line,
                     'inference_transcriptions:0': labels}
        output = self.repair_session.run('inference_op:0', feed_dict=feed_dict)
        return (255*np.clip(output[0, :, :, :], 0, 1)).astype(np.uint8)

    def inpaint_line(self, line, transcription):
        labels = self.transcriptions_to_labels([transcription])[0]
        labels = self.clip_or_pad_labels(labels).astype(np.int32)

        line = self.clip_or_pad_image(line)
        line = line[np.newaxis, :, :, :] / 255.

        feed_dict = {'inference_content:0': line,
                     'inference_style:0': line,
                     'inference_transcriptions:0': labels}
        output = self.inpainting_session.run('inference_op:0', feed_dict=feed_dict)
        return (255*np.clip(output[0, :, :, :], 0, 1)).astype(np.uint8)

    def transcriptions_to_labels(self, transcriptions):
        trans = str.maketrans(''.join(self.from_char), ''.join(self.to_char))
        labels = [np.asarray([ord(x) for x in l.translate(trans)], dtype=np.int32) for l in transcriptions]
        return labels

    def clip_or_pad_labels(self, labels):
        labels_clipped = np.zeros((1, self.max_labels))
        if labels.shape[0] > self.max_labels:
            labels_clipped[0, :] = labels[:self.max_labels]
        else:
            labels_clipped[0, :labels.shape[0]] = labels
        return labels_clipped

    def clip_or_pad_image(self, image):
        if image.shape[1] > self.max_width:
            image_clipped = image[:, :self.max_width, :]
            print('Warning: image line too long, please only attempt repairing lines shorter than {} pixels'.format(self.max_width))
        else:
            image_clipped = np.zeros((self.height, self.max_width, 3))
            image_clipped[:, :image.shape[1], :] = image
        return image_clipped


    def clip_transcriptions(self, transcriptions, batch_size, max_labels):
        new_transcriptions = np.zeros((1, max_labels))
        for i, transcription in enumerate(transcriptions):
            new_transcription = np.zeros(max_labels)
            if transcription.shape[0] > max_labels:
                new_transcription = transcription[:max_labels]
            else:
                new_transcription[:transcription.shape[0]] = transcription
            new_transcriptions[i,:] = new_transcription
        return new_transcriptions.astype(np.int32)
