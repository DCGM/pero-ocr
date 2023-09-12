# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
from torch import nn
import numpy as np
from functools import partial
from .line_ocr_engine import BaseEngineLineOCR
import sys
import multiprocessing
from typing import List
import pickle
import zmq
import time
import msgpack
import msgpack_numpy as m


# scores_probs should be N,C,T, blank is last class
def greedy_decode_ctc(scores_probs, chars):
    if len(scores_probs.shape) == 2:
        scores_probs = torch.cat((scores_probs[:, 0:1], scores_probs), axis=1)
        scores_probs[:, 0] = -1000
        scores_probs[-1, 1] = 1000
    else:
        scores_probs = torch.cat((scores_probs[:, :, 0:1], scores_probs), axis=2)
        scores_probs[:, :, 0] = -1000
        scores_probs[:, -1, 0] = 1000

    best = torch.argmax(scores_probs, 1) + 1
    mask = best[:, :-1] == best[:, 1:]
    best = best[:, 1:]
    best[mask] = 0
    best[best == scores_probs.shape[1]] = 0
    best = best.cpu().numpy() - 1

    outputs = []
    for line in best:
        line = line[np.nonzero(line >= 0)]
        outputs.append(''.join([chars[c] for c in line]))
    return outputs


class NetProcess(multiprocessing.Process):
    def __init__(self, model_file, embed_id=None):
        super(NetProcess, self).__init__(daemon=True)
        self.model_file = model_file
        self.device = None
        self.embed_id = embed_id
        self.counter = 0


    def _load_exported_model(self):
        if self.device.type == "cpu":
            self.checkpoint += ".cpu"
        self.model = torch.jit.load(self.model_file, map_location=self.device)
        self.model = self.model.to(self.device)

    def get_embedding(self):
        if self.embed_id == "mean":
            self.embed_id = self.get_mean_embed_id()
        return self.model.embeddings_layer.weight.shape[0] - 1

    def run(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_exported_model()
        with torch.no_grad():
            while True:
                with open('log_ocr.txt', 'a') as f:
                    t1 = time.time()
                    batch_data = self.socket.recv()
                    print('XXX 1', time.time() - t1, file=f)
                    batch_data = pickle.loads(batch_data)
                    print('XXX 2', time.time() - t1, file=f)

                    batch_data = torch.from_numpy(batch_data).to(self.device).float() / 255.0
                    batch_data = batch_data.permute(0, 3, 1, 2)
                    print('XXX 3', time.time() - t1, file=f)

                    if self.embed_id is not None:
                        ids_embedding = torch.LongTensor([self.embed_id] * batch_data.shape[0]).to(self.device)
                        logits = self.model(batch_data, ids_embedding)
                    else:
                        logits = self.model(batch_data)
                    logits = logits.cpu().numpy()
                    mpxs = batch_data.shape[2] * batch_data.shape[3] / 1e6 / (time.time() - t1)
                    print('XXX 5', time.time() - t1, batch_data.shape, mpxs, file=f)
                    msg = pickle.dumps(logits)
                    self.socket.send(msg)
                    print('XXX 6', time.time() - t1, len(msg) / 1e6, logits.shape, file=f)
                    self.counter += 1
                    if self.counter % 100 == 0:
                        torch.cuda.empty_cache()

class PytorchEngineLineOCR(BaseEngineLineOCR):
    def __init__(self, json_def, gpu_id=0, batch_size=32, start_engines=True):
        super(PytorchEngineLineOCR, self).__init__(json_def, gpu_id=gpu_id, batch_size=batch_size)

        self.net_subsampling = 4
        self.characters = list(self.characters) + [u'\u200B']
        #multiprocessing.set_start_method('spawn')

        if start_engines:
            print('Starting OCR engines...')
            NetProcess(self.checkpoint, self.embed_id).start()
        else:
            print('NOT starting OCR engines...')
        self.socket = None

    def get_socket(self):
        if not self.socket:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("tcp://localhost:5555")
        return self.socket

    def run_ocr(self, batch_data):
        socket = self.get_socket()
        with torch.no_grad():
            #batch_data = torch.from_numpy(batch_data).to(self.device).float() / 255.0
            #batch_data = batch_data.permute(0, 3, 1, 2)

            #if self.embed_id is not None:
            #    ids_embedding = torch.LongTensor([self.embed_id] * batch_data.shape[0]).to(self.device)
            #    logits = self.model(batch_data, ids_embedding)
            #else:

            #logits = self.model(batch_data).cpu().numpy()
            socket.send(pickle.dumps(batch_data))
            logits = pickle.loads(socket.recv())
            logits = torch.from_numpy(logits)

            decoded = greedy_decode_ctc(logits, self.characters)
            logits = logits.permute(0, 2, 1).numpy()
        return decoded, logits
