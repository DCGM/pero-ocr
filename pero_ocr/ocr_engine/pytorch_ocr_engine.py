# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
from torch import nn
import numpy as np
from functools import partial
from .line_ocr_engine import BaseEngineLineOCR
import sys


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


class PytorchEngineLineOCR(BaseEngineLineOCR):
    def __init__(self, json_def, device, batch_size=8):
        super(PytorchEngineLineOCR, self).__init__(json_def, device, batch_size=batch_size)

        self.net_subsampling = 4
        self.characters = list(self.characters) + [u'\u200B']

        self._load_exported_model()

        if self.embed_id == "mean":
            self.embed_id = self.get_mean_embed_id()

    def get_mean_embed_id(self):
        return self.model.embeddings_layer.weight.shape[0] - 1

    def _load_exported_model(self):
        if self.device.type == "cpu":
            self.checkpoint += ".cpu"

        self.model = torch.jit.load(self.checkpoint, map_location=self.device)
        self.model = self.model.to(self.device)

    def run_ocr(self, batch_data):
        with torch.no_grad():
            batch_data = torch.from_numpy(batch_data).to(self.device).float() / 255.0
            batch_data = batch_data.permute(0, 3, 1, 2)

            if self.embed_id is not None:
                ids_embedding = torch.LongTensor([self.embed_id] * batch_data.shape[0]).to(self.device)
                logits = self.model(batch_data, ids_embedding)

            else:
                logits = self.model(batch_data)

            decoded = greedy_decode_ctc(logits, self.characters)
            logits = logits.permute(0, 2, 1).cpu().numpy()

        return decoded, logits
