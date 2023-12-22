# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import numpy as np
from .line_ocr_engine import BaseEngineLineOCR
from pero_ocr.ocr_engine import transformer

import sys


class TransformerEngineLineOCR(BaseEngineLineOCR):
    def __init__(self, json_def, device, batch_size=4):
        super(TransformerEngineLineOCR, self).__init__(json_def, device, batch_size=batch_size, model_type="transformer")

        self.characters = list(self.characters) + [u'\u200B', '']

        self.sentence_boundary_ind = len(self.characters) - 2
        self.ignore_ind = len(self.characters) - 1

        self.net = transformer.build_net(net=self.net_name,
                                         input_height=self.line_px_height,
                                         input_channels=3,
                                         nb_output_symbols=len(self.characters) - 2)

        print(self.net)

        self.net.load_state_dict(torch.load(self.checkpoint))
        self.net.eval()
        self.net = self.net.to(device)

    def run_ocr(self, batch_data):
        with torch.no_grad():
            batch_data = np.transpose(batch_data, (0, 3, 1, 2))

            if batch_data.shape[3] < 1088:
                new_batch_data = np.zeros((batch_data.shape[0], batch_data.shape[1], batch_data.shape[2], 1088), dtype=batch_data.dtype)
                s = (1088 - batch_data.shape[3]) // 2
                new_batch_data[:, :, :, s:s+batch_data.shape[3]] = batch_data
                batch_data = new_batch_data

            labels, logits = self.transcribe_batch(batch_data, is_cached=True)

            logits = logits.cpu().numpy()
            decoded = self.decode(labels)

        return decoded, logits

    def transcribe_batch(self, inputs, is_cached=False):
        lines = torch.from_numpy(inputs).to(self.device).float()
        lines /= 255.0

        encoded_lines = self.net.encode(lines)
        partial_transcripts = torch.tensor([self.sentence_boundary_ind] * len(inputs), dtype=torch.long,
                                           device=self.device).unsqueeze(0)
        alive_mask = torch.full((len(inputs),), 1, dtype=torch.long, device=self.device)

        _, batch, dim_model = encoded_lines.shape  # this is weird
        label_embs: torch.Tensor = torch.empty((0, batch, dim_model)).to(self.device)

        logits = []

        while True:
            label_embs = torch.cat((label_embs, self.net.dec_embeder(partial_transcripts[-1, :]).unsqueeze(0)))
            transformed = self.net.trans_decoder.infer(self.net.pos_encoder(label_embs), encoded_lines,
                                                       is_cached=is_cached)
            last_logits = self.net.dec_out_proj(transformed)
            logits.append(last_logits)

            samples = torch.argmax(last_logits, dim=-1)

            surviving_lines = (samples != self.sentence_boundary_ind)
            alive_mask *= surviving_lines

            if sum(alive_mask) == 0:
                break

            if len(partial_transcripts) > inputs.shape[-1] // 4:  # four pixels per letter is already ridiculous
                print(f'The transcription is getting way too long ({len(partial_transcripts)}) for the line '
                      f'({inputs.shape}), aborting it at shape {partial_transcripts.shape}')
                break

            partial_transcripts = torch.cat([partial_transcripts, samples.unsqueeze(0)], dim=0)

        outs = self.postprocess_decoded(partial_transcripts[1:].permute(1, 0), self.ignore_ind, self.sentence_boundary_ind)

        logits = torch.stack(logits).permute(1, 0, 2)

        return outs, logits

    def postprocess_decoded(self, transcripts, ignore_ind, sentence_boundary_ind):
        outputs = []
        for line in transcripts:
            legit_transcription = []
            for s in line:
                if s == sentence_boundary_ind:
                    break
                elif s == ignore_ind:
                    continue
                else:
                    legit_transcription.append(s)
            outputs.append(torch.tensor(legit_transcription, device=transcripts.device))

        return outputs

    def decode(self, labels):
        outputs = []
        for line_labels in labels:
            outputs.append(''.join([self.characters[c] for c in line_labels]))

        if self.music_translator is not None:
            outputs = self.music_translator(outputs, to_longer=True)

        return outputs

