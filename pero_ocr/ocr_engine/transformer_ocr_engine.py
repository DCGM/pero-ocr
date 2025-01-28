# -*- coding: utf-8 -*-
from __future__ import print_function
import logging
import torch
import numpy as np
from .line_ocr_engine import BaseEngineLineOCR
from pero_ocr.ocr_engine import transformer


class TransformerEngineLineOCR(BaseEngineLineOCR):
    def __init__(self, json_def, device, batch_size=16, substitute_output_atomic: bool = True):
        super(TransformerEngineLineOCR, self).__init__(json_def, device, batch_size=batch_size,
                                                       model_type="transformer",
                                                       substitute_output_atomic=substitute_output_atomic)

        self.characters = list(self.characters) + [u'\u200B', '']

        self.sentence_boundary_ind = len(self.characters) - 2
        self.ignore_ind = len(self.characters) - 1

        self.logger = logging.getLogger(__name__)

        self.exported = False
        self.net = None
        self.load_net()
        self.logger.info(self.net)

        self.max_decoded_seq_length = 210

    def load_net(self):
        if self.device.type == "cpu":
            self.checkpoint += ".cpu"

        if 'exported' in self.config and self.config['exported']:
            net = torch.jit.load(self.checkpoint, map_location=self.device)
            self.exported = True

        else:
            net = transformer.build_net(net=self.net_name,
                                        input_height=self.line_px_height,
                                        input_channels=3,
                                        nb_output_symbols=len(self.characters) - 2)

            net.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
            net.eval()

        self.net = net.to(self.device)

    def run_ocr(self, batch_data):
        with torch.no_grad():
            batch_data = np.transpose(batch_data, (0, 3, 1, 2))

            if batch_data.shape[3] < 1088:
                new_batch_data = np.zeros((batch_data.shape[0], batch_data.shape[1], batch_data.shape[2], 1088), dtype=batch_data.dtype)
                s = (1088 - batch_data.shape[3]) // 2
                new_batch_data[:, :, :, s:s+batch_data.shape[3]] = batch_data
                batch_data = new_batch_data

            if self.exported:
                labels, logits = self.transcribe_batch_exported(batch_data)
            else:
                labels, logits = self.transcribe_batch(batch_data, is_cached=True)

            logits = logits.cpu().numpy()
            decoded = self.decode(labels)

        return decoded, logits

    def transcribe_batch_exported(self, inputs):
        lines = torch.from_numpy(inputs).to(self.device).float() / 255.0

        encoded_lines = self.net.encode(lines)
        encoded_lines = self.net.adapt(encoded_lines)

        partial_transcriptions = torch.tensor([self.sentence_boundary_ind] * lines.shape[0], dtype=torch.long,
                                              device=self.device).unsqueeze(1)
        alive_mask = torch.full((lines.shape[0],), 1, dtype=torch.long, device=self.device)

        logits = []

        for counter in range(self.max_decoded_seq_length):
            step_logits = self.net.decode_step(encoded_lines, partial_transcriptions)
            logits.append(step_logits)

            sampled_characters = torch.argmax(step_logits, dim=-1)

            surviving_lines = (sampled_characters != self.sentence_boundary_ind)
            alive_mask *= surviving_lines

            if sum(alive_mask) == 0:
                break

            if partial_transcriptions.shape[0] > lines.shape[-1] // 4:
                self.logger.warning(f'The transcription is getting way too long ({len(partial_transcriptions)}) for '
                                    f'the line ({lines.shape}), aborting it at shape {partial_transcriptions.shape}')
                break

            partial_transcriptions = torch.cat([partial_transcriptions, sampled_characters.unsqueeze(1)], dim=1)

        outs = self.postprocess_decoded(partial_transcriptions[:, 1:], self.ignore_ind, self.sentence_boundary_ind)
        logits = torch.stack(logits).permute(1, 0, 2)

        return outs, logits

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

        for counter in range(self.max_decoded_seq_length):
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
                self.logger.warning(f'The transcription is getting way too long ({len(partial_transcripts)}) for '
                                    f'the line ({inputs.shape}), aborting it at shape {partial_transcripts.shape}')
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
        return outputs

