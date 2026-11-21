"""A hand-weighted LM in brnolm's python interface (model, decoder, vocab), not torchscript: the decoder and
LMWrapper tests set its weights to get exact, hand-computable scores. The torchscript LM that pero-ocr loads in
production is the stand-in in test/tiny_models.py."""
import torch


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._model_i = torch.nn.Embedding(4, 1)
        self._model_i.requires_grad_(False)
        self._model_i.weight[0, 0] = 0
        self._model_i.weight[1, 0] = 1
        self._model_i.weight[2, 0] = 2
        self._model_i.weight[3, 0] = 3

        self._model_r = torch.nn.Linear(1, 1)
        self._model_r.requires_grad_(False)
        self._model_r.weight[0, 0] = 2
        self._model_r.bias[0] = -1

    def forward(self, xs, hs):
        out_h = "rubbish, never to be accessed"
        last_h = self._model_i(xs)[:, :, 0] + self._model_r(hs)
        return out_h, last_h

    def init_hidden(self, bsz):
        ref = self._model_r.weight
        return torch.ones((1, bsz, 1), dtype=ref.dtype, device=ref.device) * 10

    def process_seq(self, seq, h0):
        h = h0
        hs = []
        for x in seq:
            _, h = self.forward(x.unsqueeze(0), h)
            hs.append(h)

        return torch.cat(hs, dim=1)


class DummyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._model_o = torch.nn.Linear(1, 4)
        self._model_o.requires_grad_(False)
        self._model_o.weight[0, 0] = -100
        self._model_o.weight[1, 0] = 2
        self._model_o.weight[2, 0] = 0
        self._model_o.weight[3, 0] = 5
        self._model_o.bias[0] = -100
        self._model_o.bias[1] = 1
        self._model_o.bias[2] = 3
        self._model_o.bias[3] = -4

    def forward(self, hs):
        return self._model_o(hs)

    def neg_log_prob(self, hs, targets):
        preds = self.forward(hs)
        targets_flat = targets.view(-1)
        preds_flat = preds.view(-1, preds.size(-1))

        return preds_flat[torch.arange(0, preds_flat.shape[0]), targets_flat].sum(), preds_flat.size(0)


class DummyLm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyModel()
        self.decoder = DummyDecoder()
        self.vocab = {
            '</s>': 0,
            'a': 1,
            'b': 2,
            'c': 3,
        }

        self._unused_prefix_len = 1

    def single_sentence_nll(self, sentence, prefix):
        sentence_ids = [self.vocab[c] for c in sentence]

        if prefix:
            prefix_id = self.vocab[prefix]
            tensor = torch.tensor([prefix_id] + sentence_ids).view(-1, 1)
        else:
            tensor = torch.tensor(sentence_ids).view(-1, 1)

        h0 = self.model.init_hidden(1)
        o = self.model.process_seq(tensor[:-1], h0).detach()

        if prefix:
            nll, _ = self.decoder.neg_log_prob(o, tensor[1:])
        else:
            nll, _ = self.decoder.neg_log_prob(torch.cat([h0[0][0].unsqueeze(1), o]), tensor)

        return nll.item()
