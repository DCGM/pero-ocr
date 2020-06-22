import numpy as np
import torch


class HiddenState:
    def __init__(self, h):
        self._h = h

    def __getitem__(self, indices):
        h = self._for_every(lambda h: h[:, indices])
        return HiddenState(h)

    def _for_every(self, op):
        if isinstance(self._h, tuple):
            return tuple(op(part) for part in self._h)
        else:
            return op(self._h)

    def output(self):
        h = self._first()
        return h[-1]

    def _first(self):
        if isinstance(self._h, tuple):
            return self._h[0]
        else:
            return self._h

    def prepare_for_torch(self):
        return self._h

    def __setitem__(self, idx, other):
        if isinstance(self._h, tuple):
            for dst, src in zip(self._h, other._h):
                dst[:, idx] = src
        else:
            self._h[:, idx] = other._h

    def __add__(self, other):
        if isinstance(self._h, tuple):
            assert(isinstance(other._h, tuple))
            assert(len(self._h) == len(other._h))

        if self._first().size == 0:
            new_h = other._h
        elif other._first().size == 0:
            new_h = self._h
        else:
            if isinstance(self._h, tuple):
                new_h = tuple(torch.cat([s, o], axis=1) for s, o in zip(self._h, other._h))
            else:
                new_h = torch.cat([self._h, other._h], axis=1)

        return HiddenState(new_h)


class LMWrapper:
    def __init__(self, lm, decoder_symbols, lm_on_gpu=False):
        self._lm = lm
        self._start_symbol = '</s>'
        self._lm_device = torch.device('cuda:0') if lm_on_gpu else torch.device('cpu')

        self._lm.eval()
        self._lm.to(self._lm_device)

        self._dict = {}
        for i, c in enumerate(decoder_symbols):
            self._dict[i] = self._lm.vocab[c]

    def advance_h0(self, x, h0):
        with torch.no_grad():
            pyth_h = h0.prepare_for_torch()
            pyth_x = torch.from_numpy(x).to(dtype=torch.long, device=self._lm_device).unsqueeze(1) + self._lm._unused_prefix_len
            _, h_new = self._lm.model(pyth_x, pyth_h)
        return HiddenState(h_new)

    def log_probs(self, h):
        with torch.no_grad():
            pyth_h = h.output()
            y = self._lm.decoder(pyth_h)

            if len(y.shape) == 3:
                assert(y.shape[1] == 1)
                y = y[0]

        return y.detach().to('cpu').numpy()[:, self._lm._unused_prefix_len:]

    def eos_scores(self, h):
        with torch.no_grad():
            pyth_h = h.output()
            y = self._lm.decoder(pyth_h)

        if len(y.shape) == 3:
            assert(y.shape[1] == 1)
            y = y[0]

        return y.detach().to('cpu').numpy()[:, self._lm.vocab['</s>']]

    def initial_h(self, batch_size):
        with torch.no_grad():
            h0 = self._lm.model.init_hidden(batch_size)
            start_input = self._lm.vocab[self._start_symbol]
            x1 = torch.tensor([[start_input]]).to(self._lm_device)
            _, h1 = self._lm.model(x1, h0)
        return HiddenState(h1)

    def translate(self, symbols):
        return np.vectorize(self._dict.get)(symbols)
