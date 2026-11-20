"""Tiny torchscript stand-ins for the networks pero-ocr loads.

pero-ocr loads every network as a torchscript file, so the tests do not need the real architectures, only modules
that follow the same calling contract. The stand-ins have no learned weights; their output is a deterministic function
of the input image, so tests can predict it:

* "ink" is **bright** pixels (mean intensity > 0.5), so the test images are bright text on black, the inverse of
  real documents. The reason is padding: the engines fill everything that is not image with zeros, i.e. black. The
  OCR engines pad shorter lines of a batch to the width of the longest one, the layout engine pads the page to a size
  divisible by its downsampling, and the line cropper fills the part of a line outside the page with black. Real
  models learned to ignore that padding; the stand-ins cannot learn. If dark pixels were ink, padding would read as
  text: extra characters at the end of every line shorter than its batch, a baseline along the padded page border,
  and outputs that depend on how the lines were batched. With bright ink, padding is simply "no ink".
* CTC OCR: every horizontal run of ink becomes one character. Without an embedding id it is ``characters[1]``,
  with embedding id ``e`` it is ``characters[1 + e % (n_chars - 1)]``.
* Transformer OCR: a line with ``n`` runs of ink decodes to ``n`` characters ``characters[1], characters[2], ...``
  (cycling over ``characters[1:]``), then the end-of-sequence token. With ``stop=False`` it never emits the end token.
* ParseNet: a baseline is the bottom row of every block of ink, ascender/descender heights are constants and there
  are no line endpoints or region separators.
* Language model (brnolm torchscript format): the state is the last character seen, ignoring the line separator
  ``</s>``, so the state carries over line ends. After ``LM_PREFERS[0]`` the LM strongly prefers ``LM_PREFERS[1]``;
  every other context gives a uniform distribution.

The frozen copies in ``test/engines/consistency/data`` are regenerated with
``python test/engines/consistency/data/regenerate.py``.
"""
import json
import os
import shutil
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

CHARACTERS = ['�', 'a', 'b', 'c', 'd', ' ']
LINE_PX_HEIGHT = 40
LM_PREFERS = ('a', 'b')  # after 'a', the tiny LM strongly prefers 'b'
LM_PREFERENCE = 6.0
PARSENET_ASCENDER = 10.0
PARSENET_DESCENDER = 3.0


def _ink_columns(x: Tensor) -> Tensor:
    """N,3,H,W image in [0, 1] -> N,T fraction of ink per 4-px column block (T = W // 4)."""
    ink = (x.mean(dim=1) > 0.5).float().amax(dim=1)  # N,W: 1 where the column contains ink
    return F.avg_pool1d(ink.unsqueeze(1), 4, 4).squeeze(1)


class TinyCTC(nn.Module):
    def __init__(self, n_characters: int, embed_num: int = 4):
        super().__init__()
        self.n_out = n_characters + 1  # + blank, which is the last class
        self.n_characters = n_characters
        self.embeddings_layer = nn.Embedding(embed_num, 1)

    def forward(self, x: Tensor, embed_ids: Optional[Tensor] = None) -> Tensor:
        """N,3,H,W -> N,C,T logits, blank is the last class."""
        ink = _ink_columns(x)
        n = ink.shape[0]
        if embed_ids is None:
            char_ids = torch.ones(n, dtype=torch.long, device=x.device)
        else:
            char_ids = 1 + embed_ids % (self.n_characters - 1)
        # N,T: +11 on ink, -9 on background; never exactly 0, which the sparse logit format treats as "pruned"
        score = 20.0 * ink - 9.0
        one_hot = F.one_hot(char_ids, self.n_out).float()  # N,C
        logits = -10.0 + one_hot.unsqueeze(2) * (score.unsqueeze(1) + 10.0)
        logits[:, -1, :] = -score
        return logits


class TinyTransformer(nn.Module):
    def __init__(self, n_characters: int, stop: bool = True):
        super().__init__()
        self.n_characters = n_characters
        self.n_out = n_characters + 2  # + end of sequence + ignored symbol
        self.eos = n_characters
        self.stop = stop
        self.dtype_holder = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)

    @torch.jit.export
    def encode(self, x: Tensor) -> Tensor:
        """N,3,H,W -> T,N,1."""
        return _ink_columns(x).t().unsqueeze(2)

    @torch.jit.export
    def adapt(self, encoded: Tensor) -> Tensor:
        return encoded

    @torch.jit.export
    def decode_step(self, encoded: Tensor, partial: Tensor) -> Tensor:
        """encoded T,N,1; partial N,L (starts with the start token) -> N,C logits of the next symbol."""
        ink = (encoded[:, :, 0] > 0.5).float()  # T,N
        n_runs = ink[0] + (ink[1:] * (1.0 - ink[:-1])).sum(0)  # N
        step = partial.shape[1] - 1
        char_id = 1 + step % (self.n_characters - 1)
        ids = torch.full((partial.shape[0],), char_id, dtype=torch.long, device=encoded.device)
        if self.stop:
            ids = torch.where(n_runs > step, ids, torch.full_like(ids, self.eos))
        logits = -10.0 + 20.0 * F.one_hot(ids, self.n_out).float()
        return logits.to(self.dtype_holder.dtype)


class TinyParseNet(nn.Module):
    def __init__(self, ascender: float = PARSENET_ASCENDER, descender: float = PARSENET_DESCENDER):
        super().__init__()
        self.ascender = ascender
        self.descender = descender

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """N,3,H,W -> (N,5,H,W maps, unused). Channels: ascender, descender, baseline, endpoints, separators."""
        ink = (x.mean(dim=1, keepdim=True) > 0.5).float()
        ink_below = F.pad(ink[:, :, 1:, :], (0, 0, 0, 1))
        baseline = ink * (1.0 - ink_below)
        zeros = torch.zeros_like(ink)
        maps = torch.cat([zeros + self.ascender, zeros + self.descender, baseline, zeros, zeros], dim=1)
        return maps, maps


class TinyLMModel(nn.Module):
    """brnolm-style LM body: (tokens, (h, c)) -> (outputs, (h, c)); the state is a one-hot of the last non-</s> token."""

    def __init__(self, vocab_size: int, eos: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos = eos
        self.register_buffer('device_holder', torch.zeros(1))

    def forward(self, x: Tensor, h: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        state = h[0][-1]  # B,V
        outputs = []
        for t in range(x.shape[1]):
            token = x[:, t]
            one_hot = F.one_hot(token, self.vocab_size).to(state.dtype)
            keep = (token == self.eos).to(state.dtype).unsqueeze(1)
            state = keep * state + (1.0 - keep) * one_hot
            outputs.append(state)
        return torch.stack(outputs, dim=1), (state.unsqueeze(0), h[1])

    @torch.jit.export
    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        zeros = torch.zeros(1, batch_size, self.vocab_size, device=self.device_holder.device)
        return zeros, zeros.clone()


class TinyLMDecoder(nn.Module):
    """State -> log-probabilities of the next token: uniform, except the preferences given as a V x V matrix."""

    def __init__(self, preferences: Tensor):
        super().__init__()
        self.register_buffer('preferences', preferences)

    def forward(self, o: Tensor) -> Tensor:
        return F.log_softmax(o @ self.preferences, dim=-1)


class _LanguageModelParts:
    """What brnolm's torchscript_export() reads: .model, .decoder and .vocab."""

    def __init__(self, model, decoder, vocab):
        self.model, self.decoder, self.vocab = model, decoder, vocab


def _save_script(module: nn.Module, path: str) -> str:
    """Save as torchscript; the engines append '.cpu' to the path on CPU, so write both."""
    scripted = torch.jit.script(module.eval())
    scripted.save(path)
    shutil.copyfile(path, path + '.cpu')
    return path


def write_ctc(directory: str, characters=CHARACTERS, embed_num: Optional[int] = None, extra: dict = None) -> str:
    """Write an OCR json + torchscript CTC model, return the json path."""
    os.makedirs(directory, exist_ok=True)
    _save_script(TinyCTC(len(characters), embed_num=embed_num or 4), os.path.join(directory, 'model.pt'))
    config = {
        'checkpoint': './model.pt',
        'line_px_height': LINE_PX_HEIGHT,
        'line_vertical_scale': 1.0,
        'net_name': 'TINY_CTC',
        'characters': list(characters),
    }
    if embed_num is not None:
        config['embed_num'] = embed_num
    config.update(extra or {})
    json_path = os.path.join(directory, 'ocr_engine.json')
    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(config, f, ensure_ascii=False, indent=1)
    return json_path


def write_transformer(directory: str, characters=CHARACTERS, stop: bool = True, extra: dict = None) -> str:
    """Write an OCR json + torchscript transformer model, return the json path."""
    os.makedirs(directory, exist_ok=True)
    _save_script(TinyTransformer(len(characters), stop=stop), os.path.join(directory, 'model.pt'))
    config = {
        'checkpoint': './model.pt',
        'exported': True,
        'line_px_height': LINE_PX_HEIGHT,
        'line_vertical_scale': 1.0,
        'net_name': 'TINY_TRANSFORMER',
        'characters': list(characters),
    }
    config.update(extra or {})
    json_path = os.path.join(directory, 'ocr_engine.json')
    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(config, f, ensure_ascii=False, indent=1)
    return json_path


def write_parsenet(directory: str) -> str:
    """Write a torchscript ParseNet stand-in, return the model path (without the '.cpu' suffix)."""
    os.makedirs(directory, exist_ok=True)
    return _save_script(TinyParseNet(), os.path.join(directory, 'parsenet.pt'))


def write_lm(directory: str, characters=CHARACTERS) -> str:
    """Write a language model in brnolm's torchscript zip format, return its path.

    The vocabulary is ['</s>', '<unk>'] + characters: pero-ocr maps OCR character i to LM token i + 2."""
    from brnolm.language_models.language_model import torchscript_export
    os.makedirs(directory, exist_ok=True)
    vocab_list = ['</s>', '<unk>'] + list(characters)
    vocab = {symbol: i for i, symbol in enumerate(vocab_list)}
    preferences = torch.zeros(len(vocab_list), len(vocab_list))
    preferences[vocab[LM_PREFERS[0]], vocab[LM_PREFERS[1]]] = LM_PREFERENCE
    parts = _LanguageModelParts(TinyLMModel(len(vocab_list)).eval(), TinyLMDecoder(preferences).eval(), vocab)
    path = os.path.join(directory, 'lm.pt')
    torchscript_export(parts, path)
    return path


WRITERS = {
    'ctc': lambda d: write_ctc(os.path.join(d, 'ctc')),
    'transformer': lambda d: write_transformer(os.path.join(d, 'transformer')),
    'parsenet': lambda d: write_parsenet(os.path.join(d, 'parsenet')),
    'lm': lambda d: write_lm(os.path.join(d, 'lm')),
}


def write_all(directory: str, names=None) -> dict:
    """Write the stand-ins ``names`` (default: all of WRITERS) into subdirectories of ``directory``."""
    return {name: WRITERS[name](directory) for name in (names or WRITERS)}

