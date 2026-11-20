"""Tiny torchscript stand-ins for the networks pero-ocr loads.

pero-ocr loads every network as a torchscript file, so the tests do not need the real architectures, only modules
that follow the same calling contract. The stand-ins have no learned weights; their output is a deterministic function
of the input image, so tests can predict it:

* "ink" is **bright** pixels (mean intensity > 0.5). The engines pad line crops and pages with zeros (black), so
  bright-on-black test images keep the padding inert.
* CTC OCR: every horizontal run of ink becomes one character. Without an embedding id it is ``characters[1]``,
  with embedding id ``e`` it is ``characters[1 + e % (n_chars - 1)]``.
* Transformer OCR: a line with ``n`` runs of ink decodes to ``n`` characters ``characters[1], characters[2], ...``
  (cycling over ``characters[1:]``), then the end-of-sequence token. With ``stop=False`` it never emits the end token.
* ParseNet: a baseline is the bottom row of every block of ink, ascender/descender heights are constants and there
  are no line endpoints or region separators.

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


def write_all(directory: str) -> dict:
    return {
        'ctc': write_ctc(os.path.join(directory, 'ctc')),
        'transformer': write_transformer(os.path.join(directory, 'transformer')),
        'parsenet': write_parsenet(os.path.join(directory, 'parsenet')),
    }

