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
* OCR, CTC and transformer alike, so that a CTC + transformer pair reads the same text: every block of ink (a run of
  columns of which at least a quarter of the pixels are ink) is one character; ``ink_colour()`` gives the colours.
  The green channel picks it: 255 is ``characters[1]`` (white blocks read as ``a``), every ``LEVEL_STEP`` lower the
  next one. Lowering the red channel adds doubt (both read relative to the blue channel, which ``ink_colour()``
  keeps at 255, so that pixels dimmed at the block edges by the line cropper keep the character): the previous character of ``characters[1:]`` (cyclically) becomes
  as likely at doubt 0.5 and more likely above, which lowers the line confidence and gives decoders a real choice.
  The CTC model is peaky like a real one: the character on the first frame of every block, blank on all others; with embedding id ``e``
  every character is shifted by ``e`` places. The transformer emits one character per block, then the
  end-of-sequence token; with ``stop=False`` it never emits the end token.
* ParseNet: the colour of a pixel says which output map it feeds (channels of the image array as the engine gets it):
  - ink (white, any pixel with mean > 0.5): a baseline at the bottom row of every block of ink, with the height of
    the block above it as the ascender and ``PARSENET_DESCENDER_RATIO`` times that as the descender, all in pixels of
    the downsampled map, so the heights follow the size of the "font" and drive the adaptive downsampling;
  - ink whose third channel is dark (yellow, ``LINE_END``): also a line end, which the engine subtracts from the
    baseline (as a blob 5 rows tall) with LINE_END_WEIGHT, so a yellow mark across a line can cut it in two;
  - non-ink whose first channel is bright (red, ``SEPARATOR``): a region separator, which keeps vertically adjacent
    lines in different regions when it lies along the top of the lower line.
* Reading order (REGION_SORTER_TRANSFORMER): regions go by their left edge, then by their top edge, i.e. columns
  left to right, each top to bottom. Only the region boxes matter; the page image is passed in but not used.
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
PARSENET_DESCENDER_RATIO = 0.3
LINE_END = (255, 255, 0)  # colours of the ParseNet stand-in input, see TinyParseNet
SEPARATOR = (255, 0, 0)


LEVEL_STEP = 25  # green levels of the OCR stand-ins' input: 255 is CHARACTERS[1], each LEVEL_STEP lower the next one


def ink_colour(character: str, doubt: float = 0.0) -> Tuple[int, int, int]:
    """Colour of a block the tiny OCR models read as ``character``: the green channel picks the character, the red one
    lowered by ``doubt`` (0 to 0.9) makes the previous character of CHARACTERS[1:] (cyclically) as likely as
    ``character`` at 0.5 and more likely above. The blue channel stays 255, so the block is never a ParseNet line end."""
    level = CHARACTERS.index(character) - 1
    if level < 0 or not 0.0 <= doubt <= 0.9:
        raise ValueError(f'cannot draw {character!r} with doubt {doubt}')
    return int(round(255 * (1.0 - doubt))), 255 - LEVEL_STEP * level, 255


def _frames(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """N,3,H,W image in [0, 1] -> per 4-px column block (T = W // 4), each N,T: the fraction of its columns that are
    ink (at least a quarter of their pixels are ink), and over the ink the character level (from green / blue) and
    the doubt (from red / blue)."""
    ink_px = (x.mean(dim=1) > 0.5).float()  # N,H,W
    n_ink = ink_px.sum(dim=1)  # N,W
    column = (n_ink >= 0.25 * x.shape[2]).float()
    # relative to the blue channel, so that pixels dimmed by interpolation at the edges keep the block's character
    blue = x[:, 2].clamp(min=1e-3)
    green = (x[:, 1] / blue * ink_px).sum(dim=1) / n_ink.clamp(min=1.0)
    red = (x[:, 0] / blue * ink_px).sum(dim=1) / n_ink.clamp(min=1.0)
    ink = F.avg_pool1d(column.unsqueeze(1), 4, 4).squeeze(1)
    weight = F.avg_pool1d(column.unsqueeze(1), 4, 4).squeeze(1).clamp(min=1e-6)
    green = F.avg_pool1d((green * column).unsqueeze(1), 4, 4).squeeze(1) / weight
    red = F.avg_pool1d((red * column).unsqueeze(1), 4, 4).squeeze(1) / weight
    level = torch.round((1.0 - green) * 255.0 / 25.0)  # LEVEL_STEP (torchscript cannot read the global)
    return ink, level, 1.0 - red


class TinyCTC(nn.Module):
    def __init__(self, n_characters: int, embed_num: int = 4):
        super().__init__()
        self.n_out = n_characters + 1  # + blank, which is the last class
        self.n_characters = n_characters
        self.embeddings_layer = nn.Embedding(embed_num, 1)

    def forward(self, x: Tensor, embed_ids: Optional[Tensor] = None) -> Tensor:
        """N,3,H,W -> N,C,T logits, blank is the last class."""
        ink, level, doubt = _frames(x)
        shift = torch.zeros_like(level) if embed_ids is None else embed_ids.to(level.dtype).unsqueeze(1)
        n_letters = self.n_characters - 1
        char_ids = 1 + torch.remainder(level + shift, n_letters).long()  # N,T
        alt_ids = 1 + torch.remainder(level + shift - 1, n_letters).long()
        # peaky like a real CTC model: the character on the first frame of a block, blank on all other frames;
        # +11 / -11 and -10 elsewhere, never exactly 0, which the sparse logit format treats as "pruned"
        on = (ink > 0.5).float()
        start = on * (1.0 - F.pad(on[:, :-1], (1, 0)))  # N,T
        score = 22.0 * start - 11.0
        alt_score = -10.0 + 42.0 * doubt * start  # as likely as the character at doubt 0.5
        logits = torch.full((x.shape[0], self.n_out, ink.shape[1]), -10.0, device=x.device)
        logits.scatter_(1, alt_ids.unsqueeze(1), alt_score.unsqueeze(1))
        logits.scatter_(1, char_ids.unsqueeze(1), (-10.0 + 21.0 * start).unsqueeze(1))
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
        """N,3,H,W -> T,N,3: ink, character level and doubt per 4-px column block."""
        ink, level, doubt = _frames(x)
        return torch.stack([ink, level, doubt], dim=2).transpose(0, 1).to(self.dtype_holder.dtype)

    @torch.jit.export
    def adapt(self, encoded: Tensor) -> Tensor:
        return encoded

    @torch.jit.export
    def decode_step(self, encoded: Tensor, partial: Tensor) -> Tensor:
        """encoded T,N,3; partial N,L (starts with the start token) -> N,C logits of the next symbol: the character of
        the next block of ink (block number L - 1), then the end of sequence."""
        encoded = encoded.float()
        ink = (encoded[:, :, 0] > 0.5).float()  # T,N
        starts = torch.cat([ink[:1], ink[1:] * (1.0 - ink[:-1])], dim=0)
        block = torch.cumsum(starts, dim=0) * ink  # T,N: 1, 2, ... on the blocks, 0 between them
        step = partial.shape[1] - 1
        mask = (block == float(step + 1)).float()
        count = mask.sum(0).clamp(min=1.0)
        level = torch.round((mask * encoded[:, :, 1]).sum(0) / count)
        doubt = (mask * encoded[:, :, 2]).sum(0) / count
        n_letters = self.n_characters - 1
        char_ids = 1 + torch.remainder(level, n_letters).long()  # N
        alt_ids = 1 + torch.remainder(level - 1, n_letters).long()
        logits = torch.full((partial.shape[0], self.n_out), -10.0, device=encoded.device)
        logits.scatter_(1, alt_ids.unsqueeze(1), (-10.0 + 40.0 * doubt).unsqueeze(1))
        logits.scatter_(1, char_ids.unsqueeze(1), torch.full_like(doubt, 10.0).unsqueeze(1))
        if self.stop:
            done = (starts.sum(0) <= float(step)).unsqueeze(1)
            logits = torch.where(done, -10.0 + 20.0 * F.one_hot(torch.full_like(char_ids, self.eos), self.n_out).float(),
                                 logits)
        return logits.to(self.dtype_holder.dtype)


class TinyParseNet(nn.Module):
    def __init__(self, descender_ratio: float = PARSENET_DESCENDER_RATIO):
        super().__init__()
        self.descender_ratio = descender_ratio

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """N,3,H,W -> (N,5,H,W maps, unused). Channels: ascender, descender, baseline, line ends, separators."""
        ink = (x.mean(dim=1, keepdim=True) > 0.5).float()
        # height of the ink above every pixel, the pixel included: at the bottom row of a block, the block's height
        rows = []
        run = torch.zeros_like(ink[:, :, 0, :])
        for y in range(ink.shape[2]):
            run = (run + 1.0) * ink[:, :, y, :]
            rows.append(run)
        ascender = torch.stack(rows, dim=2)
        ink_below = F.pad(ink[:, :, 1:, :], (0, 0, 0, 1))
        baseline = ink * (1.0 - ink_below)
        # a blob around the mark, as the real ParseNet predicts line ends; one row would not survive the smoothing
        line_end = F.max_pool2d(ink * (x[:, 2:3] < 0.5).float(), (5, 1), stride=1, padding=(2, 0))
        separator = (1.0 - ink) * (x[:, 0:1] > 0.5).float()
        maps = torch.cat([ascender, self.descender_ratio * ascender, baseline, line_end, separator], dim=1)
        return maps, maps


class TinyRegionSorter(nn.Module):
    """Reading-order model with the calling contract of the real MultimodalLayoutTransformer.

    forward(boxes B,M,4 [left, top, right, bottom] in [0, 1], query_types B,M (1 = box, 0 = padding),
    images B,3,H,W uint8) -> B,M,M scores, where out[b, i, j] is high when box i comes right after box j.
    The order is by left edge, then by top edge: columns left to right, each top to bottom. The image is not used."""

    def forward(self, x: Tensor, query_types: Tensor, images: Tensor) -> Tensor:
        valid = query_types == 1
        key = x[:, :, 0] + 1e-3 * x[:, :, 1]
        key = torch.where(valid, key, torch.full_like(key, 1e9))
        rank = torch.argsort(torch.argsort(key, dim=1), dim=1)  # B,M: position of every box in the order
        follows = rank.unsqueeze(2) == rank.unsqueeze(1) + 1  # B,M,M: box i comes right after box j
        mask = valid.unsqueeze(2) & valid.unsqueeze(1)
        scores = (follows & mask).to(x.dtype) * 10.0
        return scores.masked_fill(~mask, -1e-9)  # like the real model: padded entries are masked, not removed


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


def write_region_sorter(directory: str) -> str:
    """Write a torchscript reading-order model, return the path a config names (model-cuda.pt).

    TransformerRegionSorter replaces 'cuda' with 'cpu' in the file name when it runs on CPU, so write both, like the
    files of the real models."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, 'model-cuda.pt')
    torch.jit.script(TinyRegionSorter().eval()).save(path)
    shutil.copyfile(path, os.path.join(directory, 'model-cpu.pt'))
    return path


WRITERS = {
    'ctc': lambda d: write_ctc(os.path.join(d, 'ctc')),
    'transformer': lambda d: write_transformer(os.path.join(d, 'transformer')),
    'parsenet': lambda d: write_parsenet(os.path.join(d, 'parsenet')),
    'lm': lambda d: write_lm(os.path.join(d, 'lm')),
    'region_sorter': lambda d: write_region_sorter(os.path.join(d, 'region_sorter')),
}


def write_all(directory: str, names=None) -> dict:
    """Write the stand-ins ``names`` (default: all of WRITERS) into subdirectories of ``directory``."""
    return {name: WRITERS[name](directory) for name in (names or WRITERS)}

