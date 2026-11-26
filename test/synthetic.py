"""Synthetic inputs that the tiny models in test/tiny_models.py read predictably: bright-on-black line crops and
pages (with line-end marks and region separators for the ParseNet stand-in), and CTC frames over the tiny models'
characters for decoders and language models."""
import numpy as np

from test.tiny_models import CHARACTERS, LINE_END, LINE_PX_HEIGHT, SEPARATOR, ink_colour, order_mark_level


def line_image(n_runs: int, run_width: int = 16, gap: int = 16, height: int = LINE_PX_HEIGHT,
               margin: int = 8) -> np.ndarray:
    """Line crop with ``n_runs`` bright blocks, uint8 HxWx3.

    Runs and gaps are multiples of 4 px so they align with the 4-px subsampling of the stand-in OCR models (with the
    defaults also with a subsampling of 2 or 8 of the tiny CTC model, the crop being padded by 32 px)."""
    width = 2 * margin + n_runs * run_width + max(n_runs - 1, 0) * gap
    img = np.zeros((height, max(width, 2 * margin), 3), dtype=np.uint8)
    x = margin
    for _ in range(n_runs):
        img[height // 4: 3 * height // 4, x:x + run_width] = 255
        x += run_width + gap
    return img


def text_line(text: str, doubts=None, run_width: int = 16, gap: int = 16, height: int = LINE_PX_HEIGHT,
              margin: int = 8) -> np.ndarray:
    """Line crop that the tiny OCR models read as ``text``: one block per character in its ink_colour(), with the
    given doubt per character (default 0: read with full confidence)."""
    doubts = doubts or [0.0] * len(text)
    img = line_image(len(text), run_width=run_width, gap=gap, height=height, margin=margin)
    for i, (character, doubt) in enumerate(zip(text, doubts)):
        x = margin + i * (run_width + gap)
        img[height // 4: 3 * height // 4, x:x + run_width] = ink_colour(character, doubt)
    return img


def draw_text(img: np.ndarray, left: int, top: int, bottom: int, text: str, doubts=None, char_width: int = 24,
              gap: int = 12, strip: int = 2) -> int:
    """Write ``text`` into a page for the tiny models: one block per character from ``top`` to ``bottom``, and a white
    strip ``strip`` px high under the gaps, so that ParseNet sees one continuous line while the OCR models still see
    separate characters (a column is ink for them only when at least a quarter of it is). Returns the right end."""
    doubts = doubts or [0.0] * len(text)
    x = left
    for i, (character, doubt) in enumerate(zip(text, doubts)):
        img[top:bottom, x:x + char_width] = ink_colour(character, doubt)
        if i < len(text) - 1:
            img[bottom - strip:bottom, x + char_width:x + char_width + gap] = 255
        x += char_width + gap
    return x - gap


def page_image(n_lines: int = 3, width: int = 800, height: int = 600, line_height: int = 40,
               line_gap: int = 60, left: int = 100, right: int = 100, top: int = 100) -> np.ndarray:
    """Page with ``n_lines`` full-width bright bars, uint8 HxWx3."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    y = top
    for _ in range(n_lines):
        img[y:y + line_height, left:width - right] = 255
        y += line_height + line_gap
    return img


def layout_page(text=(), line_ends=(), separators=(), size=(600, 800)) -> np.ndarray:
    """Page for the ParseNet stand-in, uint8 HxWx3: boxes (left, top, right, bottom) of white text blocks (one line
    each, its height is the line height), of yellow line-end marks (ink, cut the line they cross) and of red region
    separators (not ink). Later boxes are drawn over earlier ones."""
    img = np.zeros(size + (3,), dtype=np.uint8)
    for boxes, colour in [(text, (255, 255, 255)), (line_ends, LINE_END), (separators, SEPARATOR)]:
        for left, top, right, bottom in boxes:
            img[top:bottom, left:right] = colour
    return img


def mark_reading_order(img: np.ndarray, boxes) -> np.ndarray:
    """Fill the boxes (left, top, right, bottom), given in the wanted reading order, with the reading-order marks of
    the tiny reading-order model, which then reads them in this order whatever their position."""
    for rank, (left, top, right, bottom) in enumerate(boxes):
        img[top:bottom, left:right, 0] = order_mark_level(rank)
    return img


def ctc_frames(*specs):
    """CTC log-probability frames over CHARACTERS + blank (last), e.g. ctc_frames({'a': .98}, {'blank': .98},
    {'c': .6, 'b': .4}); every symbol not named gets a small floor probability."""
    n_classes = len(CHARACTERS) + 1
    out = []
    for spec in specs:
        frame = np.full(n_classes, 1e-4)
        for symbol, p in spec.items():
            frame[n_classes - 1 if symbol == 'blank' else CHARACTERS.index(symbol)] = p
        out.append(np.log(frame / frame.sum()))
    return np.stack(out)


A = {'a': .98}
BLANK = {'blank': .98}
C_OR_B = {'c': .6, 'b': .4}  # visually 'c' wins; the tiny LM, which prefers 'b' after 'a', can override it
AMBIGUOUS_AFTER_A = ctc_frames(A, BLANK, C_OR_B, BLANK)
