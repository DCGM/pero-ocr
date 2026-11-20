"""Synthetic bright-on-black images that the tiny models in test/tiny_models.py read predictably."""
import numpy as np

from test.tiny_models import LINE_PX_HEIGHT


def line_image(n_runs: int, run_width: int = 16, gap: int = 16, height: int = LINE_PX_HEIGHT,
               margin: int = 8) -> np.ndarray:
    """Line crop with ``n_runs`` bright blocks, uint8 HxWx3.

    Runs and gaps are multiples of 4 px so they align with the 4-px subsampling of the stand-in OCR models."""
    width = 2 * margin + n_runs * run_width + max(n_runs - 1, 0) * gap
    img = np.zeros((height, max(width, 2 * margin), 3), dtype=np.uint8)
    x = margin
    for _ in range(n_runs):
        img[height // 4: 3 * height // 4, x:x + run_width] = 255
        x += run_width + gap
    return img


def page_image(n_lines: int = 3, width: int = 800, height: int = 600, line_height: int = 40,
               line_gap: int = 60, left: int = 100, right: int = 100, top: int = 100) -> np.ndarray:
    """Page with ``n_lines`` full-width bright bars, uint8 HxWx3."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    y = top
    for _ in range(n_lines):
        img[y:y + line_height, left:width - right] = 255
        y += line_height + line_gap
    return img
