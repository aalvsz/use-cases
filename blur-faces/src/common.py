"""Blur logic shared by the build and use paths. No torch dep."""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2


def blur_boxes(
    frame,
    boxes: Iterable[Tuple[float, float, float, float]],
    pad: float = 0.15,
    kernel: int = 31,
) -> None:
    """In-place Gaussian blur of every (x1, y1, x2, y2) box."""
    h, w = frame.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad), int(bh * pad)
        x1 = max(0, int(x1) - px)
        y1 = max(0, int(y1) - py)
        x2 = min(w, int(x2) + px)
        y2 = min(h, int(y2) + py)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        k = max(kernel, (min(x2 - x1, y2 - y1) // 4) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
