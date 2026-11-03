"""Japanese text on OpenCV (BGR) images: cv2.putText cannot draw Japanese, so PIL is used."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONTS = [r"C:\Windows\Fonts\BIZ-UDGothicR.ttc", r"C:\Windows\Fonts\meiryo.ttc",
          r"C:\Windows\Fonts\msgothic.ttc",
          "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]  # (クラウドの Linux 環境)
_cache = {}


def _font(size):
    if size not in _cache:
        path = next((p for p in _FONTS if os.path.exists(p)), None)
        _cache[size] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    return _cache[size]


def put(img, items):
    """Draw many texts at once. items: list of (text, (x, y) top-left, (b, g, r), size).
    Returns the new image (same shape)."""
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    for text, (x, y), (b, g, r), size in items:
        d.text((x, y), text, font=_font(size), fill=(r, g, b))
    return np.array(pil)[:, :, ::-1].copy()


def legend(img, title, rows, box=False):
    """Title + colour legend in the top-left corner. rows: list of ((b, g, r), label)."""
    import cv2
    items = [(title, (20, 15), (0, 0, 0), 30)]
    for i, (col, label) in enumerate(rows):
        y = 70 + i * 30
        if box:
            cv2.rectangle(img, (20, y - 8), (60, y + 8), col, 2)
        else:
            cv2.line(img, (20, y), (60, y), col, 4)
        items.append((label, (70, y - 12), (0, 0, 0), 20))
    return put(img, items)
