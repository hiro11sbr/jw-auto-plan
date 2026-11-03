"""Locate plan (横断面) pieces on a maker 姿図・断面図 sheet: list W= texts in the bottom row and
render a crop around each so the piece's bounding box can be confirmed.

usage: find_plan_pieces.py <sheet.jww> <out_dir>
"""
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import ezjww
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
path, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
doc = ezjww.read_document(str(path))
ents = doc["entities"]


def text_of(e):
    return unicodedata.normalize("NFKC", e.get("text") or e.get("content") or "")


ws = [e for e in ents if e["type"] == "TEXT" and re.match(r"W=\d+$", text_of(e).replace(" ", ""))]
print("W texts:", [(text_of(e), round(e["start_x"], 1), round(e["start_y"], 1)) for e in ws])

lines = [e for e in ents if e["type"] in ("LINE",)]
for i, w in enumerate(ws):
    wx, wy = w["start_x"], w["start_y"]
    x0, x1, y0, y1 = wx - 30, wx + 40, wy - 5, wy + 30
    sel = [e for e in lines if x0 <= min(e["start_x"], e["end_x"]) and max(e["start_x"], e["end_x"]) <= x1
           and y0 <= min(e["start_y"], e["end_y"]) and max(e["start_y"], e["end_y"]) <= y1]
    S = 30
    img = np.full((int((y1 - y0) * S), int((x1 - x0) * S), 3), 255, np.uint8)
    for e in sel:
        p = (int((e["start_x"] - x0) * S), int((y1 - e["start_y"]) * S))
        q = (int((e["end_x"] - x0) * S), int((y1 - e["end_y"]) * S))
        cv2.line(img, p, q, (0, 0, 0), 1, cv2.LINE_AA)
    # grid every 1 paper mm for reading coordinates
    for gx in range(int(x0), int(x1) + 1, 5):
        cv2.line(img, (int((gx - x0) * S), 0), (int((gx - x0) * S), img.shape[0]), (200, 220, 255), 1)
        cv2.putText(img, str(gx), (int((gx - x0) * S) + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    for gy in range(int(y0), int(y1) + 1, 5):
        cv2.line(img, (0, int((y1 - gy) * S)), (img.shape[1], int((y1 - gy) * S)), (200, 220, 255), 1)
        cv2.putText(img, str(gy), (2, int((y1 - gy) * S) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    fn = out / f"{path.stem}_W{i}.png"
    cv2.imencode(".png", img)[1].tofile(str(fn))
    print(fn.name, "lines", len(sel))
