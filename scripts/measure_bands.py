"""Measure the band (outline-to-outline) width on the plan image along the user's wall centre
lines, to see how constant wall / 腰壁 / fitting widths really are.

usage: measure_bands.py <plan.png> <walls.json (for the px<->mm fit)> <correct.jww> <meta.json> <config.json>

For every 227.5 unit of every answer centre line, the image is sampled across the line; at each
sample the nearest dark outline on each side of the centre is found. Prints the spread of the
width (px) and the centre offset per unit, labelled with what the unit is in the answer.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import ezjww
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opening_detect import Mapper  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def outlines(prof, half, dark_max):
    """Distance to the nearest dark pixel on each side of the centre (None if none within half)."""
    left = next((d for d in range(1, half + 1) if prof[half - d] < dark_max), None)
    right = next((d for d in range(1, half + 1) if prof[half + d] < dark_max), None)
    return left, right


def main():
    img_path, walls_path, ans_path, meta_path, cfg_path = sys.argv[1:6]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    det = json.load(open(walls_path, encoding="utf-8"))
    meta = json.load(open(meta_path, encoding="utf-8"))
    M = Mapper(det, cfg["input"]["grid_mm"])
    S, (cx, cy) = meta["scale"], meta["center_mm"]
    v = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_GRAYSCALE).astype(int)
    H, W = v.shape
    half, dark_max = 16, 130
    lines = []
    for e in ezjww.read_document(ans_path)["entities"]:
        b = e["base"]
        if e["type"] == "LINE" and b["layer"] == 4 and b["pen_style"] == 5:
            p = (e["start_x"] * S + cx, e["start_y"] * S + cy)
            q = (e["end_x"] * S + cx, e["end_y"] * S + cy)
            if abs(p[1] - q[1]) < 1 or abs(p[0] - q[0]) < 1:
                lines.append((p, q))
    widths, filled_w, white_w = Counter(), Counter(), Counter()
    for p, q in lines:
        h = abs(p[1] - q[1]) < 1
        c_mm = p[1] if h else p[0]
        a, b = sorted((p[0], q[0]) if h else (p[1], q[1]))
        c = M.y(c_mm) if h else M.x(c_mm)
        t0, t1 = sorted((M.x(a), M.x(b)) if h else (M.y(a), M.y(b)))
        for t in range(int(t0) + 3, int(t1) - 2):
            ci = int(round(c))
            if h:
                prof = v[ci - half:ci + half + 1, t]
            else:
                prof = v[t, ci - half:ci + half + 1]
            if len(prof) != 2 * half + 1:
                continue
            l, r = outlines(prof, half, dark_max)
            if l is None or r is None:
                continue
            w = l + r
            widths[w] += 1
            mid = prof[half - l + 1:half + r] if w > 2 else prof[half:half + 1]
            if len(mid) and (mid < 210).mean() > 0.6:
                filled_w[w] += 1
            elif len(mid) and (mid > 215).mean() > 0.8:
                white_w[w] += 1
    tot = sum(widths.values())
    print("samples", tot)
    for name, cnt in (("all", widths), ("gray filled (wall)", filled_w), ("white inside (腰壁/opening)", white_w)):
        n = sum(cnt.values())
        print(f"== {name}: {n} samples")
        for w, k in sorted(cnt.items()):
            if k >= 0.01 * n:
                print(f"   width {w:2d}px : {k:5d} ({k / n:5.1%})")


if __name__ == "__main__":
    main()
