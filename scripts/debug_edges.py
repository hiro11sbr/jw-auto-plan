"""Print the judged evidence of grid edges along one line, in model mm.

usage: debug_edges.py <plan.png> <config.json> H|V <line_mm> <from_mm> <to_mm>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_detect as gd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
cfg = json.load(open(sys.argv[2], encoding="utf-8"))
rgb = cv2.imdecode(np.fromfile(sys.argv[1], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
if len(sys.argv) > 7:  # optional text-erased image, as grid_detect.py takes it
    clean = cv2.imdecode(np.fromfile(sys.argv[7], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    diff = np.abs(rgb.astype(int) - clean.astype(int)).max(axis=2) > 40
    gd.TEXT_MASK = cv2.dilate(diff.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
fits, gr = gd.calibrate(rgb, cfg)
mid = (gr[0] + gr[1]) / 2
w = cfg["grid_detection"]["gray_halfwidth"]
edges = gd.classify(gd.judge_edges(rgb, fits, (int(mid - w), int(mid + w)), cfg), cfg)
grid = cfg["input"]["grid_mm"]
nx0, ny1 = fits[False][2], fits[True][3]
h = sys.argv[3] == "H"
line_mm, a_mm, b_mm = map(float, sys.argv[4:7])
for e in edges:
    if e["h"] != h:
        continue
    if h:
        lmm = (ny1 - e["line"]) * grid
        s0, s1 = (e["step"] - nx0) * grid, (e["step"] + 1 - nx0) * grid
    else:
        lmm = (e["line"] - nx0) * grid
        s0, s1 = sorted(((ny1 - e["step"]) * grid, (ny1 - e["step"] - 1) * grid))
    if abs(lmm - line_mm) < 1 and s0 >= a_mm - 1 and s1 <= b_mm + 1:
        print(f"{s0:6.0f}-{s1:6.0f} cls={e['cls']:<9} jambs={e.get('jambs')} "
              + " ".join(f"{k}={v:.2f}" for k, v in e["f"].items()))
