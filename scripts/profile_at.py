"""Print the brightness profile across a grid line at given model-mm points, to compare what a
real wall looks like with a look-alike (counter edge, sink side, ...).

usage: profile_at.py <plan.png> <walls.json> <config.json> H|V line_mm at_mm [at_mm ...]
Each row: offset from the grid line (px) and the gray value; outlines are marked ■ (dark).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opening_detect import Mapper  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
img_path, walls_path, cfg_path, hv, line = sys.argv[1:6]
ats = [float(a) for a in sys.argv[6:]]
cfg = json.load(open(cfg_path, encoding="utf-8"))
det = json.load(open(walls_path, encoding="utf-8"))
M = Mapper(det, cfg["input"]["grid_mm"])
bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
v = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(int)
h = hv == "H"
c = M.y(float(line)) if h else M.x(float(line))
ci = int(round(c))
for at in ats:
    t = int(round(M.x(at) if h else M.y(at)))
    # average 5 neighbouring samples along the line (JPEG noise)
    rows = [v[ci - 12:ci + 13, t + k] if h else v[t + k, ci - 12:ci + 13] for k in range(-2, 3)]
    prof = np.mean(rows, axis=0).round().astype(int)
    cols = [bgr[ci - 12:ci + 13, t] if h else bgr[t, ci - 12:ci + 13]][0]
    print(f"--- {hv} {line} at {at}  (grid line px {c:.1f})")
    print("  off: " + " ".join(f"{d:4d}" for d in range(-12, 13)))
    print("  val: " + " ".join(f"{x:4d}" for x in prof))
    print("  out: " + " ".join(("   ■" if x < 130 else "   ·") for x in prof))
    mid = cols[10:15].astype(int)
    print("  centre colour (B,G,R):", [tuple(p) for p in mid])
