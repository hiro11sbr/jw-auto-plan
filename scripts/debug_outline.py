"""Debug: which grid edges are lost by the outline rule, and the profile at their middle."""
import copy
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
fits, gr = gd.calibrate(rgb, cfg)
cfg0 = copy.deepcopy(cfg)
cfg0["grid_detection"]["outline_search_px"] = 0
e0 = gd.classify(gd.judge_edges(rgb, fits, gr, cfg0), cfg0)
e1 = gd.classify(gd.judge_edges(rgb, fits, gr, cfg), cfg)
(sx, ix, *_), (sy, iy, *_) = fits[False], fits[True]
shown = 0
for a, b in zip(e0, e1):
    if a["cls"] == "wall" and b["cls"] != "wall":
        h, ln, st = a["h"], a["line"], a["step"]
        if h:
            c, p = iy + sy * ln, ix + sx * (st + 0.5)
        else:
            c, p = ix + sx * ln, iy + sy * (st + 0.5)
        print(f"lost {'H' if h else 'V'} center={c:.1f} at={p:.1f} f={ {k: round(v, 2) for k, v in b['f'].items()} }")
        if shown < 4:
            ci, pi = int(round(c)), int(round(p))
            prof = rgb[ci - 8:ci + 9, pi] if h else rgb[pi, ci - 8:ci + 9]
            print("   profile:", [int(v.mean()) for v in prof])
            shown += 1
