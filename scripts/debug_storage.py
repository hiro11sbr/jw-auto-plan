"""Debug the storage-front scan on one line: print per-position evidence summary.

usage: debug_storage.py <plan.png> <walls.json> <config.json> H|V <line_mm> <from_mm> <to_mm>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opening_detect as od  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
img = cv2.imdecode(np.fromfile(sys.argv[1], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
cfg, det = od.load(sys.argv[3], sys.argv[2])
oc = cfg["opening_detection"]
M = od.Mapper(det, cfg["input"]["grid_mm"])
h = sys.argv[4] == "H"
c_mm, a_mm, b_mm = map(float, sys.argv[5:8])
lo, hi = det["gray_range"]
mid, w = (lo + hi) / 2, cfg["grid_detection"]["gray_halfwidth"]
gr = (mid - w, mid + w)
c = M.y(c_mm) if h else M.x(c_mm)
pa, pb = sorted((M.x(a_mm), M.x(b_mm)) if h else (M.y(a_mm), M.y(b_mm)))
pos = np.arange(int(pa), int(pb))
half = oc["band_half_px"]
gray, dark, red = od.profiles(img, gr, c, pos, h, half)
wl = od.wall_at(gray, red, half)
sb, win = oc["storage_band_px"], oc["storage_window_px"]
band = dark[:, half - sb:half + sb + 1]
k = np.ones(2 * win + 1) / (2 * win + 1)
cov = np.array([np.convolve(band[:, j], k, mode="same") for j in range(band.shape[1])]).T
strong = cov >= oc["storage_line_cov"]
nl = [np.count_nonzero(np.diff(np.concatenate(([0], r.astype(int)))) == 1) for r in strong]
print(f"line px={c:.1f} span px {pos[0]}..{pos[-1]}")
for i in range(0, len(pos), 4):
    darkcols = "".join("#" if x else "." for x in dark[i])
    print(f"{pos[i]:5d} wall={int(wl[i])} nlines={nl[i]} {darkcols}")
