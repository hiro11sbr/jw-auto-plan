"""Trace the storage-front decision on one line (why a run is kept or dropped)."""
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
c_mm = float(sys.argv[5])
lo, hi = det["gray_range"]
mid, w = (lo + hi) / 2, cfg["grid_detection"]["gray_halfwidth"]
gr = (mid - w, mid + w)
half = oc["band_half_px"]
mmpp = det["mm_per_px"]
xs = [v for wl in det["walls"] for v in (wl["p1"][0], wl["p2"][0])]
ys = [v for wl in det["walls"] for v in (wl["p1"][1], wl["p2"][1])]
a_lo, a_hi = (max(0, min(xs)), max(xs)) if h else (max(0, min(ys)), max(ys))
c = M.y(c_mm) if h else M.x(c_mm)
pa, pb = sorted((M.x(a_lo), M.x(a_hi)) if h else (M.y(a_lo), M.y(a_hi)))
pos = np.arange(int(pa) + 2, int(pb) - 2)
gray, dark, red = od.profiles(img, gr, c, pos, h, half)
wall_here = od.wall_at(gray, red, half)
win, sb = oc["storage_window_px"], oc["storage_band_px"]
band = dark[:, half - sb:half + sb + 1]
k = np.ones(2 * win + 1) / (2 * win + 1)
cov = np.array([np.convolve(band[:, j], k, mode="same") for j in range(band.shape[1])]).T
strong = cov >= oc["storage_line_cov"]
nl = np.array([np.count_nonzero(np.diff(np.concatenate(([0], r.astype(int)))) == 1) for r in strong])
cand = (nl >= 2) & ~wall_here
for s, e in od.runs(~cand):
    if e - s <= 5 and s > 0 and e < len(cand):
        cand[s:e] = True
for s, e in od.runs(cand):
    p0, p1 = int(pos[s]), int(pos[e - 1])
    wmm = (e - s) * mmpp
    ends = (od.wall_near(img, gr, c, p0, -1, h), od.wall_near(img, gr, c, p1, 1, h))
    lines = od.distinct_lines(dark[s:e], half, oc["storage_leaf_cov"], sb)
    print(f"run px {p0}-{p1} width {wmm:.0f}mm ends={ends} lines={lines}")
