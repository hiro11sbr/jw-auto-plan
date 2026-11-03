"""腰壁 (low walls: beside stairs, in front of the kitchen, ...) on the plan image.

usage: koshikabe_detect.py <plan.png> <walls.json> <config.json> <out_prefix>
writes <out_prefix>_腰壁.json  [{horizontal, line_mm, a_mm, b_mm}]

On the input plans a 腰壁 is drawn like a wall band but NOT filled: two outlines a wall's
thickness apart with white between (a real wall is filled grey). Its centre sits on a grid line
(455 / 227.5); its free end need not (1F kitchen: 2580 from the wall centre, user 2026-10-22).
Runs on the band of a detected wall are skipped. 框 (between 玄関 and ホール) is not looked for:
it is not on a wall line with this band shape.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from opening_detect import Mapper  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def runs(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def main():
    img_path, walls_path, cfg_path, prefix = sys.argv[1:5]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    kc = cfg["koshikabe"]
    det = json.load(open(walls_path, encoding="utf-8"))
    M = Mapper(det, cfg["input"]["grid_mm"])
    gray = cv2.cvtColor(cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    dark = gray < kc["dark_max"]
    light = gray > kc["light_min"]
    half = cfg["input"]["half_grid_mm"]
    xs = [w["p1"][0] for w in det["walls"]] + [w["p2"][0] for w in det["walls"]]
    ys = [w["p1"][1] for w in det["walls"]] + [w["p2"][1] for w in det["walls"]]
    walls = [(abs(w["p1"][1] - w["p2"][1]) < 1, w) for w in det["walls"]]
    out = []
    for horiz in (True, False):
        lo, hi = (min(ys), max(ys)) if horiz else (min(xs), max(xs))
        n0, n1 = int(np.floor(lo / half)), int(np.ceil(hi / half))
        for n in range(n0, n1 + 1):
            line = n * half
            c = M.y(line) if horiz else M.x(line)
            ci = int(round(c))
            if not (10 <= ci < (H if horiz else W) - 10):
                continue
            L = W if horiz else H
            ok = np.zeros(L, bool)
            for t in range(L):
                prof_d = dark[ci - 8:ci + 9, t] if horiz else dark[t, ci - 8:ci + 9]
                prof_l = light[ci - 8:ci + 9, t] if horiz else light[t, ci - 8:ci + 9]
                for k in range(kc["half_min_px"], kc["half_max_px"] + 1):
                    if prof_d[8 - k - 1:8 - k + 2].any() and prof_d[8 + k - 1:8 + k + 2].any() \
                            and prof_l[8 - k + 2:8 + k - 1].all():
                        ok[t] = True
                        break
            # close small gaps (text, JPEG)
            for s, e in runs(~ok):
                if e - s <= kc["gap_px"]:
                    ok[s:e] = True
            for s, e in runs(ok):
                if e - s < kc["min_len_px"]:
                    continue
                a = M.xmm(s) if horiz else M.ymm(e - 1)
                b = M.xmm(e - 1) if horiz else M.ymm(s)
                a, b = sorted((a, b))
                # not on the band of a detected wall
                on_wall = False
                for wh, w in walls:
                    if wh != horiz:
                        continue
                    wc = w["p1"][1] if horiz else w["p1"][0]
                    wa, wb = sorted((w["p1"][0], w["p2"][0]) if horiz else (w["p1"][1], w["p2"][1]))
                    if abs(wc - line) < 1 and min(b, wb) - max(a, wa) > 0.5 * (b - a):
                        on_wall = True
                if not on_wall:
                    out.append({"horizontal": horiz, "line_mm": line, "a_mm": round(a, 1), "b_mm": round(b, 1),
                                "len_px": int(e - s)})
    json.dump(out, open(prefix + "_腰壁.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in out:
        print("腰壁", "横" if k["horizontal"] else "縦", k["line_mm"], k["a_mm"], k["b_mm"], k["len_px"])


if __name__ == "__main__":
    main()
