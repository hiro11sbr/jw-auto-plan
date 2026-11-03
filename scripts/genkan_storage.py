"""玄関収納（下駄箱）を見つける（作図はしない。框・建具の誤認を防ぐため; ユーザー 2026-10-24）。

見た目：玄関かその近くのホールに、正方形に近い四角が3〜4個並び、それぞれの四角に間隔の近い
平行線のハッチング、または対角線が描かれる。框と重なって描かれることが多い。

usage: genkan_storage.py <plan.png> <walls.json> <部屋名.json> <config.json> <out_prefix>
writes <out_prefix>_玄関収納.json: [{"px": [x0, y0, x1, y1], "mm": [...], "cells": n, "kinds": [...]}]
and <out_prefix>_玄関収納.png (check image)
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jp_text  # noqa: E402
from opening_detect import Mapper  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def cells_in(dark, gc):
    """Candidate cells: squares (side cell_px) whose inside is hatched (many close parallel
    lines in one direction) or crossed by one diagonal from corner to corner."""
    H, W = dark.shape
    lo, hi = gc["cell_px"]
    out = []
    step = gc["scan_step_px"]
    for s in range(lo, hi + 1, 3):
        for y in range(0, H - s, step):
            for x in range(0, W - s, step):
                win = dark[y + 2:y + s - 2, x + 2:x + s - 2]
                if win.size == 0:
                    continue
                ink = win.mean()
                if ink < gc["min_ink"] or ink > gc["max_ink"]:
                    continue
                # hatching: in one direction, nearly every row (column) of the window crosses
                # many lines, spaced closely
                runs_r = np.count_nonzero(np.diff(win.astype(int), axis=1) == 1, axis=1)
                runs_c = np.count_nonzero(np.diff(win.astype(int), axis=0) == 1, axis=0)
                hatch = (np.median(runs_r) >= gc["hatch_min_lines"] and np.median(runs_c) <= 2) or \
                        (np.median(runs_c) >= gc["hatch_min_lines"] and np.median(runs_r) <= 2)
                kind = "ハッチング" if hatch else None
                if not kind:
                    # diagonal corner to corner (either way), and little else inside
                    n = s - 4
                    d1 = np.mean([win[i, i] or win[i, max(0, i - 1)] or win[i, min(n - 1, i + 1)] for i in range(n)])
                    d2 = np.mean([win[i, n - 1 - i] or win[i, max(0, n - 2 - i)] or win[i, min(n - 1, n - i)]
                                  for i in range(n)])
                    if max(d1, d2) >= gc["diag_min"] and ink <= gc["diag_max_ink"]:
                        kind = "対角線"
                if kind:
                    out.append((x, y, s, kind))
    return out


def group(cells, gc, shape):
    """Vote: every candidate square adds one over its area; a cluster's core (votes at least
    vote_frac of its peak) is the storage. A plain union of the squares grew past the storage
    (windows slide over its edges), so its size test failed (プラン5)."""
    H, W = shape
    votes = np.zeros((H, W), np.int32)
    kinds_at = {}
    for x, y, s, k in cells:
        votes[y:y + s, x:x + s] += 1
        kinds_at.setdefault(k, np.zeros((H, W), bool))[y:y + s, x:x + s] = True
    n, lab = cv2.connectedComponents((votes > 0).astype(np.uint8))
    boxes = []
    for i in range(1, n):
        m = lab == i
        core = m & (votes >= gc["vote_frac"] * votes[m].max())
        ys, xs = np.nonzero(core)
        kinds = {k for k, a in kinds_at.items() if (a & core).any()}
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, kinds))
    out = []
    for x0, y0, x1, y1, kinds in boxes:
        w, h = x1 - x0, y1 - y0
        short, long_ = min(w, h), max(w, h)
        n = long_ / max(1, short)
        if gc["row_min_cells"] - 0.6 <= n <= gc["row_max_cells"] + 0.8 and gc["cell_px"][0] <= short <= gc["cell_px"][1] + 8:
            out.append({"px": [int(x0), int(y0), int(x1), int(y1)], "cells": round(n, 1), "kinds": sorted(kinds)})
    return out


def main():
    img_path, walls_path, names_path, cfg_path, prefix = sys.argv[1:6]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    gc = cfg["genkan_storage"]
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    v = bgr.astype(int).mean(axis=2)
    dark = v < gc["dark_max"]
    det = json.load(open(walls_path, encoding="utf-8"))
    M = Mapper(det, cfg["input"]["grid_mm"])
    mmpp = det["mm_per_px"]
    names = json.load(open(names_path, encoding="utf-8")) if Path(names_path).exists() else []
    genkan = [(n["x"], n["y"]) for n in names if n.get("name") in ("玄関", "ホール")]
    # only near the 玄関 / ホール labels (that is where it stands)
    R = gc["near_label_mm"] / mmpp
    H, W = dark.shape
    found = []
    for gx, gy in genkan:
        x0, y0 = int(max(0, gx - R)), int(max(0, gy - R))
        x1, y1 = int(min(W, gx + R)), int(min(H, gy + R))
        for c in group([(x + x0, y + y0, s, k) for x, y, s, k in cells_in(dark[y0:y1, x0:x1], gc)], gc, dark.shape):
            if not any(abs(c["px"][0] - f["px"][0]) < 6 and abs(c["px"][1] - f["px"][1]) < 6 for f in found):
                found.append(c)
    for f in found:
        a, b, c_, d = f["px"]
        f["mm"] = [round(M.xmm(a)), round(M.ymm(d)), round(M.xmm(c_)), round(M.ymm(b))]
    json.dump(found, open(prefix + "_玄関収納.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    chk = bgr.copy()
    for f in found:
        a, b, c_, d = f["px"]
        cv2.rectangle(chk, (a, b), (c_, d), (0, 140, 255), 2)
    name = prefix.replace("\\", "/").split("/")[-1]
    chk = jp_text.put(chk, [(f"{name}　玄関収納（下駄箱）の確認", (10, 10), (0, 0, 0), 26),
                            ("橙の四角：玄関収納と判定した所（作図はしない）", (10, 44), (0, 140, 255), 20)])
    cv2.imencode(".png", chk)[1].tofile(prefix + "_玄関収納.png")
    print(f"{name}: 玄関収納 {len(found)}か所", [(f["px"], f["kinds"], f["cells"]) for f in found])


if __name__ == "__main__":
    main()
