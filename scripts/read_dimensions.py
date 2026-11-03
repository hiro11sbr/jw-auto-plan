"""寸法線を読む（ユーザー 2026-10-24）：寸法は必ず建物の最外周まで振ってある。だから
  - 寸法の外側に外壁は無い
  - 寸法があるのに外壁が無いことも無い
寸法線（長い細い黒線）と、そこから壁へ伸びる引出線（目盛り）の位置を拾い、建物の外周と壁の位置の手がかりにする。

usage: read_dimensions.py <plan.png> <config.json> <out_prefix>
writes <out_prefix>_寸法.json: {"h": [{"y": px, "ticks": [px, ...], "side": "top"|"bottom"}, ...],
                               "v": [{"x": px, "ticks": [...], "side": "left"|"right"}, ...],
                               "extent_px": [x0, y0, x1, y1]}
"""
import json
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def runs(b):
    out, s = [], None
    for i, v in enumerate(list(b) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1))
            s = None
    return out


def dim_lines(dark, dc, horizontal):
    """Long thin dark lines with ticks (short perpendicular lines) hanging from them."""
    d = dark if horizontal else dark.T
    H, W = d.shape
    found = []
    for y in range(2, H - 2):
        for a, b in runs(d[y]):
            if b - a < dc["min_len_px"]:
                continue
            # thin: the rows just above and below are mostly empty along the run
            if d[y - 2, a:b].mean() > 0.3 or d[y + 2, a:b].mean() > 0.3:
                continue
            ticks = []
            for side in (1, -1):
                for x in range(a, b + 1):
                    col = [d[y + side * k, x] for k in range(2, dc["tick_min_px"] + 2) if 0 <= y + side * k < H]
                    if col and all(col):
                        ticks.append(x)
            ts = [round((p + q) / 2) for p, q in runs(np.isin(np.arange(a, b + 1), ticks))]
            ts = [a + t for t in ts] if ts and ts[0] < a else ts
            # ticks at wall positions, not a pattern (brick, tile, hatching: evenly close lines)
            gaps = np.diff(ts) if len(ts) > 1 else np.array([999])
            regular = len(ts) >= 6 and float(np.std(gaps)) < 0.15 * float(np.mean(gaps))
            if len(ts) >= dc["min_ticks"] and np.median(gaps) >= dc["min_tick_gap_px"] and not regular:
                found.append({"pos": y, "from": a, "to": b, "ticks": sorted(set(ts))})
    # one line per drawn line (the run is found on adjacent rows too)
    out = []
    for f in sorted(found, key=lambda f: -len(f["ticks"])):
        if not any(abs(f["pos"] - g["pos"]) <= 3 and abs(f["from"] - g["from"]) < 20 for g in out):
            out.append(f)
    return out


def wall_box(bgr, cfg):
    """Rough box of the drawn walls (wall gray, a wall's width thick)."""
    v = bgr.astype(int).mean(axis=2)
    lo, hi = cfg["wall_detection"]["gray_min"], cfg["wall_detection"]["gray_max"]
    m = ((v >= lo) & (v <= hi)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))  # bands at least ~7 px wide
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    # walls are cut into pieces by their outlines: the box of all wall-band pieces of some size
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 300]
    if not keep:
        return None
    x0 = min(st[i, 0] for i in keep)
    y0 = min(st[i, 1] for i in keep)
    x1 = max(st[i, 0] + st[i, 2] for i in keep)
    y1 = max(st[i, 1] + st[i, 3] for i in keep)
    return int(x0), int(y0), int(x1), int(y1)


def main():
    img_path, cfg_path, prefix = sys.argv[1:4]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    dc = cfg["dimensions"]
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    dark = bgr.astype(int).mean(axis=2) < dc["dark_max"]
    box = wall_box(bgr, cfg)
    hs = dim_lines(dark, dc, True)
    vs = dim_lines(dark, dc, False)
    gap = dc["outside_min_px"]
    res = {"h": [], "v": [], "wall_box_px": box}
    if box:
        x0, y0, x1, y1 = box
        # only lines outside the building, the two nearest on each side (detail + overall;
        # further out are site dimensions)
        for side, lst, test, key in (("top", hs, lambda p: p < y0 - gap, lambda f: -f["pos"]),
                                     ("bottom", hs, lambda p: p > y1 + gap, lambda f: f["pos"]),
                                     ("left", vs, lambda p: p < x0 - gap, lambda f: -f["pos"]),
                                     ("right", vs, lambda p: p > x1 + gap, lambda f: f["pos"])):
            sel = sorted([f for f in lst if test(f["pos"])], key=key)[:2]
            for f in sel:
                res["h" if side in ("top", "bottom") else "v"].append(
                    {"side": side, "pos": f["pos"], "ticks": f["ticks"], "span": [f["from"], f["to"]]})
    # the building's extent: on each side the chain with the most ticks (the detail chain; the
    # overall chain beside it may carry a site setback tick too)
    def detail(side):
        lst = [f for f in res["h"] + res["v"] if f["side"] == side]
        return max(lst, key=lambda f: len(f["ticks"]))["ticks"] if lst else []

    def agreed(chains):
        """Ticks found on two or more dimension lines (a site setback tick is on one only)."""
        allt = [t for c in chains for t in set(c)]
        return [t for t in allt if sum(any(abs(t - u) <= 4 for u in c) for c in chains) >= 2]
    ext = []
    for key, sides in (("h", ("top", "bottom")), ("v", ("left", "right"))):
        chains = [f["ticks"] for f in res[key]]
        ts = agreed(chains) or detail(sides[0]) + detail(sides[1])
        ext.append((min(ts), max(ts)) if ts else None)
    if all(ext):
        res["extent_px"] = [ext[0][0], ext[1][0], ext[0][1], ext[1][1]]
        # ticks to calibrate the grid with: inside the extent, and on two or more lines (one
        # stray tick from a site or eave dimension pulled 例1's scale off)
        for key, (lo, hi), name in (("h", ext[0], "anchor_x"), ("v", ext[1], "anchor_y")):
            ts = agreed([f["ticks"] for f in res[key]])
            res[name] = sorted({t for t in ts if lo - 2 <= t <= hi + 2})
    json.dump(res, open(prefix + "_寸法.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"壁の範囲(px)={box} 寸法の外周(px)={res.get('extent_px')}")
    for f in res["h"] + res["v"]:
        print(f"  {f['side']} 位置{f['pos']} 目盛り{f['ticks']}")


if __name__ == "__main__":
    main()
