"""寸法で外周の壁を確定する（ユーザー 2026-11-02「外壁の判定プロセス」の最後の段）：
  910 → 455 → 227.5 の順に読んだ壁（grid_detect.py）に、寸法の文字を読んで
  グリッドから外れた通り（間崩れ。例：バルコニーの壁芯 -120）と、取りこぼした外周の壁を足す。

  1. 寸法線の目盛り（read_dimensions.py の _寸法.json）の間の数字を OCR で読む（RapidOCR）。
     目盛りの間隔×mm/px と大きく違う読みは捨て、その区間は画素からの推定を使う
  2. 目盛りの実寸を、グリッドに乗る目盛りを起点に数字を足して決める
  3. 各目盛りの通りに沿って、灰色の帯が通りを中心に乗っている所を拾う（開口の途切れは橋渡し）
  4. 建物の外形（今の壁で囲まれた範囲）の外側か外形上にある所だけを外壁として足す。
     端は直交する通りに合わせ、足した壁に届かない既存の壁は伸ばしてつなぐ

usage: dim_walls.py <plan.png> <config.json> <out_prefix>
  reads/updates <out_prefix>_walls.json (walls get "exact": true, not snapped to the grid),
  writes the read values into <out_prefix>_寸法.json ("values")
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from opening_detect import Mapper  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

_ocr = None


def ocr_text(crop):
    """Text of one small crop, or '' (RapidOCR not installed → '')."""
    global _ocr
    if _ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = RapidOCR()
        except ImportError:
            _ocr = False
    if not _ocr:
        return ""
    best = ""
    for f in (3, 2):
        c = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        c = cv2.copyMakeBorder(c, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        r, _ = _ocr(c, use_det=False, use_cls=False)
        t = next((x for x in r[0] if isinstance(x, str)), "") if r else ""
        if len(re.sub(r"\D", "", t)) > len(re.sub(r"\D", "", best)):
            best = t
    return best


def label_groups(band, dark_max):
    """Horizontal extents (s, e) of the text groups in a band (text upright, along x). The
    ticks (dark from top to bottom of the band) are left out."""
    dark = band < dark_max
    tick = dark.mean(axis=0) > 0.8
    cols = dark.any(axis=0) & ~tick
    out, s, gap, e = [], None, 0, 0
    for x, v in enumerate(list(cols) + [False] * 8):
        if v:
            if s is None:
                s = x
            gap, e = 0, x
        elif s is not None:
            gap += 1
            if gap > 5:
                if e - s >= 3:
                    out.append((s, e))
                s = None
    return out


def read_values(img, dims, mpp, cfg):
    """{(key, index): mm} for each dimension line's tick intervals."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_max = cfg["dimensions"].get("text_dark_max", 140)
    values = {}
    for key in ("h", "v"):
        for li, d in enumerate(dims[key]):
            pos, ticks = d["pos"], d["ticks"]
            a, b = ticks[0] - 40, ticks[-1] + 40
            # the text sits on the side of the line with more dark pixels next to it
            sides = []
            for sd in (-1, 1):
                y0, y1 = (pos - 24, pos - 2) if sd < 0 else (pos + 3, pos + 25)
                band = g[y0:y1, a:b] if key == "h" else g[a:b, y0:y1].T
                sides.append(((band < dark_max).sum(), y0, y1))
            _, y0, y1 = max(sides)
            if key == "h":
                band = img[y0:y1, a:b]
            else:  # the text reads upwards: turn it upright (x then runs from b down to a)
                band = cv2.rotate(img[a:b, y0:y1], cv2.ROTATE_90_COUNTERCLOCKWISE)
            gb = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            rows = np.where((gb < dark_max).mean(axis=1) < 0.6)[0]  # (not the dimension line)
            lab = []
            for s, e in label_groups(gb, dark_max):
                sub = gb[:, s:e + 1] < dark_max
                rr = np.where(sub.any(axis=1))[0]
                rr = rr[np.isin(rr, rows)]
                if not len(rr):
                    continue
                r0, r1 = max(0, rr[0] - 4), min(gb.shape[0], rr[-1] + 5)
                t = ocr_text(band[r0:r1, max(0, s - 4):e + 5])
                mid = a + (s + e) / 2  # (turned counter-clockwise, x still runs from a)
                digits = re.sub(r"\D", "", t)
                if digits:
                    lab.append((mid, int(digits), t))
            for i, (p, q) in enumerate(zip(ticks, ticks[1:])):
                est = abs(q - p) * mpp
                cands = [(abs((p + q) / 2 - m), v, t) for m, v, t in lab if min(p, q) - 25 <= m <= max(p, q) + 25]
                cands.sort()
                tol = max(40, 0.15 * est)
                # a misread must agree with the tick spacing. A lost leading digit (a thin '1' on a
                # tick, '.820') or an extra one ('1455]') is put back / dropped when that agrees
                v = None
                for _d, val, _t in cands:
                    tries = [val] + [int(f"{k}{val}") for k in range(1, 10)] + \
                        ([int(str(val)[1:])] if len(str(val)) > 1 else [])
                    ok = [x for x in tries if abs(x - est) <= tol]
                    if ok:
                        v = min(ok, key=lambda x: abs(x - est))
                        break
                values[(key, li, i)] = (v, round(est), cands[0][2] if cands else "")
    return values


def tick_mm(dims, values, M, cfg):
    """{(key, li): [mm per tick]}: x mm for the ticks of an "h" line, y mm for a "v" line."""
    q = cfg["input"]["half_grid_mm"]
    out = {}
    for key in ("h", "v"):
        for li, d in enumerate(dims[key]):
            ticks = d["ticks"]
            est = [M.xmm(t) if key == "h" else M.ymm(t) for t in ticks]
            sgn = 1 if key == "h" else -1  # (y mm runs up the page)
            # start from the tick nearest the grid; add the read values both ways
            k = min(range(len(ticks)), key=lambda i: abs(est[i] - round(est[i] / q) * q))
            mm, read = [None] * len(ticks), [True] * len(ticks)
            mm[k] = round(est[k] / q) * q
            for i in range(k, len(ticks) - 1):
                v = values[(key, li, i)][0]
                step = v if v is not None else round((est[i + 1] - est[i]) * sgn / 5) * 5
                mm[i + 1] = mm[i] + sgn * step
                read[i + 1] = read[i] and v is not None
            for i in range(k, 0, -1):
                v = values[(key, li, i - 1)][0]
                step = v if v is not None else round((est[i] - est[i - 1]) * sgn / 5) * 5
                mm[i - 1] = mm[i] - sgn * step
                read[i - 1] = read[i] and v is not None
            # off the grid only where every number from the start was read (a misread or a
            # guessed step would make a line a few mm beside a real one)
            mm = [m if r else round(m / q) * q for m, r in zip(mm, read)]
            out[(key, li)] = list(zip(ticks, mm))
    return out


def band_runs(gray, M, vertical, c_px, lo_mm, hi_mm, cfg):
    """Runs (mm) along a line where a gray band is centred on it: [(a, b, width_px)]."""
    dc = cfg["dimensions"]
    step = 12.5
    half = int(dc.get("wall_band_max_px", 30))
    hits = []
    t = lo_mm
    while t <= hi_mm:
        p = int(round(M.y(t) if vertical else M.x(t)))
        c = int(round(c_px))
        prof = gray[p, c - half:c + half + 1] if vertical else gray[c - half:c + half + 1, p]
        ok, w = False, 0
        if prof.size == 2 * half + 1 and prof[half]:
            l_ = half
            while l_ > 0 and prof[l_ - 1]:
                l_ -= 1
            r_ = half
            while r_ < 2 * half and prof[r_ + 1]:
                r_ += 1
            w = r_ - l_ + 1
            centre = (l_ + r_) / 2 - half
            # (a band running across the line, not along it, fills the whole window)
            ok = w >= dc.get("wall_band_min_px", 7) and abs(centre) <= dc.get("wall_centre_tol_px", 3) \
                and l_ > 0 and r_ < 2 * half and w <= cfg["wall_detection"]["thickness_px"][1] + 4
        hits.append((t, ok, w))
        t += step
    runs, cur = [], None
    for t, ok, w in hits:
        if ok:
            if cur is None:
                cur = [t, t, []]
            cur[1] = t
            cur[2].append(w)
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    # bridge openings (a window, the railing of a balcony) between two pieces of the same band
    out = []
    for r in runs:
        if out and r[0] - out[-1][1] <= dc.get("bridge_mm", 1820) and \
                abs(np.median(r[2]) - np.median(out[-1][2])) <= 4:
            out[-1][1] = r[1]
            out[-1][2] += r[2]
        else:
            out.append(r)
    return [(a, b, float(np.median(ws))) for a, b, ws in out]


def main():
    img_path, cfg_path, prefix = sys.argv[1:4]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    dc = cfg["dimensions"]
    dims_path, walls_path = Path(prefix + "_寸法.json"), Path(prefix + "_walls.json")
    if not dims_path.exists() or not dc.get("dim_walls", True):
        return
    dims = json.load(open(dims_path, encoding="utf-8"))
    data = json.load(open(walls_path, encoding="utf-8"))
    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    grid = cfg["input"]["grid_mm"]
    M = Mapper(data, grid)
    mpp = data["mm_per_px"]
    values = read_values(img, dims, mpp, cfg)
    dims["values"] = {f"{k}{li}_{i}": {"mm": v, "estimate": e, "text": t} for (k, li, i), (v, e, t) in values.items()}
    json.dump(dims, open(dims_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tmm = tick_mm(dims, values, M, cfg)
    lo, hi = data["gray_range"]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = (g >= lo) & (g <= hi)

    # (a run again replaces what an earlier run added)
    walls = [w for w in data["walls"] if not w.get("source", "").startswith("寸法")]

    def wl(w):
        return LineString([tuple(w["p1"]), tuple(w["p2"])])

    def snapped(v):
        return round(v / grid) * grid

    cur = [((tuple(snapped(c) for c in w["p1"]) if not w.get("exact") else tuple(w["p1"])),
            (tuple(snapped(c) for c in w["p2"]) if not w.get("exact") else tuple(w["p2"]))) for w in walls]
    polys = list(polygonize(unary_union([LineString(s) for s in cur])))
    fp = unary_union(polys) if polys else None
    inner = fp.buffer(-1) if fp is not None else None
    # the extent of the building read from the dimensions (mm)
    xs = [m for (k, _li), tm in tmm.items() if k == "h" for _t, m in tm]
    ys = [m for (k, _li), tm in tmm.items() if k == "v" for _t, m in tm]
    if not xs or not ys:
        return
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    # the lines of the dimensions: x lines from "h" dimension lines, y lines from "v" ones
    lines, first = {}, set()
    q = cfg["input"]["half_grid_mm"]
    for (k, _li), tm in tmm.items():
        for i, (t, m) in enumerate(tm):
            lines.setdefault((k == "h", round(m, 1)), []).append(t)
            # the outermost lines of the dimensions, and lines off the grid (間崩れ)
            if i in (0, len(tm) - 1) or abs(m - round(m / q) * q) > 1:
                first.add((k == "h", round(m, 1)))
    perp = {True: sorted({m for (v, m) in lines if not v} | {snapped(p[1]) for s in cur for p in s}),
            False: sorted({m for (v, m) in lines if v} | {snapped(p[0]) for s in cur for p in s})}
    runs = {}
    for (vertical, m), tps in sorted(lines.items()):
        c_px = float(np.median(tps))  # the tick's own pixel: more exact than the grid fit
        lo_mm, hi_mm = (y0 - 300, y1 + 300) if vertical else (x0 - 300, x1 + 300)
        runs[(vertical, m)] = band_runs(gray, M, vertical, c_px, lo_mm, hi_mm, cfg)
    reach = dc.get("end_reach_mm", 250)

    def supported(vertical, p, t):
        """A wall (drawn, or a band found on a dimension line) on line p reaches t."""
        if any(a - reach <= t <= b + reach for a, b, _w in runs.get((vertical, p), [])):
            return True
        for s_ in cur:
            (ax, ay), (bx, by) = s_
            if vertical and abs(ax - bx) < 1 and abs(ax - p) < 1 and min(ay, by) - reach <= t <= max(ay, by) + reach:
                return True
            if not vertical and abs(ay - by) < 1 and abs(ay - p) < 1 and min(ax, bx) - reach <= t <= max(ax, bx) + reach:
                return True
        return False

    def end_at(vertical, m, t, d):
        """The crossing line a band's end t belongs to: a band stops at the crossing wall's inner
        face, up to half its thickness short of its centre (d: outward direction)."""
        cand = [p for p in perp[vertical] if -50 <= (p - t) * d <= reach]
        sup = [p for p in cand if supported(not vertical, p, m)]
        pick = sup or [p for p in cand if abs(p - t) <= 150]
        return min(pick, key=lambda p: abs(p - t)) if pick else None

    added = []

    def touches(vertical, m, t):
        """An end at t of a wall on line m meets a wall added from an outer / off-grid line."""
        pt = Point((m, t) if vertical else (t, m))
        return any(LineString([tuple(w_["p1"]), tuple(w_["p2"])]).distance(pt) < 1 for w_ in added)

    # the outer and off-grid lines first; the other lines only where they join those (a balcony's
    # side wall on a grid line)
    order = [k for k in sorted(runs) if k in first] + [k for k in sorted(runs) if k not in first]
    for (vertical, m) in order:
        rr = runs[(vertical, m)]
        for a, b, w in rr:
            ea, eb = end_at(vertical, m, a, -1), end_at(vertical, m, b, 1)
            if ea is None or eb is None or eb - ea < dc.get("min_wall_mm", 300):
                continue
            seg = LineString([(m, ea), (m, eb)] if vertical else [(ea, m), (eb, m)])
            # only outside or on the building's outline (the outer walls: balcony, porch)
            have = unary_union([LineString(s_) for s_ in cur]).buffer(1)
            rest = seg.difference(have)
            ends = sorted({ea, eb} | {p for s_ in cur for q in s_ for p in q})
            for piece in getattr(rest, "geoms", [rest]):
                if piece.is_empty or piece.length < dc.get("min_wall_mm", 300):
                    continue
                if inner is not None and inner.contains(piece.interpolate(0.5, normalized=True)):
                    continue
                t1, t2 = [(q[1] if vertical else q[0]) for q in (piece.coords[0], piece.coords[-1])]
                t1, t2 = [min(ends, key=lambda e: abs(e - t)) if min(abs(e - t) for e in ends) <= 2 else t
                          for t in (t1, t2)]
                if (vertical, m) not in first and not (touches(vertical, m, t1) or touches(vertical, m, t2)):
                    continue
                q1, q2 = ([m, t1], [m, t2]) if vertical else ([t1, m], [t2, m])
                added.append({"type": "exterior" if w >= cfg["wall_detection"]["exterior_min_thickness_px"] else "interior",
                              "p1": q1, "p2": q2, "exact": True, "source": "寸法", "band_px": w, "flags": []})
                cur.append((tuple(q1), tuple(q2)))
    # an existing wall that stops short of an added wall's line, its band running on: extend it
    for nw in list(added):
        v_new = abs(nw["p1"][0] - nw["p2"][0]) < 1
        c_new = nw["p1"][0] if v_new else nw["p1"][1]
        span = sorted((nw["p1"][1], nw["p2"][1]) if v_new else (nw["p1"][0], nw["p2"][0]))
        for w in walls:
            p1, p2 = [list(map(float, w["p1"])), list(map(float, w["p2"]))]
            h = abs(p1[1] - p2[1]) < 1
            if h != v_new:
                continue
            c = p1[1] if h else p1[0]
            if not span[0] - 1 <= c <= span[1] + 1:
                continue
            for end in (p1, p2):
                t = end[0] if h else end[1]
                if 1 < abs(t - c_new) <= grid / 2:
                    # (the gap is the corner where the two bands meet: gray all along the line)
                    c_px = int(round(M.y(c) if h else M.x(c)))
                    ts = np.arange(min(t, c_new), max(t, c_new) + 1, 12.5)
                    on = [gray[c_px, int(round(M.x(u)))] if h else gray[int(round(M.y(u))), c_px] for u in ts]
                    if np.mean(on) >= 0.8:
                        added.append({"type": w["type"], "p1": [c, t] if not h else [t, c],
                                      "p2": [c, c_new] if not h else [c_new, c], "exact": True,
                                      "source": "寸法（延長）", "flags": []})
    for nw in added:
        print(f"寸法から足した壁: {nw['source']} {nw['type']} {nw['p1']}〜{nw['p2']}")
    bad = [(k, v) for k, v in values.items() if v[0] is None]
    print(f"寸法の数字: 読めた {len(values) - len(bad)} / {len(values)}  足した壁 {len(added)}")
    data["walls"] = walls + added
    json.dump(data, open(walls_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
