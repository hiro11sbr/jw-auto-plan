"""Interior fittings the first pass gets wrong (user, 2026-10-23):
  - 折戸: each folding pair is drawn as a V (two straight leaves leaning opposite ways, no arc).
    Under 910 there is one V at one jamb; wider ones (a closet) have one at each jamb. A door
    read as 引違い戸 with diagonals both ways, or a "片開き" with many diagonals, is a 折戸. The V's
    jamb is its hinge end and its side the side it opens to.
  - 框 (上がり框): the edge of the step between the 玄関 土間 and the raised floor. Not a wall and
    not a fitting: an open edge you step up over. Drawn as a double line along the edge (can be
    diagonal or bent; only the straight ones on the grid are caught here). Taken as 框: a
    "fitting" seen from the 玄関 label with no wall between, drawn as two full-length lines (no
    staggered sashes, no swing arc). It gets no part, and its line no wall.

usage: interior_type.py <plan.png> <openings.json> <walls.json> <部屋名.json> <config.json> [--report]
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opening_detect import Mapper  # noqa: E402
from window_type import cues  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def diagonals(dark, o, ic):
    """Straight segments at 15-84 deg to the wall beside the opening (a swing door's leaf stands
    at 90 and its arc gives few straight pieces). Returns (leaning one way, the other way,
    [(position along 0..1, side across -1/+1), ...])."""
    p0, p1, c = [int(round(v)) for v in o["px"]]
    L = p1 - p0
    h = o["horizontal"]
    reach = max(20, L // 2)
    x0, x1, y0, y1 = (p0, p1, c - reach, c + reach) if h else (c - reach, c + reach, p0, p1)
    H, W = dark.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    m = (dark[y0:y1, x0:x1] * 255).astype(np.uint8)
    segs = cv2.HoughLinesP(m, 1, np.pi / 180, ic["hough_votes"], minLineLength=max(8, L * 0.12), maxLineGap=2)
    pos = neg = 0
    where = []
    if segs is None:
        return 0, 0, where
    lo, hi = ic["diag_deg"]
    for xa, ya, xb, yb in segs.reshape(-1, 4):
        xa, ya, xb, yb = int(xa), int(ya), int(xb), int(yb)
        dx, dy = (xb - xa, yb - ya) if h else (yb - ya, xb - xa)
        a = (math.degrees(math.atan2(dy, dx)) + 180) % 180
        if lo <= a <= hi:
            pos += 1
        elif 180 - hi <= a <= 180 - lo:
            neg += 1
        else:
            continue
        mx, my = x0 + (xa + xb) / 2, y0 + (ya + yb) / 2
        t, d = (mx, my - c) if h else (my, mx - c)
        # which side it opens to: only the legs that start at the frame count (hangers and
        # shelves inside a closet are diagonals too, プラン2 2F)
        ends = [(y0 + ya, y0 + yb) if h else (x0 + xa, x0 + xb)]
        near = min(abs(v - c) for v in ends[0]) <= ic["leg_from_frame_px"]
        where.append(((t - p0) / max(1, L), (1 if d > 0 else -1) if near else 0))
    return pos, neg, where


def quarter_arc(dark, o, hinge_t, toward, side, r):
    """Share of a quarter circle (radius r, hinge on the wall at hinge_t) that is drawn."""
    c0, h = o["px"][2], o["horizontal"]
    H, W = dark.shape
    best = 0.0
    # the pivot may sit on the wall face rather than the centre line (プラン2 2F 主寝室)
    for k, sh in [(k, sh) for k in (0.85, 0.9, 0.95, 1.0, 1.05) for sh in (0, 3, 6)]:
        c = c0 + side * sh
        hits = n = 0
        for deg in range(10, 81, 5):
            a = math.radians(deg)
            t = hinge_t + toward * r * k * math.cos(a)
            d = c + side * r * k * math.sin(a)
            y, x = (int(round(d)), int(round(t))) if h else (int(round(t)), int(round(d)))
            if 1 <= y < H - 1 and 1 <= x < W - 1:
                n += 1
                hits += bool(dark[y - 1:y + 2, x - 1:x + 2].any())
        best = max(best, hits / max(n, 1))
    return best


def full_arc(dark, o):
    """(score, hinge px, side) of a single swing: a quarter circle of the full width at one jamb."""
    p0, p1, _c = o["px"]
    L = p1 - p0
    best = (0.0, None, None)
    for hp, toward in ((p0, 1), (p1, -1)):
        for side in (1, -1):
            sc = max(quarter_arc(dark, o, hp + toward * dh, toward, side, L) for dh in (0, 2, 4))
            if sc > best[0]:
                best = (sc, hp, side)
    return best


def hatched(dark, o, side, ic):
    """True when the area beside the opening on `side` (half its width deep) is filled with close
    parallel lines: the hatching of a 玄関収納 or cupboard, which a double-door test reads as two
    leaves and arcs (プラン5 1F, user 10/24)."""
    p0, p1, c = [int(round(v)) for v in o["px"]]
    L = p1 - p0
    h = o["horizontal"]
    d0, d1 = sorted((c + side * 6, c + side * max(12, L // 2)))
    win = dark[d0:d1, p0 + 3:p1 - 2] if h else dark[p0 + 3:p1 - 2, d0:d1]
    if win.size == 0 or min(win.shape) < 6:
        return False
    w = win.astype(int)
    # lines per px across each direction: hatching is dense (プラン5 0.2, a line every ~5 px);
    # a double door's arcs and leaves, or floor lines, are sparse (0.08-0.1 on 例1・2・4)
    dens_r = np.median(np.count_nonzero(np.diff(w, axis=1) == 1, axis=1)) / w.shape[1]
    dens_c = np.median(np.count_nonzero(np.diff(w, axis=0) == 1, axis=0)) / w.shape[0]
    return max(dens_r, dens_c) >= ic["hatch_min_density"]


def half_arcs(dark, o):
    """(score, side) of 両開き arcs: a quarter circle of half the width at BOTH jambs on one side.
    Its two arcs are drawn all round (0.87-1.0 at each jamb on both plans); a 折戸's V crosses
    that circle at a point or two (0.53 at most)."""
    p0, p1, c = o["px"]
    L = p1 - p0
    h = o["horizontal"]
    H, W = dark.shape

    def quarter(hinge_t, toward, side, r):
        best = 0.0
        for k in (0.85, 0.9, 0.95, 1.0, 1.05):
            hits = n = 0
            for deg in range(10, 81, 5):
                a = math.radians(deg)
                t = hinge_t + toward * r * k * math.cos(a)
                d = c + side * r * k * math.sin(a)
                y, x = (int(round(d)), int(round(t))) if h else (int(round(t)), int(round(d)))
                if 1 <= y < H - 1 and 1 <= x < W - 1:
                    n += 1
                    hits += bool(dark[y - 1:y + 2, x - 1:x + 2].any())
            best = max(best, hits / max(n, 1))
        return best

    best = (0.0, None)
    for side in (1, -1):
        a = max(quarter(p0 + dh, 1, side, L / 2) for dh in (0, 2, 4))
        b = max(quarter(p1 - dh, -1, side, L / 2) for dh in (0, 2, 4))
        if min(a, b) > best[0]:
            best = (min(a, b), side)
    return best


def cut_wall(det, o, half, jamb):
    """Remove the wall under a 框 (snapped to the jamb grid lines)."""
    h = o["horizontal"]
    snap = lambda v: round(v / half) * half  # noqa: E731
    line = snap(o["p1_mm"][1] if h else o["p1_mm"][0])
    a, b = sorted((o["p1_mm"][0], o["p2_mm"][0]) if h else (o["p1_mm"][1], o["p2_mm"][1]))
    a, b = snap(a - jamb), snap(b + jamb)
    out = []
    for w in det["walls"]:
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        on = (h and abs(y1 - line) < 1 and abs(y2 - line) < 1) or (not h and abs(x1 - line) < 1 and abs(x2 - line) < 1)
        if not on:
            out.append(w)
            continue
        k = 0 if h else 1
        s, e = sorted((w["p1"][k], w["p2"][k]))
        for u, v in ((s, min(e, a)), (max(s, b), e)):
            if v - u > 1:
                p, q = [u, line] if h else [line, u], [v, line] if h else [line, v]
                out.append({**w, "p1": p, "p2": q})
    det["walls"] = out


def pocket_side(dark, o, ic):
    """'before' / 'after' when a line inside the wall band continues beyond that jamb into the
    wall (a 片引戸's pocket, often dashed), else None."""
    p0, p1, c = [int(round(v)) for v in o["px"]]
    L = p1 - p0
    h = o["horizontal"]
    H, W = dark.shape
    best = (0.0, None)
    for name, t0, t1 in (("before", p0 - int(0.9 * L), p0 - 3), ("after", p1 + 3, p1 + int(0.9 * L))):
        for d in range(-4, 5):
            ts = [t for t in range(t0, t1) if 0 <= t < (W if h else H)]
            if not ts:
                continue
            cov = np.mean([bool(dark[c + d, t]) if h else bool(dark[t, c + d]) for t in ts])
            if cov > best[0]:
                best = (cov, name)
    return best[1] if best[0] >= ic["pocket_line_min"] else None


def drop_fit_only_wall(det, o, cfg):
    """Remove the wall the rejected opening lies on when that wall is made only of fitting edges
    (no wall gray anywhere on it): it was only ever the fitting. True when one was removed."""
    h = o["horizontal"]
    half = cfg["input"]["half_grid_mm"]
    line = round((o["p1_mm"][1] if h else o["p1_mm"][0]) / half) * half
    mid = ((o["p1_mm"][0] + o["p2_mm"][0]) / 2) if h else ((o["p1_mm"][1] + o["p2_mm"][1]) / 2)
    grid = cfg["input"]["grid_mm"]
    keep = []
    removed = False
    for w in det["walls"]:
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        on = (h and abs(y1 - line) < 1 and abs(y2 - line) < 1 and min(x1, x2) <= mid <= max(x1, x2)) or \
             (not h and abs(x1 - line) < 1 and abs(x2 - line) < 1 and min(y1, y2) <= mid <= max(y1, y2))
        length = abs(x2 - x1) + abs(y2 - y1)
        if on and w.get("fit_edges", 0) * grid >= length - 1:
            removed = True
            print(f"  壁も消す（建具の線だけの壁）: {w['p1']}→{w['p2']}")
            continue
        keep.append(w)
    det["walls"] = keep
    return removed


def seen_from_genkan(o, labels, walls, M, reach):
    """True when a 玄関 label sees the opening's middle with no wall in between."""
    (x1, y1), (x2, y2) = o["p1_mm"], o["p2_mm"]
    mid = ((x1 + x2) / 2, (y1 + y2) / 2)
    own = LineString([o["p1_mm"], o["p2_mm"]])
    for lab in labels:
        sight = LineString([(M.xmm(lab["x"]), M.ymm(lab["y"])), mid])
        if sight.length > reach:
            continue
        if not any(w.crosses(sight) for w in walls if not w.buffer(1).contains(own) and w.distance(own) > 1):
            return True
    return False


def main():
    img_path, ops_path, walls_path, names_path, cfg_path = sys.argv[1:6]
    report = "--report" in sys.argv
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    ic = cfg["interior_type"]
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR).astype(int)
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    dark = (np.abs(r - g) <= 30) & (np.abs(g - b) <= 30) & (bgr.mean(axis=2) < ic["dark_max"])
    dark_strict = bgr.mean(axis=2) < 110  # as opening_detect's arc test
    dark_light = (np.abs(r - g) <= 30) & (np.abs(g - b) <= 30) & (bgr.mean(axis=2) < ic["light_line_max"])
    ops = json.load(open(ops_path, encoding="utf-8"))
    det = json.load(open(walls_path, encoding="utf-8"))
    M = Mapper(det, cfg["input"]["grid_mm"])
    labels = [n for n in json.load(open(names_path, encoding="utf-8")) if n.get("name") == "玄関"]
    walls = [LineString([w["p1"], w["p2"]]) for w in det["walls"]]
    cut = False
    for i, o in enumerate(ops):
        if o.get("rejected") or o["wall_type"] == "exterior":
            continue
        if labels and o["kind"] in ic["kamachi_kinds"] and o.get("arc", 0) < cfg["opening_detection"]["arc_min"] \
                and seen_from_genkan(o, labels, walls, M, ic["kamachi_reach_mm"]):
            _fan, stagger, full, _span = cues(dark, int(o["px"][0]), int(o["px"][1]), float(o["px"][2]),
                                              o["horizontal"], 0, cfg["window_type"])
            if stagger == 0 and full >= ic["kamachi_min_full_lines"]:
                o["kind_before"] = o["kind"]
                o["kind"] = "kamachi"
                o["rejected"] = "框（上がり框：建具でも壁でもない）"
                cut_wall(det, o, cfg["input"]["half_grid_mm"], cfg["placement"]["jamb_search_mm"])
                cut = True
                print(f"框: {i} 幅{round(o['width_mm'])}（玄関に面した二本線。壁を消す）")
                continue
        if o["kind"] == "storage" and o.get("swing_side") and hatched(dark_strict, o, o["swing_side"], ic):
            print(f"建具ではない（ハッチング）: {i} storage 幅{round(o['width_mm'])}")
            o["kind_before"] = o["kind"]
            o["rejected"] = "玄関収納・カップボードのハッチング（建具ではない）"
            continue
        if o["kind"] == "storage":
            # a single swing door read as 両開き (プラン3 1F 洗面所, user 10/24): no half-width
            # arcs at both jambs (0.33; real 両開き 0.6-1.0), but one full-width arc
            score, _side = half_arcs(dark_strict, o)
            fa, hp, side = full_arc(dark_strict, o)
            single = score < ic["double_arc_max_single"] and fa >= cfg["opening_detection"]["arc_min"]
            # (wide closets read as storage are 折戸: they go on to the V test below)
            if score < ic["double_arc_confirm"] and not single and o["width_mm"] <= ic["swing_max_mm"]:
                # neither two half arcs nor one full arc: not a door that can be confirmed (プラン2
                # 1F, above the toilet: 0.6; real 両開き 0.93-1.0). No door rather than a wrong one
                print(f"建具とみなさない（円弧が確かめられない）: {i} storage 幅{round(o['width_mm'])} 半円弧{score:.2f}")
                o["kind_before"] = o["kind"]
                o["rejected"] = "両開きの円弧が確かめられない"
                # a wall made only of fitting edges, standing for that fitting, goes with it
                # (プラン2 1F: the stair's dashed lines over the toilet split it in three, user 10/25)
                if drop_fit_only_wall(det, o, cfg):
                    cut = True
                    continue
                # on a real wall: a sliding door whose leaf line runs on (dashed) into the wall
                # beyond one jamb is a 片引戸 (プラン4 2F 書斎, drawn with ←→)
                pk = pocket_side(dark_light, o, ic)
                if o.get("lines") and pk:
                    print(f"  → 片引戸（引き込みが {pk}）")
                    o["kind"], o["pocket"], o["rejected"] = "sliding_pocket", pk, None
                continue
            if score >= ic["double_arc_min"]:
                continue  # two half arcs drawn: a confirmed 両開き (not a 折戸 because of hangers)
            if score < ic["double_arc_max_single"] and fa >= cfg["opening_detection"]["arc_min"]:
                print(f"片開き: {i} storage 幅{round(o['width_mm'])} 半円弧{score:.2f} 円弧{fa:.2f}")
                o["kind_before"] = o["kind"]
                o["kind"] = "swing"
                o["hinge_px"], o["swing_side"] = hp, side
                continue
        if o["kind"] == "sliding" and o["width_mm"] <= ic["swing_max_mm"]:
            # a swing door whose arc is drawn light grey was read as a sliding door (プラン2 2F
            # 主寝室, user 10/24): look for a full-width arc on the lighter lines too
            fa, fhp, fside = full_arc(dark_light, o)
            if fa >= ic["swing_arc_over_fold"]:
                print(f"片開き（薄い円弧）: {i} sliding 幅{round(o['width_mm'])} 円弧{fa:.2f}")
                o["kind_before"] = o["kind"]
                o["kind"] = "swing"
                o["hinge_px"], o["swing_side"] = fhp, fside
                continue
        if o["kind"] in ("sliding", "swing"):
            # half-width arcs at both jambs: a 両開き the first pass missed (プラン2 1F 物入, it
            # was read as 引違い戸 and then as 折戸; user 10/23)
            score, side = half_arcs(dark_strict, o)
            if score >= ic["double_arc_min"]:
                print(f"両開き: {i} {o['kind']} 幅{round(o['width_mm'])} 円弧{score:.2f}")
                o["kind_before"] = o["kind"]
                o["kind"] = "storage"
                o["swing_side"], o["hinge_px"] = side, o["px"][0]
                continue
        pos, neg, where = diagonals(dark, o, ic)
        o["diagonals"] = [pos, neg]
        folding = (o["kind"] == "sliding" and min(pos, neg) >= ic["fold_min_each"]) or \
                  (o["kind"] in ("swing", "storage") and pos + neg >= ic["fold_min_total"])
        if folding:
            # a full-width swing arc means a swing door, whatever diagonals lie beside it (stair
            # lines next to a door made プラン2 2F's swing door a 折戸, user 10/24)
            fa, fhp, fside = full_arc(dark_strict, o)
            # (only door-sized openings: a wide closet's hanger lines fill any big arc)
            if fa >= ic["swing_arc_over_fold"] and o["width_mm"] <= ic["swing_max_mm"]:
                print(f"片開き（円弧あり）: {i} {o['kind']} 幅{round(o['width_mm'])} 円弧{fa:.2f}")
                o["kind_before"] = o["kind"]
                o["kind"] = "swing"
                o["hinge_px"], o["swing_side"] = fhp, fside
                continue
            o["kind_before"] = o["kind"]
            o["kind"] = "folding"
            # which jamb the V hangs from (one V under 910, one at each jamb when wider) and
            # which side it opens to
            low = sum(1 for t, _s in where if t < 0.5)
            high = len(where) - low
            o["hinge_px"] = None
            if o["width_mm"] >= ic["fold_pair_min_mm"]:
                pass  # 910 and wider: a V at each jamb (user 10/23)
            elif low >= high:
                o["hinge_px"] = o["px"][0]
            else:
                o["hinge_px"] = o["px"][1]
            sides = sum(s for _t, s in where)
            o["swing_side"] = (1 if sides > 0 else -1) if sides else None
            ends = "片側" if o["hinge_px"] is not None else "両側"
            print(f"折戸: {i} 幅{round(o['width_mm'])} 斜線{pos}/{neg} V字は{ends}")
        elif report:
            print(f"  {i} {o['kind']} 幅{round(o['width_mm'])} 斜線{pos}/{neg}")
    json.dump(ops, open(ops_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if cut:
        json.dump(det, open(walls_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
