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


def v_pair(dark, dark_strict, o, ic):
    """A 折戸 drawn with few lines (プラン6 1F 階段下物入・WIC: 2-3 straight pieces each way, so the
    swing/storage count of 10 was not reached and the V was read as a swing arc / 両開き): diagonals
    leaning both ways, and neither the two half arcs of a 両開き (0.93-1.0) nor one full-width
    swing arc (0.8 and up) drawn (user 11/17)."""
    pos, neg, where = diagonals(dark, o, ic)
    # wider than any swing door: a V at each jamb (closet 4枚折れ戸, プラン6 1F クローゼット 1/4). Its
    # V legs also fill a full-width arc; what tells it is legs that start at the frame (2 there,
    # none on プラン4 2F's W1229 swing door) and no confirmed 両開き half arcs
    wide = o["width_mm"] > ic["swing_max_mm"]
    if min(pos, neg) < (1 if wide else ic["fold_min_each"]) or (wide and pos + neg < 4):
        return False
    half, _s = half_arcs(dark_strict, o)
    if wide:
        return sum(1 for _t, s in where if s) >= 2 and half < ic["double_arc_confirm"]
    fa, _h, _s2 = full_arc(dark_strict, o)
    return half <= ic["fold_half_arc_max"] and fa < ic["swing_arc_over_fold"]


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


def count_leaves(bgr, o, ic):
    """引き戸の戸の枚数：戸1枚＝厚みの2本線の組。開いた所（帯に灰色がない所）で、通り芯から見た1本目の線の
    位置が変わる所で区切り、同じ位置が続く区間を1枚と数える（プラン6 1F ゲストルーム：-5px の組と -2px の組
    → 2枚。user 11/17）。戸袋（灰色の壁がある所）は数えない。"""
    p0, p1, c = o["px"]
    h = o["horizontal"]
    hw = ic.get("leaf_half_px", 7)
    ci = int(round(c))
    H, W = bgr.shape[:2]
    sigs = []
    for q in range(int(p0) + 2, int(p1) - 1):
        if h:
            prof = bgr[ci - hw:ci + hw + 1, q] if 0 <= q < W else None
        else:
            prof = bgr[q, ci - hw:ci + hw + 1] if 0 <= q < H else None
        if prof is None or len(prof) != 2 * hw + 1:
            continue
        v = prof.mean(axis=1)
        neutral = (np.abs(prof[:, 0] - prof[:, 1]) <= 12) & (np.abs(prof[:, 1] - prof[:, 2]) <= 12)
        if (neutral & (v > 150) & (v < 215)).sum() >= 2:
            sigs.append(None)  # 灰色の壁（戸袋・袖壁）
            continue
        d = np.nonzero(v < 110)[0]
        sigs.append(int(d[0]) if len(d) >= 2 else None)
    # 同じ位置が続く区間（1px のぶれは同じとみる）
    runs = []
    for s in sigs:
        if s is None:
            if runs and runs[-1][0] is not None:
                runs.append([None, 1])
            continue
        if runs and runs[-1][0] is not None and abs(runs[-1][0] - s) <= 1:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    total = sum(n for s, n in runs if s is not None)
    if total < 10:
        return 0
    leaves = [r for r in runs if r[0] is not None and r[1] >= ic.get("leaf_min_frac", 0.2) * total]
    # 続けて同じ位置の区間（途中に中心の直交線などが入っただけ）は1枚
    n = 0
    last = None
    for s, _n in leaves:
        if last is None or abs(s - last) > 1:
            n += 1
        last = s
    return n


def gap_only(o, M):
    """The opening without its pocket (the wall gap only), for a sliding_pocket."""
    gp = o.get("gap_pocket_px")
    if not gp or not o.get("pocket"):
        return o
    p0, p1, c = o["px"]
    q0, q1 = (p0, p0 + gp[0] - 1) if o["pocket"] == "after" else (p1 - gp[0] + 1, p1)
    if o["horizontal"]:
        a, b = sorted((M.xmm(q0), M.xmm(q1)))
        p1m, p2m = [a, o["p1_mm"][1]], [b, o["p1_mm"][1]]
    else:
        a, b = sorted((M.ymm(q0), M.ymm(q1)))
        p1m, p2m = [o["p1_mm"][0], a], [o["p1_mm"][0], b]
    return {**o, "px": [q0, q1, c], "p1_mm": p1m, "p2_mm": p2m, "width_mm": b - a}


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
    # 玄関と、玄関から続く土間（SIC など）の名前。土間の側から見える框もある（プラン6 1F：SIC の框は、
    # 玄関の名前からだと袖壁に隠れた。user 11/17）
    genkan_names = ic.get("kamachi_label_names", ["玄関"])
    labels = [n for n in json.load(open(names_path, encoding="utf-8")) if n.get("name") in genkan_names]
    walls = [LineString([w["p1"], w["p2"]]) for w in det["walls"]]
    cut = False
    for i, o in enumerate(ops):
        if o.get("rejected") or o["wall_type"] == "exterior":
            continue
        # （片引戸は、アウトセットと読んだものだけ：框の帯は隣の壁の面の上まで続き、壁が細くならない。ふつうの
        # 片引戸は戸袋の所で壁が細い。プラン3・4・5 1F の玄関近くの片引戸を框にしてしまった。user 11/17）
        og = gap_only(o, M) if o["kind"] == "sliding_pocket" and o.get("outset") else o
        if labels and o["kind"] in list(ic["kamachi_kinds"]) + (["sliding_pocket"] if ic.get("kamachi_pocket", True) and o.get("outset") else []) \
                and o.get("arc", 0) < cfg["opening_detection"]["arc_min"] \
                and seen_from_genkan(og, labels, walls, M, ic["kamachi_reach_mm"]):
            # 片引戸と読んだ框（プラン6 1F SIC）：框の白い帯が隣の壁の上まで続き、引き込みに見えた。
            # 確かめるのは壁の切れ目（引き込みを除いた所）だけ。線は1本でよい（帯の片側が壁の線と重なる）
            _fan, stagger, full, _span = cues(dark, int(og["px"][0]), int(og["px"][1]), float(og["px"][2]),
                                              og["horizontal"], 0, cfg["window_type"])
            need = ic["kamachi_min_full_lines"] if og is o else ic["kamachi_pocket_min_full_lines"]
            if o["kind"] == "sliding" and o["width_mm"] > cfg["opening_detection"].get("sliding_gap_max_mm", 99999):
                # 戸1枚より広く、戸のずれ（引違いの段）もない：框。帯の片側は壁の線と重なり、線は1本でよい
                # （プラン6 1F SIC 1189。user 11/17）
                need = ic["kamachi_pocket_min_full_lines"]
            # 框の白い帯は通り芯からずれる（プラン6 1F SIC -5.5px。建具は ±3.5px 以内。user 11/17）
            off_ = o.get("white_off_px")
            shifted = off_ is not None and abs(off_) >= ic.get("kamachi_white_off_px", 99)
            if (stagger == 0 and full >= need) or shifted:
                if og is not o:
                    o.update({k: og[k] for k in ("px", "p1_mm", "p2_mm", "width_mm")})
                    o["pocket"], o["outset"] = None, False
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
            if score < ic["double_arc_confirm"] and not single and o["width_mm"] <= ic["swing_max_mm"] \
                    and not v_pair(dark, dark_strict, o, ic):
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
            if o["width_mm"] > ic["swing_max_mm"] and not v_pair(dark, dark_strict, o, ic):
                # wider than any swing door, and no 折戸 V: a sliding door whose leaf runs on into the wall
                # (プラン6 1F ゲストルーム：2枚の戸が戸袋に引き込む。2枚の戸の端が円弧に見えた。user 11/17)
                pk = pocket_side(dark_light, o, ic)
                if o.get("lines") and pk:
                    print(f"片引戸（広い）: {i} storage 幅{round(o['width_mm'])} 引き込みが {pk}")
                    o["kind_before"] = o["kind"]
                    o["kind"], o["pocket"] = "sliding_pocket", pk
                    continue
            if score < ic["double_arc_max_single"] and fa >= cfg["opening_detection"]["arc_min"] \
                    and o["width_mm"] <= ic["swing_max_mm"]:
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
                  (o["kind"] in ("swing", "storage") and pos + neg >= ic["fold_min_total"]) or \
                  (o["kind"] in ("swing", "storage") and v_pair(dark, dark_strict, o, ic))
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
    for i, o in enumerate(ops):
        if o["kind"] == "sliding_pocket" and not o.get("rejected") and not o.get("outset"):
            n = count_leaves(bgr, o, ic)
            if n:
                o["leaves"] = n
                if n >= 2:
                    print(f"引き戸の戸の枚数: {i} 幅{round(o['width_mm'])} → {n}枚")
    json.dump(ops, open(ops_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if cut:
        json.dump(det, open(walls_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
