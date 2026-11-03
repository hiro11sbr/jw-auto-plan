"""Tell the window type from how the window is drawn, for boards that print no window code.

usage: window_type.py <plan.png> <openings.json> <config.json> <walls.json> [--report]
adds "window_type" (引違い窓 / 縦すべり窓 / FIX窓) and "window_cues" to every window in
openings.json; a printed code (read_window_labels.py, "label") still wins when there is one.

Cues (measured on the plan, 2026-10-23):
  - 縦すべり: a fan outside the wall: a diagonal line and an arc, whose distance from the wall
    changes along the window (a dashed shutter box stays at one distance).
  - 引違い: two sash lines inside the band, each covering only part of the width, staggered.
  - 横すべり: a dashed line parallel to the wall outside it (a dashed box; user 2026-10-23).
    Over only part of the window: 横すべり+FIX (TY).
  - FIX: one sash line across the whole width and nothing outside.
"""
import json
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def cues(dark, p0, p1, c, h, out_sign, wc):
    """(fan, stagger, full) for one window. dark: bool image; c: wall centre px; out_sign: the
    image direction (+1/-1 across the wall) of the outside, 0 when unknown (both sides tried)."""
    H, W = dark.shape
    half = wc["band_half_px"]
    L = p1 - p0 + 1
    pos = range(p0 + 2, p1 - 1)

    def px(t, d):
        return (int(round(c + d)), t) if h else (t, int(round(c + d)))

    # fan: outside the band, row by row away from the wall, where the ink sits along the window.
    # A fan (diagonal + arc) moves along the window as it goes out; frame or shutter lines
    # parallel to the wall fill whole rows, and the shutter's short sides sit at the two ends.
    fan, span = 0.0, 1.0
    inner = [t for t in pos if p0 + 0.08 * L <= t <= p1 - 0.08 * L]
    for s in ([out_sign] if out_sign else [1, -1]):
        # start beyond any full-length line just outside the band (a frame / sill line drawn
        # along the window, プラン2): it is not part of a fan
        # only the full-length rows themselves are skipped (and their neighbours): a long line
        # just outside (a balcony or eave edge, プラン3 2F) crosses the fan, which must still be
        # seen on both sides of it
        full_rows = set()
        for d in range(half + 2, wc["fan_reach_px"]):
            cov = np.mean([bool(dark[px(t, s * d)]) for t in inner
                           if 0 <= px(t, s * d)[0] < H and 0 <= px(t, s * d)[1] < W] or [False])
            if cov >= wc["full_cov"]:
                full_rows |= {d - 1, d, d + 1}
        # the frame / sill lines right outside the band are part of the window, not a fan
        start = half + 2
        while start in full_rows:
            start += 1
        # column by column, how far out the first ink is: a fan's distance sweeps, a shutter
        # box line stays put
        firsts = []
        for t in inner:
            for d in range(start, wc["fan_reach_px"]):
                if d in full_rows:
                    continue
                y, x = px(t, s * d)
                if 0 <= y < H and 0 <= x < W and dark[y, x]:
                    firsts.append(d)
                    break
        if len(firsts) >= wc["fan_min_rows"] and float(np.std(firsts)) > fan:
            fan = float(np.std(firsts))
            span = len(firsts) / max(1, len(inner))  # the share of the window with fan ink
    # sash lines inside the band: coverage of each row across the band
    rows = []
    for d in range(-half, half + 1):
        cov = []
        for t in pos:
            y, x = px(t, d)
            cov.append(bool(dark[y, x]) if 0 <= y < H and 0 <= x < W else False)
        cov = np.array(cov)
        if cov.any():
            # the longest continuous stretch: the sash itself (short jamb ticks at the window
            # ends would otherwise make every row look as if it reached both ends)
            best, start, run_a = (0, 0, 0), None, 0
            for i, v in enumerate(list(cov) + [False]):
                if v and start is None:
                    start = i
                elif not v and start is not None:
                    if i - start > best[0]:
                        best = (i - start, start, i - 1)
                    start = None
            n, a_, b_ = best
            rows.append((d, n / len(cov), a_ / len(cov), b_ / len(cov)))
    full = sum(1 for _d, m, _a, _b in rows if m >= wc["full_cov"])
    part = [(d, a, b) for d, m, a, b in rows if wc["part_cov"][0] <= m < wc["part_cov"][1]]
    stagger = 0
    for d1, a1, b1 in part:
        for d2, a2, b2 in part:
            # two partial sashes at different depths, one from each end, overlapping in the middle
            if abs(d1 - d2) >= 2 and a1 < 0.15 and b2 > 0.85 and a2 > 0.2 and b1 < 0.8:
                stagger += 1
    return round(fan, 1), stagger, full, round(span, 2)


def dashed(dark, p0, p1, c, h, out_sign, wc):
    """(dash count, share of the window the dashes cover end to end) of the best dashed line
    parallel to the wall outside it: the 横すべり symbol (the sash pushed out, drawn as a
    dashed box; user 2026-10-23). Dashes must be even in length and spacing (gravel is not)."""
    H, W = dark.shape
    half = wc["band_half_px"]
    L = p1 - p0 + 1
    best = (0, 0.0)

    def row(s, d):
        out = []
        for t in range(p0, p1 + 1):
            y, x = (int(round(c + s * d)), t) if h else (t, int(round(c + s * d)))
            out.append(bool(dark[y, x]) if 0 <= y < H and 0 <= x < W else False)
        return np.array(out)

    for s in ([out_sign] if out_sign else [1, -1]):
        for d in range(half + 2, wc["fan_reach_px"] + 15):
            r = row(s, d)
            runs, start = [], None
            for i, v in enumerate(list(r) + [False]):
                if v and start is None:
                    start = i
                elif not v and start is not None:
                    runs.append((start, i - 1))
                    start = None
            if len(runs) < wc["dash_min_count"]:
                continue
            lens = np.array([b - a + 1 for a, b in runs])
            gaps = np.array([runs[k + 1][0] - runs[k][1] - 1 for k in range(len(runs) - 1)])
            dl, dg = wc["dash_len_px"], wc["dash_gap_px"]
            ok_len = (lens >= dl[0]) & (lens <= dl[1])
            ok_gap = (gaps >= dg[0]) & (gaps <= dg[1])
            if ok_len.mean() < 0.8 or ok_gap.mean() < 0.8:
                continue
            # a line of its own: the rows just beside it are mostly blank (gravel is not)
            side = max(row(s, d - 3).mean(), row(s, d + 3).mean())
            if side > 0.5 * r.mean():
                continue
            cover = (runs[-1][1] - runs[0][0] + 1) / L
            if len(runs) > best[0]:
                best = (len(runs), round(cover, 2))
    return best


def fan_diagonal(dark, p0, p1, c, h, out_sign, wc):
    """The straight leg of a 縦すべり fan: a diagonal (20-70 deg to the wall) on the outside, starting
    at the window frame. Returns (found, pivot px along the wall, share of the window it covers)
    or (False, None, 0). Run on the speck-free image, so gravel dots cannot form it."""
    import math
    L = p1 - p0
    best = (False, None, 0.0)
    for s in ([out_sign] if out_sign else [1, -1]):
        d0, d1 = sorted((c + s * 4, c + s * (wc["fan_reach_px"] + 10)))
        d0, d1 = int(d0), int(d1)
        y0, y1, x0, x1 = (d0, d1, p0 - 4, p1 + 5) if h else (p0 - 4, p1 + 5, d0, d1)
        H, W = dark.shape
        y0, x0, y1, x1 = max(0, y0), max(0, x0), min(H, y1), min(W, x1)
        m = (dark[y0:y1, x0:x1] * 255).astype(np.uint8)
        if m.size == 0:
            continue
        segs = cv2.HoughLinesP(m, 1, np.pi / 180, 10, minLineLength=max(wc["fan_leg_min_px"], wc["fan_diag_min"] * L),
                               maxLineGap=2)
        if segs is None:
            continue
        for xa, ya, xb, yb in segs.reshape(-1, 4):
            xa, ya, xb, yb = int(xa) + x0, int(ya) + y0, int(xb) + x0, int(yb) + y0
            (ta, da), (tb, db) = ((xa, ya - c), (xb, yb - c)) if h else ((ya, xa - c), (yb, xb - c))
            ang = math.degrees(math.atan2(abs(db - da), abs(tb - ta) + 1e-9))
            if not 20 <= ang <= 70:
                continue
            # one end at the frame (close to the wall), the other out
            near, far = ((ta, da), (tb, db)) if abs(da) < abs(db) else ((tb, db), (ta, da))
            if abs(near[1]) > wc["band_half_px"] + 6:
                continue
            share = abs(tb - ta) / max(1, L)
            if share > best[2]:
                best = (True, near[0], share)
    return best


def decide(fan, stagger, full, span, wc, dash=(0, 0.0)):
    if dash[0] >= wc["dash_min_count"] and not (fan >= wc.get("dash_yields_fan_px", 8) and span >= wc.get("dash_yields_fan_span", 0.5)):
        # (a strong fan wins: dashes far from the window are an outside edge, not its box: プラン3 2F 入隅の窓, user 10/29)
        # dashed box over only part of the window: a 横すべり beside a FIX (TY 連窓)
        return "横すべり+FIX" if dash[1] < wc["ty_max_span"] else "横すべり窓"
    if fan >= wc["fan_min_spread_px"] and span >= wc["fan_min_span"]:
        # a fan over only part of a long window: an opening sash beside a FIX (TF 連窓)
        return "縦すべり+FIX" if span < wc["tf_max_span"] else "縦すべり窓"
    if stagger:
        return "引違い窓"
    return "FIX窓"


def outside_signs(walls_path, ops, grid):
    """Image direction (+1/-1 across the wall) of the outdoors for each window, from the
    building outline made of the detected walls; 0 when it cannot be told."""
    from pathlib import Path

    from shapely.geometry import LineString, Point
    from shapely.ops import polygonize, unary_union

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from opening_detect import Mapper

    det = json.load(open(walls_path, encoding="utf-8"))
    polys = list(polygonize(unary_union([LineString([w["p1"], w["p2"]]) for w in det["walls"]])))
    if not polys:
        return {}
    fp = unary_union(polys)
    out = {}
    for i, o in enumerate(ops):
        if o.get("rejected") or o["kind"] != "window":
            continue
        (x1, y1), (x2, y2) = o["p1_mm"], o["p2_mm"]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if o["horizontal"]:
            plus_out = not fp.contains(Point(mx, my + 300))
            out[i] = -1 if plus_out else 1  # model +Y is image -y
        else:
            plus_out = not fp.contains(Point(mx + 300, my))
            out[i] = 1 if plus_out else -1
    return out


def main():
    img_path, ops_path, cfg_path, walls_path = sys.argv[1:5]
    report = "--report" in sys.argv
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    wc = cfg["window_type"]
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR).astype(int)
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    neutral = (np.abs(r - g) <= 30) & (np.abs(g - b) <= 30)  # red labels are not drawing lines
    # a thin dark line over the lawn takes a green tint from JPEG and drops out of `neutral`, which
    # broke a 縦すべり's fan leg into specks (プラン3 1F TF window read as 引違い, user 10/27). Leaving
    # out only red finds that leg, but used everywhere it lost other legs and found dashes in
    # hatching: it is the second look for a leg only
    dark_wide = ~((r - g > 40) & (r - b > 40)) & (bgr.mean(axis=2) < wc["dark_max"])
    dark = neutral & (bgr.mean(axis=2) < wc["dark_max"])
    # gravel round the building is drawn as scattered dots; lined up by chance they looked like a
    # fan (プラン2 1F: four FIX windows read as 縦すべり, user 10/24). A fan is a long connected
    # stroke: drop small separate specks first
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) < wc["speck_max_px"]
    dark_fan = dark & ~small[lab]
    ops = json.load(open(ops_path, encoding="utf-8"))
    signs = outside_signs(walls_path, ops, cfg["input"]["grid_mm"])
    # a 両開き (storage) in an exterior wall with hatching filling its swing side: the close
    # parallel lines beside a window (its 格子 / a storage by it) score as two arcs. A storage's
    # double doors are never in an exterior wall: a window (プラン3 1F 玄関, user 10/27)
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
    from interior_type import hatched
    dark_strict = bgr.mean(axis=2) < 110
    for i, o in enumerate(ops):
        if (not o.get("rejected") and o["kind"] == "storage" and o.get("wall_type") == "exterior"
                and o.get("swing_side") and hatched(dark_strict, o, o["swing_side"], cfg["interior_type"])):
            o["kind_before"], o["kind"] = o["kind"], "window"
            print(f"窓に直した（外壁の両開き＋ハッチング）: {i} 幅{round(o['width_mm'])}")
    for i, o in enumerate(ops):
        if o.get("rejected") or o["kind"] != "window":
            continue
        p0, p1, c = o["px"]
        out_sign = signs.get(i, 0)
        # (fans on the speck-free image; dashed 横すべり lines are made of short dashes, so they
        # keep the full image)
        fan, stagger, full, span = cues(dark_fan, int(p0), int(p1), float(c), o["horizontal"], out_sign, wc)
        dash = dashed(dark, int(p0), int(p1), float(c), o["horizontal"], out_sign, wc)
        o["window_cues"] = {"fan": fan, "stagger": stagger, "full": full, "fan_span": span,
                            "dashes": dash[0], "dash_span": dash[1]}
        o["window_type"] = decide(fan, stagger, full, span, wc, dash)
        # the fan's straight leg, found as a line: decides a fan the sweep test misses (small
        # windows, frame lines in the way) and gives the pivot end (user 10/24: a 縦すべり opened
        # the wrong way)
        # (on the full image: a thin diagonal breaks into specks too; a straight run of nearly
        # touching pixels is still not something scattered gravel dots make)
        found, pivot, share = fan_diagonal(dark, int(p0), int(p1), float(c), o["horizontal"], out_sign, wc)
        if not found:
            found, pivot, share = fan_diagonal(dark_wide, int(p0), int(p1), float(c), o["horizontal"], out_sign, wc)
        o["window_cues"]["diagonal"] = round(share, 2) if found else 0.0
        if found and o["window_type"] in ("FIX窓", "縦すべり窓", "縦すべり+FIX", "引違い窓") and dash[0] < wc["dash_min_count"]:
            # the leg says "a fan is here" and where it pivots; whether it is a TF 連窓 stays with
            # the sweep test (a single 縦すべり's leg also covers only 37-68% of the width)
            # (an 引違い has no leg: two sashes side by side, one 縦すべり, read as staggered)
            if o["window_type"] in ("FIX窓", "引違い窓"):
                # a short leg on a wide window: only part of it opens (TF 連窓, プラン2 2F W1235:
                # leg over 14% of the width; a single 縦すべり's leg covers 37-68%)
                o["window_type"] = "縦すべり+FIX" if share < wc["tf_leg_max_share"] else "縦すべり窓"
            # a window drawn like a door (a quarter circle from the pivot jamb, a straight leaf) keeps the
            # pivot the circle gives: the diagonal found here may be the circle's chord, with its near end
            # at the wrong jamb (プラン2 2F バルコニーの縦すべり, user 10/29)
            if o.get("arc", 0) < wc.get("arc_keeps_hinge", 0.95):
                o["hinge_px"] = int(pivot)
        if report:
            lab = (o.get("label") or {}).get("code", "")
            print(i, "横" if o["horizontal"] else "縦", round(o["width_mm"]), o["window_cues"], o["window_type"], lab)
    json.dump(ops, open(ops_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
