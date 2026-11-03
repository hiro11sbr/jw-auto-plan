"""Openings (doors / windows) along detected walls, at pixel resolution.

usage: opening_detect.py <plan.png> <walls.json> <config.json> <out_prefix>
writes <out_prefix>_openings.json and <out_prefix>_openings_check.png

Along each wall the gray band is followed pixel by pixel. Where it is interrupted, the gap is an
opening candidate, judged by what is drawn there:
  - continuous lines parallel to the wall across the gap = a fitting (window / sliding leaf)
  - a quarter arc from one jamb = swing door
  - leaf lines that run on into the wall beside the gap = sliding door pocket (引込み部)
  - dark strokes that are not continuous lines = text over the wall -> treated as wall
  - narrow interior gaps = built-in joinery (造作) -> ignored
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
import jp_text  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def load(cfg_path, walls_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    det = json.load(open(walls_path, encoding="utf-8"))
    return cfg, det


class Mapper:
    """mm <-> px using the per-axis grid fits written by grid_detect.py."""

    def __init__(self, det, grid):
        sx, ix, nx0, _ = det["fits"]["x"]
        sy, iy, _, ny1 = det["fits"]["y"]
        self.sx, self.ix, self.nx0, self.sy, self.iy, self.ny1, self.g = sx, ix, nx0, sy, iy, ny1, grid

    def x(self, xmm):
        return self.ix + self.sx * (xmm / self.g + self.nx0)

    def y(self, ymm):
        return self.iy + self.sy * (self.ny1 - ymm / self.g)

    def xmm(self, px):
        return ((px - self.ix) / self.sx - self.nx0) * self.g

    def ymm(self, py):
        return (self.ny1 - (py - self.iy) / self.sy) * self.g


def profiles(img, gray_range, c, positions, horizontal, half):
    """Return (gray, dark, red) bool arrays of shape (len(positions), 2*half+1) across the wall."""
    ci = int(round(c))
    rows = []
    for p in positions:
        if horizontal:
            prof = img[ci - half:ci + half + 1, p]
        else:
            prof = img[p, ci - half:ci + half + 1]
        rows.append(prof)
    a = np.array(rows).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    v = a.mean(axis=2)
    neutral = (np.abs(r - g) <= 12) & (np.abs(g - b) <= 12)
    lo, hi = gray_range
    gray = neutral & (v >= lo) & (v <= hi)
    dark = v < 110
    red = (r - g > 60) & (r - b > 60)
    return gray, dark, red


def wall_at(gray, red, half, tmin=4, tmax=22):
    """Per position: is the wall band (gray run through the centre) present?"""
    out = np.zeros(len(gray), bool)
    fill = gray | red
    for i, row in enumerate(fill):
        if not row[half - 3:half + 4].any():
            continue
        # run covering the centre region
        idx = [k for k in range(half - 3, half + 4) if row[k]][0]
        a = idx
        while a > 0 and row[a - 1]:
            a -= 1
        e = idx
        while e < len(row) - 1 and row[e + 1]:
            e += 1
        t = e - a + 1
        out[i] = t >= tmin  # a crossing wall (long run) also counts as wall here
    return out


def runs(mask):
    m = np.concatenate(([0], mask.astype(int), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def parallel_lines(dark, half, min_cov):
    """Offsets (across the wall) where a dark line runs along most of the span."""
    cov = []
    for o in range(dark.shape[1]):
        lo, hi = max(0, o - 1), min(dark.shape[1], o + 2)
        cov.append(dark[:, lo:hi].any(axis=1).mean())
    cov = np.array(cov)
    lines = []
    for o in range(len(cov)):
        if cov[o] >= min_cov and cov[o] == cov[max(0, o - 1):o + 2].max():
            if not lines or o - lines[-1] > 1:
                lines.append(o)
    return [o - half for o in lines], cov


def distinct_lines(dark, half, min_cov, reach):
    """Physical lines across the wall: contiguous offsets above the coverage threshold count as
    one line (parallel_lines() reports a 1px line at several offsets, which the opening
    thresholds were tuned on; storage fronts need the true count)."""
    cov = np.array([dark[:, o].mean() for o in range(dark.shape[1])])
    above = [o for o in range(len(cov)) if cov[o] >= min_cov and abs(o - half) <= reach]
    groups = []
    for o in above:
        if groups and o - groups[-1][-1] == 1:
            groups[-1].append(o)
        else:
            groups.append([o])
    return [round(sum(g) / len(g) - half, 1) for g in groups]


def arc_score(img_dark, hinge, radius, along_h, side, toward):
    """Fraction of a quarter circle that is drawn: hinge at `hinge`, leaf closing toward the
    other jamb (`toward` = +1/-1 along the wall), swinging to `side` of the wall."""
    hx, hy = hinge
    hits, n = 0, 0
    H, W = img_dark.shape
    for deg in range(15, 76, 3):
        t = math.radians(deg)
        if along_h:
            x = hx + toward * radius * math.cos(t)
            y = hy + side * radius * math.sin(t)
        else:
            x = hx + side * radius * math.sin(t)
            y = hy + toward * radius * math.cos(t)
        xi, yi = int(round(x)), int(round(y))
        if 1 <= xi < W - 1 and 1 <= yi < H - 1:
            n += 1
            hits += img_dark[yi - 1:yi + 2, xi - 1:xi + 2].any()
    return hits / max(n, 1)


def leaf_extent(dark_line_fn, start, step, limit):
    """Follow a leaf line from `start` in direction `step` while it stays drawn."""
    miss, p, last = 0, start, start
    while abs(p - start) < limit:
        if dark_line_fn(p):
            last, miss = p, 0
        else:
            miss += 1
            if miss > 3:
                break
        p += step
    return last


def detect(img, cfg, det):
    oc = cfg["opening_detection"]
    grid = cfg["input"]["grid_mm"]
    M = Mapper(det, grid)
    half = oc["band_half_px"]
    v = img.astype(int).mean(axis=2)
    img_dark = v < 110
    lo, hi = det["gray_range"]
    mid, w = (lo + hi) / 2, cfg["grid_detection"]["gray_halfwidth"]
    gray_range = (mid - w, mid + w)
    mmpp = det["mm_per_px"]
    out = []
    for wall in det["walls"]:
        (x1, y1), (x2, y2) = wall["p1"], wall["p2"]
        h = abs(y1 - y2) < 1
        if h:
            c = M.y(y1)
            pa, pb = sorted((M.x(x1), M.x(x2)))
        else:
            c = M.x(x1)
            pa, pb = sorted((M.y(y1), M.y(y2)))
        pa, pb = int(math.ceil(pa)), int(math.floor(pb))
        pos = np.arange(pa, pb + 1)
        if len(pos) < 10:
            continue
        gray, dark, red = profiles(img, gray_range, c, pos, h, half)
        wl = wall_at(gray, red, half)
        # wall ends sit inside the crossing walls: never call the first/last px an opening
        wl[:oc["end_guard_px"]] = True
        wl[-oc["end_guard_px"]:] = True
        # close short gaps in the wall (red text, JPEG)
        for s, e in runs(~wl):
            if e - s <= oc["min_gap_px"]:
                wl[s:e] = True
        for s, e in runs(~wl):
            seg_dark = dark[s:e]
            lines, cov = parallel_lines(seg_dark, half, oc["line_min_cov"])
            band = seg_dark[:, half - 8:half + 9]
            ink = band.mean()
            p0, p1 = pos[s], pos[e - 1]
            length_px = p1 - p0 + 1
            # swing door: quarter arc from either jamb, on either side
            best = (0.0, None, None)
            # the leaf is a little shorter than the gap (frames), and the hinge sits a few px
            # inside the jamb, so try a small range of radii and hinge offsets
            for hinge_p, toward in ((p0, 1), (p1, -1)):
                for side in (1, -1):
                    for k in (0.8, 0.85, 0.9, 0.95, 1.0):
                        for dh in (0, 2, 4):
                            hp = hinge_p + toward * dh
                            hinge = (hp, c) if h else (c, hp)
                            sc = arc_score(img_dark, hinge, length_px * k, h, side, toward)
                            if sc > best[0]:
                                best = (sc, hinge_p, side)
            arc, hinge_p, swing_side = best
            # double door (両開き): an open leaf of half the width standing at BOTH jambs on the
            # same side (a single door has one, of the full width, at its hinge jamb)
            def leaf(pj, side, t0, t1):
                ok = 0
                for t in range(int(t0), int(t1)):
                    yy, xx = (int(round(c + side * t)), pj) if h else (pj, int(round(c + side * t)))
                    win = img_dark[yy, max(0, xx - 3):xx + 4] if h else img_dark[max(0, yy - 3):yy + 4, xx]
                    ok += bool(win.any())
                return ok / max(1, int(t1) - int(t0))

            double = (0.0, None)
            t0 = half  # start beyond the wall band
            # the gap ends may sit a few px off the leaves (frame lines, or a label over the
            # wall stretching the gap: 1F 階段下収納 「06509」), so each end is searched separately
            def end_leaf(pj, inward, side):
                best = 0.0
                for d in range(-4, 13, 2):
                    pp = pj + inward * d
                    # a half-width leaf that does not run on to the full width (a single door's
                    # leaf, or a wall standing at the jamb, would)
                    if leaf(pp, side, length_px * 0.65, length_px * 0.95) < 0.5:
                        best = max(best, leaf(pp, side, t0, length_px / 2))
                return best

            for side in (1, -1):
                sc = min(end_leaf(p0, 1, side), end_leaf(p1, -1, side))
                if sc > double[0]:
                    double = (sc, side)
            kind = None
            pocket = None
            pocket_px = None  # (gap, pocket) in px, for the sliding door size
            pocket_face = None  # image side of the wall centre the leaf runs on (-1 up/left, +1 down/right)
            one_side = bool(lines) and (min(lines) > 0 or max(lines) < 0)
            spread = (max(lines) - min(lines)) if lines else 0
            if arc >= oc["entrance_arc_min"] and double[0] >= oc["double_leaf_min"] and (
                    wall["type"] != "exterior" or (p1 - p0 + 1) * mmpp >= oc["storage_min_mm"]):
                # 両開き: open leaves at both jambs. Checked first and on any wall: a storage front
                # may continue an exterior wall line (1F 階段下収納); single doors and windows
                # never show two half-width leaves. A narrow one in an outside wall is a 縦すべり
                # window whose fan looks like two leaves (プラン2 1F, W414)
                kind = "storage"
                swing_side, hinge_p, arc = double[1], p0, double[0]
            elif wall["type"] == "exterior" and arc >= oc["entrance_arc_min"] and one_side:
                kind = "entrance_door"  # door + frame lines on one face, swing arc outside
            elif wall["type"] == "exterior" and len(lines) >= 3:
                kind = "window"  # sash lines straddle the wall; windows have no swing arc
            elif arc >= oc["arc_min"] and lines and spread >= oc["joinery_min_spread_px"]:
                kind = "ub_door"  # bathroom door: swing arc plus frame lines across the depth
            elif arc >= oc["arc_min"] and (lines or ink <= oc["swing_max_ink"]):
                kind = "swing"  # a door gap is empty; text / dotted lines over the wall are not
            elif wall["type"] != "exterior" and spread >= oc["joinery_min_spread_px"]:
                kind = "joinery"  # lines across the whole depth: a cabinet built into the wall
            elif lines:
                # does a leaf line run on into the wall beside the gap? -> sliding pocket.
                # Only lines inside the wall band count: a box drawn against the wall face
                # (e.g. a 370L unit) would otherwise look like a leaf running on.
                inner = [o for o in lines if abs(o) <= oc["pocket_line_max_offset_px"]] or lines
                ext = {}
                for o in inner:
                    oi = o + half

                    def drawn(p, oi=oi):
                        if h:
                            col = img_dark[int(round(c)) - half + oi - 1:int(round(c)) - half + oi + 2, p]
                        else:
                            col = img_dark[p, int(round(c)) - half + oi - 1:int(round(c)) - half + oi + 2]
                        return col.any()

                    lim = int(length_px * 1.3)
                    ext[o] = (p0 - leaf_extent(drawn, p0, -1, lim), leaf_extent(drawn, p1, 1, lim) - p1)
                before = max(e_[0] for e_ in ext.values())
                after = max(e_[1] for e_ in ext.values())
                if max(before, after) >= oc["pocket_min_ratio"] * length_px:
                    kind = "sliding_pocket"
                    pocket = "before" if before >= after else "after"
                    pocket_px = [int(p1 - p0 + 1), int(max(before, after))]
                    # the face the door slides on: the side where the line that runs on into the wall
                    # (the leaf, drawn dashed in the pocket) sits (user 10/29: it is the drawn face)
                    k_ = 0 if pocket == "before" else 1
                    best_ = max(e_[k_] for e_ in ext.values())
                    # (of the lines that run on as far as the longest, the one furthest from the wall
                    # centre: the dashed leaf sits on the face, a frame line may sit on the centre)
                    o_star = max((o_ for o_ in ext if ext[o_][k_] >= 0.9 * best_), key=abs)
                    pocket_face = (1 if o_star > 0 else -1) if abs(o_star) >= 1 else None
                    if pocket == "before":
                        p0 = p0 - before
                    else:
                        p1 = p1 + after
                elif len(lines) >= 3 and wall["type"] == "exterior":
                    kind = "window"
                else:
                    kind = "sliding" if wall["type"] != "exterior" else "window"
            elif ink >= oc["text_min_ink"]:
                kind = "text"  # strokes over the wall but no lines: lettering, not an opening
            else:
                kind = "open"
            width_mm = (p1 - p0 + 1) * mmpp
            reason = None
            if kind in ("text", "open", "joinery"):
                reason = kind
            elif wall["type"] != "exterior" and width_mm < oc["min_interior_mm"]:
                reason = "joinery"  # narrow interior gap: built-in storage / shelf front
            elif width_mm < oc["min_exterior_mm"]:
                reason = "too_small"
            if h:
                a_mm, b_mm = M.xmm(p0), M.xmm(p1)
                pts = ((a_mm, M.ymm(c)), (b_mm, M.ymm(c)))
            else:
                a_mm, b_mm = M.ymm(p1), M.ymm(p0)
                pts = ((M.xmm(c), a_mm), (M.xmm(c), b_mm))
            out.append({
                "kind": kind, "rejected": reason, "horizontal": h, "wall_type": wall["type"],
                "p1_mm": [round(v_, 1) for v_ in pts[0]], "p2_mm": [round(v_, 1) for v_ in pts[1]],
                "width_mm": round(width_mm), "px": [int(p0), int(p1), round(float(c), 1)],
                "lines": lines, "arc": round(arc, 2), "double_arc": round(double[0], 2), "ink": round(float(ink), 3),
                # the hinge end matters for every opening that swings: doors, the entrance door,
                # the UB door and 縦すべり windows (their sash arc). Kept when an arc was found.
                "hinge_px": int(hinge_p) if hinge_p is not None and arc >= oc["hinge_arc_min"] else None,
                "swing_side": swing_side if hinge_p is not None and arc >= oc["hinge_arc_min"] else None,
                "pocket": pocket, "gap_pocket_px": pocket_px, "pocket_face": pocket_face,
            })
    # A fitting never sits on a line with no wall centre (user, 2026-10-22): fittings missed
    # here mean the wall judgement must improve, so storage_fronts() is off by default.
    if oc.get("scan_lines_without_walls"):
        out += storage_fronts(img, cfg, det, M, gray_range, img_dark, out)
    return out


def storage_fronts(img, cfg, det, M, gray_range, img_dark, found):
    """Fittings on lines that are not walls at all (closet / storage fronts).

    Every 455 and 227.5 line inside the building is followed pixel by pixel. A stretch where
    two or more lines run parallel to it (door leaves / frames), the wall band is absent, and
    both ends stop against a wall (a stub, jamb or crossing wall), is a storage fitting.
    """
    oc = cfg["opening_detection"]
    half = oc["band_half_px"]
    grid_half = cfg["input"]["half_grid_mm"]
    mmpp = det["mm_per_px"]
    xs = [v for w in det["walls"] for v in (w["p1"][0], w["p2"][0])]
    ys = [v for w in det["walls"] for v in (w["p1"][1], w["p2"][1])]
    x_lo, x_hi, y_lo, y_hi = max(0, min(xs)), max(xs), max(0, min(ys)), max(ys)
    wall_lines = {}
    for w in det["walls"]:
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        h = abs(y1 - y2) < 1
        key = (h, round(y1 if h else x1, 1))
        wall_lines.setdefault(key, []).append(sorted((x1, x2) if h else (y1, y2)))
    gray_img = None
    out = []
    win = oc["storage_window_px"]
    for h in (True, False):
        lo, hi = (y_lo, y_hi) if h else (x_lo, x_hi)
        a_lo, a_hi = (x_lo, x_hi) if h else (y_lo, y_hi)
        n0, n1 = int(math.ceil(lo / grid_half)), int(math.floor(hi / grid_half))
        for n in range(n0 + 1, n1):  # outer lines are exterior walls, not storage fronts
            c_mm = n * grid_half
            c = M.y(c_mm) if h else M.x(c_mm)
            pa, pb = sorted((M.x(a_lo), M.x(a_hi)) if h else (M.y(a_lo), M.y(a_hi)))
            pos = np.arange(int(pa) + 2, int(pb) - 2)
            if len(pos) < 40:
                continue
            gray, dark, red = profiles(img, gray_range, c, pos, h, half)
            wall_here = wall_at(gray, red, half)
            # local parallel-line count around each position
            band = dark[:, half - oc["storage_band_px"]:half + oc["storage_band_px"] + 1]
            k = np.ones(2 * win + 1) / (2 * win + 1)
            cov = np.array([np.convolve(band[:, j], k, mode="same") for j in range(band.shape[1])]).T
            strong = cov >= oc["storage_line_cov"]
            nlines = np.array([np.count_nonzero(np.diff(np.concatenate(([0], r.astype(int)))) == 1)
                               for r in strong])
            cand = (nlines >= 2) & ~wall_here
            for s, e in runs(~cand):  # close small gaps
                if e - s <= 5 and s > 0 and e < len(cand):
                    cand[s:e] = True
            for s, e in runs(cand):
                if (e - s) * mmpp < oc["storage_min_mm"] or (e - s) * mmpp > oc["storage_max_mm"]:
                    continue
                p0, p1 = int(pos[s]), int(pos[e - 1])
                # one end must meet a real (gray) wall; the other a wall, a frame line, or a
                # detected wall such as a pocket. Furniture edges end on furniture lines.
                ends = []
                for p, d_ in ((p0, -1), (p1, 1)):
                    e_ = wall_near(img, gray_range, c, p, d_, h)
                    if e_ is None and anchored(det, M, h, c_mm, p, d_):
                        e_ = "wall"
                    ends.append(e_)
                if None in ends or not ({"gray", "wall"} & set(ends)):
                    continue
                if h:
                    a_mm, b_mm = M.xmm(p0), M.xmm(p1)
                    pts = ((a_mm, c_mm), (b_mm, c_mm))
                else:
                    a_mm, b_mm = M.ymm(p1), M.ymm(p0)
                    pts = ((c_mm, a_mm), (c_mm, b_mm))
                # skip what the wall scan already found, and stretches lying on a detected wall
                lo_mm, hi_mm = sorted((a_mm, b_mm))
                on_wall = any(lo_mm < s2 - 100 and hi_mm > s1 + 100 or (s1 <= lo_mm + 50 and hi_mm - 50 <= s2)
                              for s1, s2 in wall_lines.get((h, round(c_mm, 1)), []))
                dup = any(f["horizontal"] == h and abs((f["p1_mm"][1] if h else f["p1_mm"][0]) - c_mm) < 100
                          and min(f["p1_mm"][0 if h else 1], f["p2_mm"][0 if h else 1]) < hi_mm
                          and max(f["p1_mm"][0 if h else 1], f["p2_mm"][0 if h else 1]) > lo_mm
                          for f in found + out)
                if on_wall or dup:
                    continue
                seg_lines = distinct_lines(dark[s:e], half, oc["storage_leaf_cov"], oc["storage_band_px"])
                spread = (max(seg_lines) - min(seg_lines)) if seg_lines else 99
                centre = (sum(seg_lines) / len(seg_lines)) if seg_lines else 99
                # swing storage doors (single or a pair) show quarter arcs from the jambs
                L = p1 - p0 + 1
                arc = 0.0
                for radius, hinges in ((L, ((p0, 1), (p1, -1))), (L / 2, ((p0, 1), (p1, -1)))):
                    for hp, toward in hinges:
                        for side in (1, -1):
                            hinge = (hp, c) if h else (c, hp)
                            arc = max(arc, arc_score(img_dark, hinge, radius * 0.9, h, side, toward))
                # a closed leaf (sliding / folding) is two lines close together, on the grid
                # line; counters, bath tubs, stair treads and furniture give lines off the line,
                # far apart, or more of them
                leaf = (len(seg_lines) == 2 and spread <= oc["storage_max_spread_px"]
                        and abs(centre) <= oc["storage_max_centre_px"])
                reason = None if (leaf or arc >= oc["arc_min"]) else "not_leaf"
                # both sides of a storage front are open space (room / inside the closet); the
                # edge of a counter top has the gray-filled counter on one side
                if reason is None and side_filled(img, gray_range, c, p0, p1, h):
                    reason = "counter"
                out.append({
                    "kind": "storage", "rejected": reason, "horizontal": h, "wall_type": "none",
                    "p1_mm": [round(v_, 1) for v_ in pts[0]], "p2_mm": [round(v_, 1) for v_ in pts[1]],
                    "width_mm": round((p1 - p0 + 1) * mmpp), "px": [p0, p1, round(float(c), 1)],
                    "ends": ends,
                    "lines": seg_lines, "arc": round(arc, 2), "ink": 0.0, "hinge_px": None, "swing_side": None,
                    "pocket": None,
                })
    return out


def wall_near(img, gray_range, c, p, direction, horizontal, reach=20, across=12):
    """Is there wall (gray) right past position p along the line, or crossing it, at the end?"""
    lo, hi = gray_range
    ci, pi = int(round(c)), int(round(p))
    a, b = sorted((pi, pi + direction * reach))
    if horizontal:
        box_ = img[ci - across:ci + across + 1, a:b + 1]
    else:
        box_ = img[a:b + 1, ci - across:ci + across + 1]
    if box_.size == 0:
        return None
    v = box_.astype(int).mean(axis=2)
    if float(((v >= lo) & (v <= hi)).mean()) >= 0.2:
        return "gray"
    # a frame / jamb line drawn across the line (black, not gray) also closes the end
    dk = v < 110
    across_frac = dk.mean(axis=1) if horizontal else dk.mean(axis=0)
    return "line" if bool((across_frac >= 0.6).any()) else None


def side_filled(img, gray_range, c, p0, p1, horizontal, near=12, far=26):
    """Is one side of the stretch (offset near..far px from the line) mostly gray fill?"""
    lo, hi = gray_range
    ci = int(round(c))
    a, b = int(p0) + 3, int(p1) - 3
    if b <= a:
        return False
    for sign in (1, -1):
        r0, r1 = sorted((ci + sign * near, ci + sign * far))
        box_ = img[r0:r1, a:b] if horizontal else img[a:b, r0:r1]
        if box_.size == 0:
            continue
        v = box_.astype(int).mean(axis=2)
        if float(((v >= lo) & (v <= hi)).mean()) >= 0.5:
            return True
    return False


def anchored(det, M, h, c_mm, p, direction, reach_px=20):
    """Does a detected wall cross this line just past position p (e.g. a pocket wall)?"""
    for w in det["walls"]:
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        wh = abs(y1 - y2) < 1
        if wh == h:
            continue
        lo_, hi_ = sorted((y1, y2) if h else (x1, x2))  # the crossing wall's own span
        if not (lo_ - 50 <= c_mm <= hi_ + 50):
            continue
        t_px = M.y(y1) if h is False else M.x(x1)  # the crossing wall's position along our line
        if 0 <= (t_px - p) * direction <= reach_px:
            return True
    return False


COLORS = {"swing": (0, 160, 0), "sliding_pocket": (200, 0, 200), "sliding": (0, 140, 255),
          "window": (220, 120, 0), "entrance_door": (0, 90, 170), "ub_door": (120, 160, 0),
          "storage": (0, 200, 200)}
KIND_JA = {"swing": "開き戸", "sliding_pocket": "引戸＋引込み部", "sliding": "引戸・その他の建具",
           "window": "窓", "entrance_door": "玄関ドア", "ub_door": "浴室（UB）ドア",
           "storage": "収納建具（壁のない線上）"}


def draw(img, openings, path, title):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(bgr, 0.55, np.full_like(bgr, 255), 0.45, 0)
    n = 0
    for o in openings:
        p0, p1, c = o["px"]
        c = int(round(c))
        pad = 9
        if o["rejected"]:
            if o["rejected"] in ("text", "joinery"):
                col = (150, 150, 150)
                if o["horizontal"]:
                    cv2.line(out, (p0, c), (p1, c), col, 1)
                else:
                    cv2.line(out, (c, p0), (c, p1), col, 1)
            continue
        n += 1
        o["no"] = n
        col = COLORS.get(o["kind"], (0, 0, 220))
        if o["horizontal"]:
            cv2.rectangle(out, (p0, c - pad), (p1, c + pad), col, 2)
            t = (int((p0 + p1) / 2) - 8, c - pad - 4)
        else:
            cv2.rectangle(out, (c - pad, p0), (c + pad, p1), col, 2)
            t = (c + pad + 3, int((p0 + p1) / 2) + 5)
        cv2.putText(out, str(n), t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
    rows = [(COLORS[k], KIND_JA[k]) for k in COLORS]
    rows.append(((150, 150, 150), "除外（壁にかかる文字・造作）"))
    out = jp_text.legend(out, title, rows, box=True)
    cv2.imencode(".png", out)[1].tofile(path)


def main():
    img_path, walls_path, cfg_path, prefix = sys.argv[1:5]
    cfg, det = load(cfg_path, walls_path)
    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    ops = detect(img, cfg, det)
    name = prefix.replace("\\", "/").split("/")[-1]
    draw(img, ops, prefix + "_建具の判定.png", f"{name}　建具の判定")
    json.dump(ops, open(prefix + "_openings.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    kept = [o for o in ops if not o["rejected"]]
    print(f"{name}: candidates={len(ops)} kept={len(kept)} "
          + " ".join(f"{k}={sum(o['kind'] == k for o in kept)}" for k in COLORS))
    for o in kept:
        print(f"  #{o.get('no')}: {o['kind']:<15} {o['width_mm']:>5}mm {o['p1_mm']}->{o['p2_mm']} "
              f"lines={o['lines']} arc={o['arc']} pocket={o['pocket']}")


if __name__ == "__main__":
    main()
