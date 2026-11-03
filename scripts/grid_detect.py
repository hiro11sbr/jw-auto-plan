"""Stage 1 (v2): judge every 455 grid edge as wall / fitting / nothing.

usage: grid_detect.py <plan.png> <config.json> <out_prefix>
writes <out_prefix>_walls.json (same format as wall_detect.py), <out_prefix>_grid_check.png,
and <out_prefix>_edges.json (per-edge features, used for calibration / scoring)
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
import jp_text  # noqa: E402
import wall_detect as wd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- calibration
def calibrate(rgb, cfg):
    """Coarse scale from v1 detection, then separate x / y linear fits on confident walls."""
    mask, gray_range = wd.wall_mask(rgb, cfg)
    segs = wd.detect_direction(mask, cfg, True) + wd.detect_direction(mask, cfg, False)
    segs = wd.connect_junctions(wd.bridge_openings(segs, cfg), cfg)
    k, _ = wd.fit_scale(segs, cfg)
    # the per-column test closes holes itself, so it can use a wider gray window
    mid = (gray_range[0] + gray_range[1]) / 2
    w = cfg["grid_detection"]["gray_halfwidth"]
    gray_range = (int(mid - w), int(mid + w))
    grid = cfg["input"]["grid_mm"]
    fits = {}
    half = cfg["input"]["half_grid_mm"]
    tol = cfg["grid_detection"]["calib_tol_mm"]
    for horizontal in (False, True):
        lst = [s for s in segs if s["horizontal"] == horizontal]
        # the scale and grid phase that put the most wall length on 227.5 lines. (Anchoring on
        # the longest wall failed on プラン4 1F: its outer wall band sits ~60 mm off the grid,
        # which pushed every interior wall 4 px off and they were all rejected.)
        cs = np.array([s["c"] for s in lst], float)
        ws = np.array([s["b"] - s["a"] for s in lst], float)
        # walls the dimensions point at (a tick beside the wall band) are certain walls across
        # the whole building: they weigh more in the fit (プラン3 1F: the garage walls at the far
        # end did not fit the house walls' scale). The ticks themselves are not used as points:
        # outer wall bands are drawn a few px outward of the true centre the ticks mark, and
        # walls are judged on the bands (例1 lost 5% when the ticks were points)
        def search(wts):
            best = (-1.0, k, 0.0)
            for kk in np.linspace(k * 0.97, k * 1.03, 121):
                mm = cs * kk
                for ph in np.arange(0, half, 2.5):
                    r = np.abs((mm - ph + half / 2) % half - half / 2)
                    score = wts[r <= tol].sum()
                    if score > best[0]:
                        best = (score, kk, ph)
            return best[1], best[2]

        kk, ph = search(ws)
        ticks = DIM_TICKS.get(horizontal, [])
        if ticks:
            near = np.array([min(abs(c - t) for t in ticks) <= cfg["dimensions"]["tick_band_px"] for c in cs])
            wt = np.where(near, ws * cfg["dimensions"]["tick_weight"], ws)
            k2, ph2 = search(wt)
            # only a small correction (プラン3 1F: 0.6%); a big jump means the weighted walls were
            # not the building's (プラン3 2F: 3.7%, balcony lines)
            if abs(k2 - kk) <= cfg["dimensions"]["max_scale_change"] * kk:
                kk, ph, ws = k2, ph2, wt
        # which of the two 227.5 phases are the 455 lines: the one carrying more wall
        on455 = np.abs((cs * kk - ph + grid / 2) % grid - grid / 2) <= tol
        on_half = np.abs((cs * kk - ph - half + grid / 2) % grid - grid / 2) <= tol
        if ws[on_half].sum() > ws[on455].sum():
            ph += half
        pts = []
        for c_, w_ in zip(cs, ws):
            n = (c_ * kk - ph) / grid  # in 455 units; 227.5 lines are halves
            if abs(n * 2 - round(n * 2)) * half <= tol:
                pts.append((round(n * 2) / 2, c_, w_))
        n_, c_, w_ = map(np.array, zip(*pts))
        A = np.vstack([n_, np.ones_like(n_)]).T * np.sqrt(w_)[:, None]
        slope, icpt = np.linalg.lstsq(A, c_ * np.sqrt(w_), rcond=None)[0]
        fits[horizontal] = (slope, icpt, int(math.floor(n_.min())), int(math.ceil(n_.max())))
    return fits, gray_range


# ---------------------------------------------------------------- per-column state
def column_states(rgb, gray_range, cfg, centers, positions, horizontal):
    """State of each sample column across a band centred on `centers` (float px)."""
    g = cfg["grid_detection"]
    half = g["profile_half_px"]
    lo, hi = gray_range
    img = rgb.astype(int)
    out = []
    for c, p in zip(centers, positions):
        ci, pi = int(round(c)), int(round(p))
        if horizontal:
            prof = img[ci - half:ci + half + 1, pi]
        else:
            prof = img[pi, ci - half:ci + half + 1]
        if len(prof) != 2 * half + 1:
            out.append("none")
            continue
        r, gg, b = prof[:, 0], prof[:, 1], prof[:, 2]
        v = prof.mean(axis=1)
        neutral = (np.abs(r - gg) <= 10) & (np.abs(gg - b) <= 10)
        gray = neutral & (v >= lo) & (v <= hi)
        red = (r - gg > 60) & (r - b > 60)
        dark = neutral & (v < 100)
        edge_dark = v < g["outline_max_value"]  # JPEG blurs 1px outlines, so a looser test
        fill = gray | red
        # JPEG noise breaks the gray band; close 1-2px holes that are not outline (dark)
        for i in range(1, len(fill) - 2):
            if fill[i - 1] and not fill[i]:
                j = i
                while j < len(fill) and not fill[j] and j - i < 2:
                    j += 1
                if j < len(fill) and fill[j] and not (v[i:j] < 100).any():
                    fill[i:j] = True
        st = "none"
        # gray run covering the center
        tol = g["center_tol_px"]
        tol_ext = g["center_tol_ext_px"]
        idx = [i for i in range(half - tol_ext, half + tol_ext + 1) if fill[i]]
        idx.sort(key=lambda i: abs(i - half))
        if idx:
            i = idx[0]
            a = i
            while a > 0 and fill[a - 1]:
                a -= 1
            e = i
            while e < len(fill) - 1 and fill[e + 1]:
                e += 1
            t = e - a + 1
            mid = (a + e) / 2
            # a drawn wall has a black outline on both faces; shaded ground / eaves do not
            ow = g["outline_search_px"]
            outlined = ow == 0 or (edge_dark[max(0, a - ow):a].any() and edge_dark[e + 1:e + 1 + ow].any())
            omax = g.get("outline_max_px", 0)  # 試験中（10/30）：輪郭線の太さの上限。0 = 見ない
            if outlined and omax:
                def thin_line(start, step):
                    # the first dark run within ow px of the band edge, and its width
                    i = start
                    while 0 <= i < len(edge_dark) and abs(i - start) < ow and not edge_dark[i]:
                        i += step
                    n = 0
                    while 0 <= i < len(edge_dark) and edge_dark[i]:
                        n += 1
                        i += step
                    return 0 < n <= omax
                outlined = thin_line(a - 1, -1) and thin_line(e + 1, 1)
            if gray[a:e + 1].sum() >= 0.5 * t and outlined:
                tmax = cfg["wall_detection"]["thickness_px"][1]
                tmin = g["int_min_thickness_px"]  # thin partitions (e.g. toilet) are ~5px
                ext_min = cfg["wall_detection"]["exterior_min_thickness_px"]
                # exterior bands are drawn shifted outward by up to ~4px, so they get a wider tolerance
                if tmin <= t < ext_min and abs(mid - half) <= tol:
                    st = "int"
                elif ext_min <= t <= tmax and abs(mid - half) <= tol_ext:
                    st = "ext"
                elif t > tmax:
                    st = "blob"
        if st == "none":
            # 腰壁: a wall-wide band drawn as two outlines with white between (not filled)
            for k in range(g["koshi_half_min_px"], g["koshi_half_max_px"] + 1):
                if edge_dark[half - k - 1:half - k + 2].any() and edge_dark[half + k - 1:half + k + 2].any() \
                        and (v[half - k + 2:half + k - 1] > g["koshi_light_min"]).all():
                    st = "koshi"
                    break
        if st == "none":
            # fitting (door / window): >=2 thin dark lines inside the wall band, no fill at center
            band = dark[half - g["fitting_half_px"]:half + g["fitting_half_px"] + 1]
            lines = np.count_nonzero(np.diff(np.concatenate(([0], band.astype(int)))) == 1)
            on_text = False
            if TEXT_MASK is not None:  # lettering on the line is not a fitting (e.g. 土間収納)
                fh = g["fitting_half_px"]
                tm = TEXT_MASK[ci - fh:ci + fh + 1, pi] if horizontal else TEXT_MASK[pi, ci - fh:ci + fh + 1]
                on_text = bool(tm.any())
            if lines >= 2 and not fill[half] and not on_text:
                st = "fit"
        out.append(st)
    return out


TEXT_MASK = None  # pixels of lettering found by clean_text.py (set in main when given)
DIM_TICKS = {}  # dimension ticks (px) from <prefix>_寸法.json: {False: x ticks, True: y ticks}
DIM_EXTENT = None  # building extent from the dimensions (px): x0, y0, x1, y1
EXTENT_LINES = set()  # the grid lines at that extent: (horizontal, line index)
GENKAN_PTS = []  # 玄関 labels in grid-index coordinates (x index, y index), from <prefix>_部屋名.json


# ---------------------------------------------------------------- edges
def judge_edges(rgb, fits, gray_range, cfg):
    g = cfg["grid_detection"]
    grid = cfg["input"]["grid_mm"]
    (sx, ix, nx0, nx1), (sy, iy, ny0, ny1) = fits[False], fits[True]
    m = g["grid_margin"]
    xs = range(nx0 - m, nx1 + m + 1)
    ys = range(ny0 - m, ny1 + m + 1)
    H, W = rgb.shape[:2]

    def X(n):
        return ix + sx * n

    def Y(n):
        return iy + sy * n

    edges = []
    for horizontal in (True, False):
        lines = ys if horizontal else xs
        steps = xs if horizontal else ys
        # 455 lines plus the 227.5 lines between them (closets / storage use the 910/4 module)
        cand = list(lines)
        if g["half_lines"]:
            cand += [n + 0.5 for n in lines if n + 1 in lines]
        for ln in cand:
            for st in steps:
                if st + 1 not in steps:
                    continue
                c = Y(ln) if horizontal else X(ln)
                p0, p1 = (X(st), X(st + 1)) if horizontal else (Y(st), Y(st + 1))
                p0, p1 = sorted((p0, p1))
                pad = g["end_pad_px"]
                pos = np.arange(p0 + pad, p1 - pad + 1)
                if not (0 <= c < (H if horizontal else W)) or len(pos) == 0:
                    continue
                states = column_states(rgb, gray_range, cfg, [c] * len(pos), pos, horizontal)
                n = len(states)
                f = {s: states.count(s) / n for s in ("int", "ext", "fit", "koshi", "blob", "none")}
                nodes = (X(st) if horizontal else Y(st), X(st + 1) if horizontal else Y(st + 1))
                jambs = sum(wall_ends_here(rgb, gray_range, c, node, horizontal) for node in nodes)
                # gray right at each end node: a thin or short wall the edge judgement misses
                ng = [node_gray(rgb, gray_range, c, node, horizontal) for node in nodes]
                # share of the edge where lettering lies on the line (its fitting lines may be
                # hidden: プラン2 2F 洋室A, クローゼット written along the closet front)
                text = 0.0
                if TEXT_MASK is not None:
                    fh, ci = g["fitting_half_px"], int(round(c))
                    pi = np.clip(pos.astype(int), 0, (W if horizontal else H) - 1)
                    tm = TEXT_MASK[ci - fh:ci + fh + 1, :][:, pi] if horizontal else TEXT_MASK[pi, :][:, ci - fh:ci + fh + 1].T
                    text = float(tm.any(axis=0).mean()) if tm.size else 0.0
                f["text"] = round(text, 2)
                edges.append({"h": horizontal, "line": ln, "step": st, "f": f, "jambs": jambs,
                              "node_gray": ng, "through": through_lines(rgb, c, p0, p1, horizontal, cfg, gray_range)})
    return edges


def through_lines(rgb, c, p0, p1, horizontal, cfg, gray_range):
    """Positions (px along the line) of thin lines that cross a strip on BOTH sides of line c at
    the same place: stair treads running through a stair's centre line. A door frame or storage
    front interrupts the lines beside it (shelves and doors on one side only), a stair spine does
    not (プラン2 2F x=3640: 7 matched lines, plan 1 storage fronts: 0-1)."""
    g = cfg["grid_detection"]
    d0, d1 = g["stair_strip_px"]
    dark = g["stair_dark_max"]
    v = rgb.astype(int).mean(axis=2) if rgb.ndim == 3 else rgb
    H, W = v.shape
    ci = int(round(c))
    sides = []
    for s in (-1, 1):
        offs = [ci + s * d for d in range(d0, d1)]
        if min(offs) < 0 or max(offs) >= (H if horizontal else W):
            return []
        ts = np.arange(int(p0), int(p1) + 1)
        ts = ts[(ts >= 0) & (ts < (W if horizontal else H))]
        strip = v[np.ix_(offs, ts)] if horizontal else v[np.ix_(ts, offs)].T
        hit = list((strip < dark).mean(axis=0) >= 0.8) + [False]
        col = np.median(strip, axis=0)
        grey = (col >= gray_range[0]) & (col <= gray_range[1])
        pos, cur = [], None
        for i, hv in enumerate(hit):
            if hv and cur is None:
                cur = i
            elif not hv and cur is not None:
                # a thin line, not a wall band or hatching blob; and not the outline of a
                # crossing wall (wall gray right beside it)
                near = grey[max(0, cur - 6):cur].any() or grey[i:i + 6].any()
                if i - cur <= 3 and not near:
                    pos.append(float(ts[0] + (cur + i - 1) / 2))
                cur = None
        sides.append(pos)
    return [p for p in sides[0] if any(abs(p - q) <= 2 for q in sides[1])]


def mark_stairs(edges, cfg):
    """A fitting-looking run with many lines running straight through it is a stair's centre
    line (spine or the line between flights), not a wall or a fitting: drop it and keep later
    passes from reviving it (プラン2 2F got a sliding door in its stair, 2026-10-23)."""
    g = cfg["grid_detection"]
    rows = {}
    for e in edges:
        rows.setdefault((e["h"], e["line"]), {})[e["step"]] = e
    for row in rows.values():
        steps = sorted(row)
        i = 0
        while i < len(steps):
            if row[steps[i]]["cls"] not in ("fit", "koshi"):
                i += 1
                continue
            j = i
            while j + 1 < len(steps) and steps[j + 1] == steps[j] + 1 and row[steps[j + 1]]["cls"] in ("fit", "koshi"):
                j += 1
            run = [row[t] for t in steps[i:j + 1]]
            pos = sorted(p for e in run for p in e.get("through", []))
            lines = [p for k, p in enumerate(pos) if k == 0 or p - pos[k - 1] > 4]  # one per line
            if len(lines) >= g["stair_min_lines"]:
                for e in run:
                    e["cls"] = "none"
                    e["stair"] = True
            i = j + 1


def node_gray(rgb, gray_range, c, node, horizontal, r=6):
    """Fraction of wall gray in a small box around the grid node (c on this line, node along it)."""
    lo, hi = gray_range
    ci, ni = int(round(c)), int(round(node))
    box_ = rgb[ci - r:ci + r + 1, ni - r:ni + r + 1] if horizontal else rgb[ni - r:ni + r + 1, ci - r:ci + r + 1]
    if box_.size == 0:
        return 0.0
    v = box_.astype(int).mean(axis=2)
    return round(float(((v >= lo) & (v <= hi)).mean()), 2)


def wall_ends_here(rgb, gray_range, c, node, horizontal):
    """True when a perpendicular wall band at `node` stops at line `c` (gray on one side only).

    A door / closet front sits exactly where the side walls end (their jambs); a hanger pipe
    or shelf line inside a closet does not, which tells them apart.
    """
    lo, hi = gray_range
    ci, ni = int(round(c)), int(round(node))
    r = 10  # the jamb end may sit a few px off the grid line (frame lines, JPEG)
    if horizontal:
        a = rgb[ci - r:ci - 2, ni - 2:ni + 3]
        b = rgb[ci + 3:ci + r + 1, ni - 2:ni + 3]
    else:
        a = rgb[ni - 2:ni + 3, ci - r:ci - 2]
        b = rgb[ni - 2:ni + 3, ci + 3:ci + r + 1]
    if a.size == 0 or b.size == 0:
        return False

    def gray_frac(p):
        v = p.astype(int).mean(axis=2)
        return float(((v >= lo) & (v <= hi)).mean())

    ga, gb = gray_frac(a), gray_frac(b)
    return (ga >= 0.5 and gb <= 0.25) or (gb >= 0.5 and ga <= 0.25)


def perp_steps(ln):
    """Steps of perpendicular edges that touch a line at index ln (a 227.5 line sits mid-edge)."""
    return [ln - 1, ln] if float(ln).is_integer() else [math.floor(ln)]


def resolve_half_lines(edges):
    """A wall on a 455 line and one on the neighbouring 227.5 line at the same step are the same
    wall seen twice (wide door symbols reach both): keep the one with more solid gray band."""
    # whole runs are compared, so a closet front wins over the pipe line along its full length
    runs = {}
    for e in edges:
        if e["cls"] != "none":
            runs.setdefault((e["h"], e["line"]), []).append(e)

    def score(run):
        solid = sum(r["f"]["int"] + r["f"]["ext"] for r in run)
        jambs = sum(r["jambs"] for r in run)
        # real wall gray on a good part of the run wins first: a window frame drawn wider than
        # the wall reaches the next 227.5 line with jambs but no gray (プラン2 2F, y=455/682.5)
        return (solid >= 0.3 * len(run), jambs, solid)

    for (h, ln), run in list(runs.items()):
        if float(ln).is_integer():
            continue
        for other_ln in (ln - 0.5, ln + 0.5):
            other = runs.get((h, other_ln))
            if not other:
                continue
            steps = {r["step"] for r in run} & {r["step"] for r in other}
            if not steps:
                continue
            a = [r for r in run if r["step"] in steps]
            b = [r for r in other if r["step"] in steps]
            loser = b if score(a) > score(b) else a  # ties go to the 455 line
            for r in loser:
                r["cls"] = "none"


def classify(edges, cfg):
    g = cfg["grid_detection"]
    for e in edges:
        f = e["f"]
        wall = f["int"] + f["ext"]
        if wall >= g["wall_min_frac"]:
            e["cls"] = "wall"
            e["ext"] = f["ext"] > f["int"]
        elif f.get("koshi", 0) >= g["koshi_min_frac"]:
            e["cls"] = "koshi"  # 腰壁: not a wall, but a fitting may end on it
        elif f["fit"] + wall >= g["fitting_min_frac"]:
            e["cls"] = "fit"
        else:
            e["cls"] = "none"
    resolve_half_lines(edges)
    mark_stairs(edges, cfg)
    # fittings count as wall only when both ends meet a wall (same line, or a crossing wall at the node)
    lines = {}
    for e in edges:
        lines.setdefault((e["h"], e["line"]), {})[e["step"]] = e
    maxrun = g["fitting_max_edges"]
    wall_set = {(e["h"], e["line"], e["step"]) for e in edges if e["cls"] == "wall"}
    # a single blank edge between fitting edges is part of the same fitting (e.g. the middle of
    # a closet's folding doors, drawn with no line inside the band)
    for row in lines.values():
        for st, e in row.items():
            if e["cls"] == "none" and not e.get("stair") and row.get(st - 1, {}).get("cls") == "fit" \
                    and row.get(st + 1, {}).get("cls") == "fit":
                e["cls"] = "fit"

    def crossing_wall(h, ln, node):
        # walls on the perpendicular grid line `node`, touching grid point (node, ln)
        return any((not h, node, s) in wall_set for s in perp_steps(ln))

    for (h, ln), row in lines.items():
        # a weak fitting edge next to a fitting, whose other end meets a wall (in line or
        # crossing), is part of that fitting: a storage front whose frame lines are partly
        # hidden by its label (e.g. 06823) or JPEG
        def growable(e, first):
            if not e or e["cls"] != "none" or e.get("stair"):
                return False
            weak = e["f"]["fit"] + e["f"].get("koshi", 0) >= g["fitting_weak_frac"]
            # or lettering over the line hides the fitting lines (they are not counted under text)
            # — but some fitting line must still show: text alone grew walls on plan 1
            hidden = e["f"]["fit"] + e["f"].get("text", 0) >= g["fitting_text_frac"] and e["f"].get("text", 0) > 0 \
                and e["f"]["fit"] >= g["fitting_text_min_fit"]
            # beyond one edge only under lettering: two weak edges in a row grew shelf lines
            # into walls on plan 1 (X=1820, Y=1820, X=4095)
            return (weak or hidden) if first else hidden

        for st, nb in list(row.items()):
            # the fitting grown from must be a clear one: stair lines next to a stair spine are
            # only half-fitting-like (2F, 0.44-0.61) and must not grow into a wall
            if nb["cls"] != "fit" or nb["f"]["fit"] < g["fitting_strong_frac"]:
                continue
            for d in (1, -1):
                path = []
                t = st + d
                while len(path) < g["fitting_grow_max_edges"] and growable(row.get(t), not path):
                    path.append(t)
                    node = t + 1 if d == 1 else t  # grid node at the far end of edge t
                    if row.get(t + d, {}).get("cls") in ("wall", "koshi") or crossing_wall(h, ln, node):
                        for k in path:
                            row[k]["cls"] = "fit"
                        break
                    t += d
        # a 腰壁-looking run closed off by walls at BOTH ends is a storage front (its door frame is
        # drawn as two outlines a wall's width apart); a real 腰壁 has a free end
        steps = sorted(row) if g.get("koshi_storage_front", True) else []
        i = 0
        while i < len(steps):
            if row[steps[i]]["cls"] != "koshi":
                i += 1
                continue
            j = i
            while j + 1 < len(steps) and row[steps[j + 1]]["cls"] == "koshi" and steps[j + 1] == steps[j] + 1:
                j += 1
            a, b = steps[i], steps[j] + 1
            left, right = row.get(a - 1, {}).get("cls"), row.get(b, {}).get("cls")
            if (left == "wall" or crossing_wall(h, ln, a)) and (right == "wall" or crossing_wall(h, ln, b)) \
                    and j - i + 1 <= g["fitting_max_edges"]:
                for t in steps[i:j + 1]:
                    row[t]["cls"] = "fit"
            i = j + 1

    solid = ("wall", "wall_fit")
    # pass 1: fittings between solid walls; pass 2: also fittings that end on another fitting
    # counted as wall in pass 1 (e.g. a storage front ending on a sliding-door pocket)
    for last in (False, True):
        for (h, ln), row in lines.items():
            steps = sorted(row)
            i = 0
            while i < len(steps):
                e = row[steps[i]]
                if e["cls"] != "fit":
                    i += 1
                    continue
                j = i
                while j + 1 < len(steps) and row[steps[j + 1]]["cls"] == "fit" and steps[j + 1] == steps[j] + 1:
                    j += 1
                    # a crossing wall inside the run ends it: shelf lines beside a storage front
                    # must not join the front into one run with no wall at its ends
                    if crossing_wall(h, ln, steps[j]):
                        j -= 1
                        break
                left = row.get(steps[i] - 1)
                right = row.get(steps[j] + 1)
                ng_min = g["fitting_end_node_gray"]
                support = solid + ("koshi",)  # a fitting may also end on a 腰壁 (stair-side storage)
                left_ok = ((left and left["cls"] in support) or crossing_wall(h, ln, steps[i])
                           or row[steps[i]]["node_gray"][0] >= ng_min)
                right_ok = ((right and right["cls"] in support) or crossing_wall(h, ln, steps[j] + 1)
                            or row[steps[j]]["node_gray"][1] >= ng_min)
                ok = left_ok and right_ok and j - i + 1 <= maxrun
                ext = bool(left and left["cls"] == "wall" and left["ext"] and right and right["cls"] == "wall"
                           and right["ext"])
                for t in steps[i:j + 1]:
                    if ok:
                        row[t]["cls"] = "wall_fit"
                        row[t]["ext"] = ext
                    elif last:
                        row[t]["cls"] = "none"
                        row[t]["was_fit"] = True
                i = j + 1
        wall_set = {(e["h"], e["line"], e["step"]) for e in edges if e["cls"] in solid}
    complete_dangling(edges, cfg)
    refit_after_completion(lines, crossing_wall_of(edges), maxrun, crossing_wall_of(edges, completed_only=True))
    drop_furniture_runs(edges, cfg)
    # isolated single wall edges with no neighbour (in line or crossing) are noise
    walls = {(e["h"], e["line"], e["step"]) for e in edges if e["cls"] in ("wall", "wall_fit")}
    for e in edges:
        if e["cls"] != "wall":
            continue
        h, ln, st = e["h"], e["line"], e["step"]
        nb = {(h, ln, st - 1), (h, ln, st + 1)}
        nb |= {(not h, node, s) for node in (st, st + 1) for s in perp_steps(ln)}
        if not nb & walls:
            e["cls"] = "none"
    if g.get("outline_first", True):
        settle_outline(edges, cfg)
    return edges


def drop_furniture_runs(edges, cfg):
    """A run on one line made of at most one edge of real wall gray plus "fitting" edges, with a
    free end (no wall in line, none crossing), is furniture: the gray box of an appliance in a
    cupboard let the cupboard's lines pass as a fitting ending on it (プラン3 1F, user 10/24).
    A real fitting stands between two walls."""
    g = cfg["grid_detection"]
    solid = ("wall", "wall_fit")
    by_key = {(e["h"], e["line"], e["step"]): e for e in edges}
    walls = {k for k, e in by_key.items() if e["cls"] in solid}
    rows = {}
    for (h, ln, st), e in by_key.items():
        rows.setdefault((h, ln), []).append(st)

    def crossing(h, ln, node):
        return any((not h, node, s) in walls for s in perp_steps(ln))

    for (h, ln), steps in rows.items():
        steps = sorted(s for s in steps if (h, ln, s) in walls)
        i = 0
        while i < len(steps):
            j = i
            while j + 1 < len(steps) and steps[j + 1] == steps[j] + 1:
                j += 1
            run = [by_key[(h, ln, s)] for s in steps[i:j + 1]]
            n_wall = sum(e["cls"] == "wall" for e in run)
            n_fit = len(run) - n_wall
            a, b = steps[i], steps[j] + 1  # end nodes
            free = (not crossing(h, ln, a)) or (not crossing(h, ln, b))
            if n_wall <= g["furniture_max_wall_edges"] and n_fit >= 1 and free and not any(
                    e.get("ext") for e in run):
                for e in run:
                    e["cls"] = "none"
                    e["furniture"] = True
            i = j + 1


def classify_staged(edges, cfg):
    """Walls stage by stage (user 2026-10-24): most walls, and nearly the whole outline, sit on
    the 910 grid, so the 910 lines are judged first, then the 455 lines, and only then the 227.5
    lines, to pick up what was missed. Judging every 227.5 line at once let a wrong wall win
    over the right one. A wall settled in an earlier stage is never taken away by a later one,
    and a 227.5 wall is not taken beside a wall already settled on the 455 line next to it."""
    solid = ("wall", "wall_fit")
    g = cfg["grid_detection"]

    def is_int(e):
        return float(e["line"]).is_integer()

    # the 910 lines: per direction, the parity of 455 lines carrying more wall band
    parity = {}
    for h in (True, False):
        score = {0: 0.0, 1: 0.0}
        for e in edges:
            if e["h"] == h and is_int(e):
                score[int(e["line"]) % 2] += e["f"]["int"] + e["f"]["ext"]
        parity[h] = max(score, key=score.get)
    stages = [
        [e for e in edges if is_int(e) and int(e["line"]) % 2 == parity[e["h"]]],
        [e for e in edges if is_int(e)],
        list(edges),
    ]
    settled = {}  # id -> (cls, ext) of walls settled in an earlier stage
    for k, part in enumerate(stages):
        for e in part:
            if id(e) not in settled:
                e["cls"] = "none"
                e.pop("completed", None)
        classify(part, cfg)
        for e in part:
            if id(e) in settled and e["cls"] not in solid:
                e["cls"], e["ext"] = settled[id(e)]  # never taken away by a later stage
        if k == 2:
            # 227.5: not beside a wall settled on the 455 line at that step
            on = {(e["h"], e["line"], e["step"]) for e in edges if is_int(e) and id(e) in settled}
            for e in edges:
                if not is_int(e) and e["cls"] in solid and (
                        (e["h"], e["line"] - 0.5, e["step"]) in on or (e["h"], e["line"] + 0.5, e["step"]) in on):
                    e["cls"] = "none"
        for e in part:
            # settled: everything from the 910 stage, and real (gray) walls from the 455 stage.
            # A fitting-like run on a 455 line is left open to the 227.5 stage, where it is
            # compared with the line beside it on its evidence: a closet's hanger pipe (例1 2F)
            # or a line in a kitchen counter (プラン3 1F) must not beat the closet front
            # (a fitting between exterior walls, a window in the outline, is settled too: プラン4
            # 2F's window lost to a stair's side line on the 227.5 line beside it)
            if e["cls"] == "wall" or (k == 0 and e["cls"] in solid) or (e["cls"] == "wall_fit" and e.get("ext")):
                settled[id(e)] = (e["cls"], e.get("ext", False))
        print(f"  段階{k + 1}（{['910', '455', '227.5'][k]}）: 壁 {sum(e['cls'] == 'wall' for e in part)} "
              f"建具 {sum(e['cls'] == 'wall_fit' for e in part)}")
    # a wall settled early but left with no neighbour at the end is noise after all (プラン3 1F:
    # one 455 edge of a kitchen counter's gray, kept by the settle rule)
    walls = {(e["h"], e["line"], e["step"]) for e in edges if e["cls"] in solid}
    for e in edges:
        if e["cls"] != "wall":
            continue
        h, ln, st = e["h"], e["line"], e["step"]
        nb = {(h, ln, st - 1), (h, ln, st + 1)} | {(not h, node, s) for node in (st, st + 1) for s in perp_steps(ln)}
        if not nb & walls:
            e["cls"] = "none"
    return edges


def edge_line(e):
    """The edge as a segment in grid-index coordinates (x index, y index)."""
    h, ln, st = e["h"], e["line"], e["step"]
    return ((st, ln), (st + 1, ln)) if h else ((ln, st), (ln, st + 1))


def settle_outline(edges, cfg):
    """Settle the building outline before trusting anything else (user, 2026-10-23):
    1. close gaps between exterior walls on the same line where something is drawn (windows,
       doors), so the outline is one loop;
    2. the outline = the largest closed loop of walls; every wall on it is exterior;
    3. walls outside it are dropped, except exterior walls running on from it (porch walls);
    4. an exterior-width band inside it that touches no other wall is not a wall (a sofa)."""
    from shapely.geometry import LineString, Point
    from shapely.ops import polygonize, unary_union

    g = cfg["grid_detection"]
    solid = ("wall", "wall_fit")
    rows = {}
    for e in edges:
        rows.setdefault((e["h"], e["line"]), {})[e["step"]] = e

    def evidence(e):
        f = e["f"]
        return f["int"] + f["ext"] + f["fit"] + f.get("koshi", 0)

    # 1. gaps between exterior walls on one line
    for row in rows.values():
        ext_steps = sorted(st for st, e in row.items() if e["cls"] in solid and e.get("ext"))
        for a, b in zip(ext_steps, ext_steps[1:]):
            gap = [row.get(s) for s in range(a + 1, b)]
            if not gap or len(gap) > g["outline_gap_max_edges"] or any(x is None for x in gap):
                continue
            # only a plain hole (window / door): a gap with walls in it is not one opening, and
            # bridging it once joined a wall to a sofa back (プラン2 1F, y=1820)
            if any(x["cls"] == "wall" for x in gap) or all(x["cls"] in solid for x in gap):
                continue
            if sum(evidence(x) for x in gap) / len(gap) >= g["complete_min_frac"]:
                for x in gap:
                    if x["cls"] not in solid:
                        x["cls"], x["completed"] = "wall_fit", True
                    x["ext"] = True
    # 1b. the outermost lines of the dimensions: the building's outer walls stand there, so a
    # gap between walls on such a line is an opening (a garage's open shutter side; user 10/24:
    # 寸法があるのに外壁が無いことは無い)
    for (h, ln) in EXTENT_LINES:
        row = rows.get((h, ln), {})
        sts = sorted(st for st, e in row.items() if e["cls"] in solid)
        if len(sts) >= 2:
            for st in range(sts[0], sts[-1] + 1):
                x = row.get(st)
                if x is not None and x["cls"] not in solid and not x.get("stair"):
                    x["cls"], x["completed"], x["ext"] = "wall_fit", True, True
    # 2. the outline
    segs = [LineString(edge_line(e)) for e in edges if e["cls"] in solid]
    polys = list(polygonize(unary_union(segs))) if segs else []
    if not polys:
        return
    fp = unary_union(polys)
    if fp.geom_type == "MultiPolygon":
        fp = max(fp.geoms, key=lambda p: p.area)
    from shapely.geometry import Polygon
    fp = Polygon(fp.exterior)
    # an outline broken at a gap (プラン4 1F: the 玄関/porch side did not close, so only the
    # bottom block was kept and every wall above it was dropped): close gaps up to
    # outline_close_edges grid units and take that shape when it is clearly larger
    R = g["outline_close_edges"] / 2
    closed = unary_union(segs).buffer(R, join_style=2)
    if closed.geom_type == "MultiPolygon":
        closed = max(closed.geoms, key=lambda p: p.area)
    closed = Polygon(closed.exterior).buffer(-R, join_style=2)
    if closed.geom_type == "MultiPolygon":
        closed = max(closed.geoms, key=lambda p: p.area)
    if not closed.is_empty and fp.area < g["outline_close_ratio"] * closed.area:
        # back on the grid (buffering leaves float noise, and walls must lie exactly on the ring)
        fp = Polygon([(round(x * 2) / 2, round(y * 2) / 2) for x, y in closed.exterior.coords]).buffer(0)
    # the 玄関 is always inside the building (user 10/24: プラン3 1F left the 玄関 and 土間収納
    # outside, beside the garage). A closed area holding a 玄関 label next to the outline joins it.
    # (A garage is outside space: its walls stay outside the outline.)
    for gp in GENKAN_PTS:
        pt = Point(gp)
        if fp.contains(pt):
            continue
        face = next((p for p in polys if p.contains(pt)), None)
        if face is not None and face.distance(fp) < 0.01:
            joined = unary_union([fp, face])
            if joined.geom_type == "Polygon":
                fp = Polygon(joined.exterior)
    # (a garage stays inside the building outline: the dimensions run round it, user 10/24. It
    # is outside space within it; make_plan draws its walls as such)
    ring = fp.exterior
    inside = fp.buffer(-0.05)
    on_ring = lambda e: ring.buffer(0.01).contains(LineString(edge_line(e)))  # noqa: E731
    for e in edges:
        if e["cls"] in solid and on_ring(e):
            e["ext"] = True
    # 3. outside the outline: only exterior walls continuing a ring line (porch walls)
    ring_nodes = {tuple(map(float, c)) for c in ring.coords}
    changed = True
    keep = set()
    while changed:
        changed = False
        for e in edges:
            if e["cls"] not in solid or id(e) in keep:
                continue
            (p, q) = edge_line(e)
            mid = Point((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            if fp.buffer(0.01).contains(mid):
                continue
            if not e.get("ext"):
                continue
            ends = {tuple(map(float, p)), tuple(map(float, q))}
            touch = any(ring.distance(Point(x)) < 0.01 for x in ends) or any(
                k in ends for k in kept_nodes(edges, keep))
            if touch:
                keep.add(id(e))
                changed = True
    for e in edges:
        if e["cls"] not in solid:
            continue
        (p, q) = edge_line(e)
        mid = Point((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        if not fp.buffer(0.01).contains(mid) and id(e) not in keep:
            e["cls"], e["outside"] = "none", True
    # 3a. "exterior" means on the outline (or a porch wall running on from it), never a thick
    #     band inside: windows and the entrance door are only judged on exterior walls, so a
    #     window can only sit on the outline (user, 2026-10-23)
    for e in edges:
        if e["cls"] in solid:
            e["ext"] = on_ring(e) or id(e) in keep
    # 3b. walls form one network hanging off the outline; a small island of walls standing alone
    #     inside a room (sofa: a thick back plus a thin side, プラン2 1F) is furniture
    live = [e for e in edges if e["cls"] in solid]
    parent = list(range(len(live)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, e in enumerate(live):
        for n in edge_line(e):
            n = tuple(map(float, n))
            if n in owner:
                parent[root(i)] = root(owner[n])
            else:
                owner[n] = i
    comps = {}
    for i, e in enumerate(live):
        comps.setdefault(root(i), []).append(e)
    for members in comps.values():
        on_outline = any(ring.buffer(0.01).contains(LineString(edge_line(e))) for e in members)
        if not on_outline and len(members) <= g["island_max_edges"]:
            for e in members:
                e["cls"], e["island"] = "none", True
    # 4. exterior-width pieces inside the outline that, together, touch neither the outline nor
    #    any interior wall (a sofa drawn as an L of thick bands, プラン2 1F)
    inner_ext = [e for e in edges if e["cls"] in solid and e.get("ext")
                 and inside.contains(LineString(edge_line(e)))]
    others = [e for e in edges if e["cls"] in solid and e not in inner_ext]
    other_nodes = set()
    for e in others:
        p, q = edge_line(e)
        other_nodes |= {tuple(map(float, p)), tuple(map(float, q))}
    comp = {id(e): i for i, e in enumerate(inner_ext)}

    def find(i):
        while comp_parent[i] != i:
            comp_parent[i] = comp_parent[comp_parent[i]]
            i = comp_parent[i]
        return i

    comp_parent = list(range(len(inner_ext)))
    node_owner = {}
    for i, e in enumerate(inner_ext):
        for n in edge_line(e):
            n = tuple(map(float, n))
            if n in node_owner:
                comp_parent[find(i)] = find(node_owner[n])
            else:
                node_owner[n] = i
    groups = {}
    for i, e in enumerate(inner_ext):
        groups.setdefault(find(i), []).append(e)
    for members in groups.values():
        nodes = {tuple(map(float, n)) for e in members for n in edge_line(e)}
        touches = bool(nodes & other_nodes) or any(ring.distance(Point(n)) < 0.01 for n in nodes)
        if not touches:
            for e in members:
                e["cls"], e["lone_ext"] = "none", True
    walls = {(e["h"], e["line"], e["step"]) for e in edges if e["cls"] in solid}
    for (h, ln), row in rows.items():
        steps = sorted(st for st, e in row.items() if e["cls"] in solid and e.get("ext")
                       and inside.contains(LineString(edge_line(e))))
        runs_, cur = [], []
        for st in steps:
            if cur and st != cur[-1] + 1:
                runs_.append(cur)
                cur = []
            cur.append(st)
        if cur:
            runs_.append(cur)
        for run in runs_:
            a, b = run[0], run[-1] + 1
            touches = ((h, ln, a - 1) in walls or (h, ln, b) in walls
                       or any((not h, node, s) in walls for node in (a, b) for s in perp_steps(ln)))
            if not touches:
                for st in run:
                    row[st]["cls"], row[st]["lone_ext"] = "none", True


def kept_nodes(edges, keep):
    out = set()
    for e in edges:
        if id(e) in keep:
            p, q = edge_line(e)
            out |= {tuple(map(float, p)), tuple(map(float, q))}
    return out


def crossing_wall_of(edges, completed_only=False):
    """crossing(h, ln, node): a wall on the perpendicular grid line `node` touches (node, ln)
    (only one complete_dangling ran through, with completed_only)."""
    solid = {(e["h"], e["line"], e["step"]) for e in edges if e["cls"] in ("wall", "wall_fit")
             and (e.get("completed") or not completed_only)}
    return lambda h, ln, node: any((not h, node, s) in solid for s in perp_steps(ln))  # noqa: E731


def refit_after_completion(lines, crossing, maxrun, new_crossing=None):
    """A run of fitting evidence turned down for want of a wall at an end gets a second look
    once the gaps are completed: the wall at its end may only now be there (プラン5 2F: the
    window, pier and door between the hall and the 青空リビング, y=4550 910〜2730, ended on the
    x=2730 wall that complete_dangling ran through, so the hall fell outside the outline, 10/27)."""
    for (h, ln), row in lines.items():
        steps = sorted(row)
        i = 0
        while i < len(steps):
            if not row[steps[i]].get("was_fit") or row[steps[i]]["cls"] != "none":
                i += 1
                continue
            j = i
            while j + 1 < len(steps) and row[steps[j + 1]].get("was_fit") and row[steps[j + 1]]["cls"] == "none" \
                    and steps[j + 1] == steps[j] + 1:
                j += 1
            left, right = row.get(steps[i] - 1), row.get(steps[j] + 1)
            left_ok = (left and left["cls"] in ("wall", "wall_fit")) or crossing(h, ln, steps[i])
            right_ok = (right and right["cls"] in ("wall", "wall_fit")) or crossing(h, ln, steps[j] + 1)
            # only where an end is held by a wall that the completion just ran through (a run
            # turned down for another reason stays down: プラン2 1F's 框 lines)
            new_end = new_crossing is None or new_crossing(h, ln, steps[i]) or new_crossing(h, ln, steps[j] + 1)
            # and only running on from an exterior wall in line (the outline's windows and doors;
            # a 框's two lines inside the house are not walls: プラン2 1F y=2730, プラン3 1F y=8645)
            ext_in_line = any(n is not None and n["cls"] == "wall" and n.get("ext") for n in (left, right))
            if left_ok and right_ok and new_end and ext_in_line and j - i + 1 <= maxrun:
                ext = bool(left and left.get("ext") or right and right.get("ext"))
                for t in steps[i:j + 1]:
                    row[t]["cls"], row[t]["ext"], row[t]["completed"] = "wall_fit", ext, True
            i = j + 1


def complete_dangling(edges, cfg):
    """Run a wall straight through a short gap (sliding door, thin wall, text over the band).

    Only along the same grid line: from a wall end, continue straight until another wall on the
    line, or a crossing wall. Never turns a corner (that turned shelf lines into walls).
    """
    g = cfg["grid_detection"]
    weak, maxlen = g["complete_min_frac"], g["complete_max_edges"]
    by_key = {(e["h"], e["line"], e["step"]): e for e in edges}
    is_wall = lambda e: e is not None and e["cls"] in ("wall", "wall_fit")  # noqa: E731
    solid = {k for k, e in by_key.items() if is_wall(e)}
    real = {k for k, e in by_key.items() if e["cls"] == "wall"}

    def crossing(h, ln, node, among=None):
        return any((not h, node, s) in (among or solid) for s in perp_steps(ln))

    def evidence(e):
        f = e["f"]
        return f["int"] + f["ext"] + f["fit"]

    filled = []
    for (h, ln, st), e in by_key.items():
        if (h, ln, st) not in solid:
            continue
        for d in (1, -1):
            nxt = st + d
            if (h, ln, nxt) in solid:
                continue  # not an end in this direction
            node = st + 1 if d == 1 else st  # grid node at this end
            if crossing(h, ln, node, real):
                continue  # already meets a crossing wall (T / L)
            if crossing(h, ln, node):
                # only a fitting (e.g. a storage front) crosses here: the wall may run on past
                # it, but only where the next edge shows wall band itself
                n_e = by_key.get((h, ln, nxt))
                if not n_e or n_e["f"]["int"] + n_e["f"]["ext"] < weak:
                    continue
            # an opening (door with no lines in the band) may have zero evidence itself,
            # so the path is judged on its average, and must end on a wall
            path, cur = [], nxt
            while len(path) < maxlen:
                k = (h, ln, cur)
                if k not in by_key:
                    path = None
                    break
                path.append(k)
                far = cur + 1 if d == 1 else cur
                if (h, ln, cur + d) in solid or crossing(h, ln, far):
                    break
                cur += d
            else:
                path = None
            if not path or any(by_key[k].get("stair") for k in path) \
                    or sum(evidence(by_key[k]) for k in path) / len(path) < weak:
                continue
            # one bare opening (a door, <= 455) is fine; two in a row is open floor, not a wall
            zeros = longest = 0
            for k in path:
                zeros = zeros + 1 if evidence(by_key[k]) < 0.05 else 0
                longest = max(longest, zeros)
            if longest <= g["complete_max_bare_edges"]:
                filled += path
    for k in filled:
        e = by_key[k]
        if e["cls"] not in ("wall", "wall_fit"):
            e["cls"] = "wall_fit"
            e["completed"] = True
            e["ext"] = e["f"]["ext"] > e["f"]["int"]


def edges_to_walls(edges, fits, cfg):
    grid = cfg["input"]["grid_mm"]
    nx0 = fits[False][2]
    ny0 = fits[True][2]
    lines = {}
    for e in edges:
        if e["cls"] in ("wall", "wall_fit"):
            lines.setdefault((e["h"], e["line"]), []).append(e)
    walls = []
    for (h, ln), lst in lines.items():
        lst.sort(key=lambda e: e["step"])
        run = [lst[0]]
        for e in lst[1:] + [None]:
            # a run is split where it leaves the outline: its outline part is exterior, the
            # part running on inside is interior (windows are judged only on exterior walls)
            if e is not None and e["step"] == run[-1]["step"] + 1 and \
                    bool(e.get("ext")) == bool(run[-1].get("ext")):
                run.append(e)
                continue
            a, b = run[0]["step"], run[-1]["step"] + 1
            ext = sum(r["ext"] for r in run if "ext" in r) * 2 >= len(run)
            if h:
                # image y grows downward: the largest y index is the bottom (Y=0)
                yv = (fits[True][3] - ln) * grid
                p1, p2 = ((a - nx0) * grid, yv), ((b - nx0) * grid, yv)
            else:
                xv = (ln - nx0) * grid
                p1, p2 = (xv, (fits[True][3] - b) * grid), (xv, (fits[True][3] - a) * grid)
            walls.append({"type": "exterior" if ext else "interior", "p1": p1, "p2": p2,
                          "fit_edges": sum(r["cls"] == "wall_fit" for r in run), "flags": []})
            run = [e] if e is not None else []
    trim_to_half_lines(walls, grid / 2)
    # inside the building no wall (or fitting counted as wall) is shorter than 455 (user rule):
    # shorter interior pieces are furniture / joinery edges or trimming leftovers
    min_len = cfg["grid_detection"]["interior_min_length_mm"]
    kept = []
    for w in walls:
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        length = abs(x2 - x1) + abs(y2 - y1)
        if w["type"] == "interior" and length < min_len - 1:
            continue
        kept.append(w)
    return kept


def retype_inner_walls(walls, cfg):
    """A wall flagged exterior (its band looked thick) that stands well inside the building
    outline is an interior wall: its openings are doors and 建具, not windows or a 玄関ドア
    (プラン5 2F: the wall between the hall and the toilet, x=2730, had a long run flagged
    exterior and the toilet's sliding door was read as a 玄関ドア; 10/27). The outline is the
    largest face the walls close; a wall within `inner_wall_margin_mm` of it keeps its type."""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union
    margin = cfg["grid_detection"]["inner_wall_margin_mm"]
    lines = [LineString([w["p1"], w["p2"]]) for w in walls]
    faces = list(polygonize(unary_union(lines))) if lines else []
    if not faces:
        return
    ring = Polygon(max(faces, key=lambda f: f.area).exterior)
    for f in faces:  # (the building may be several faces: take the union's outer boundary)
        ring = Polygon(unary_union([ring, f]).exterior) if f.intersects(ring) else ring
    for w in walls:
        if w["type"] != "exterior":
            continue
        seg = LineString([w["p1"], w["p2"]])
        if cfg["grid_detection"].get("whole_inner_retype", False) and seg.distance(ring.exterior) > margin:
            w["type"] = "interior"
            print(f"  外壁の印の壁は外形の内側 → 内壁: {w['p1']}〜{w['p2']}")
        # (a wall partly on the ring keeps its type: the part inside is judged by opening_detect)
    split_inner_runs(walls, ring, cfg)


def split_inner_runs(walls, ring, cfg):
    """An exterior-marked wall that runs on the outline for one stretch and then continues
    inside it (between two rooms) is split at the crossing walls; the stretches inside become
    interior walls (プラン5 2F: x=2730, 青空リビング側は外壁、便所とホールの間は内壁; 10/27).
    Only stretches of at least `split_inner_min_mm` and further than `inner_wall_margin_mm`."""
    from shapely.geometry import LineString, Point
    margin = cfg["grid_detection"]["inner_wall_margin_mm"]
    min_len = cfg["grid_detection"].get("split_inner_min_mm", 900)
    for w in list(walls):
        if w["type"] != "exterior":
            continue
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        h = abs(y1 - y2) < 1
        a, b = (min(x1, x2), max(x1, x2)) if h else (min(y1, y2), max(y1, y2))
        c = y1 if h else x1
        cuts = {a, b}
        for o in walls:
            if o is w:
                continue
            (ox1, oy1), (ox2, oy2) = o["p1"], o["p2"]
            if (abs(oy1 - oy2) < 1) == h:
                continue
            pos, lo, hi = (ox1, min(oy1, oy2), max(oy1, oy2)) if h else (oy1, min(ox1, ox2), max(ox1, ox2))
            if a < pos < b and lo - 1 <= c <= hi + 1:
                cuts.add(pos)
        cuts = sorted(cuts)
        if len(cuts) < 3:
            continue
        pieces = []
        for s0, s1 in zip(cuts, cuts[1:]):
            p = ([s0, c], [s1, c]) if h else ([c, s0], [c, s1])
            inner = ring.contains(LineString(p)) and LineString(p).distance(ring.exterior) > margin
            pieces.append([s0, s1, p, inner])
        merged = []
        for pc in pieces:  # join neighbouring stretches of the same kind
            if merged and merged[-1][3] == pc[3]:
                merged[-1][1] = pc[1]
                merged[-1][2] = (merged[-1][2][0], pc[2][1])
            else:
                merged.append(pc)
        if len(merged) < 2 or not any(m[3] and m[1] - m[0] >= min_len for m in merged):
            continue
        idx = walls.index(w)
        walls.remove(w)
        for k, (s0, s1, p, inner) in enumerate(merged):
            nw = dict(w, p1=list(p[0]), p2=list(p[1]))
            if inner and s1 - s0 >= min_len:
                nw["type"] = "interior"
            walls.insert(idx + k, nw)
        print(f"  外壁の印の壁の途中が内側 → 分けて内壁: {(x1, y1)}〜{(x2, y2)}")


def trim_to_half_lines(walls, half):
    """A wall whose end edge is crossed mid-way by a wall on a 227.5 line, and which meets
    nothing at its own end, really stops at that crossing wall: trim it there."""
    def span(w):
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        h = abs(y1 - y2) < 1
        return h, (y1 if h else x1), min(x1, x2) if h else min(y1, y2), max(x1, x2) if h else max(y1, y2)

    info = [span(w) for w in walls]

    def touches(h, c, pos):
        return any(o[0] != h and abs(o[1] - pos) < 1 and o[2] - 1 <= c <= o[3] + 1 for o in info)

    for w, (h, c, a, b) in zip(walls, info):
        for end, inward in (("a", 1), ("b", -1)):
            pos = a if end == "a" else b
            if touches(h, c, pos):
                continue
            mid = pos + inward * half
            if touches(h, c, mid) and abs((mid / half) % 2 - 1) < 1e-6:
                if end == "a":
                    a = mid
                else:
                    b = mid
        if h:
            w["p1"], w["p2"] = (a, c), (b, c)
        else:
            w["p1"], w["p2"] = (c, a), (c, b)


# ---------------------------------------------------------------- check image
def draw(rgb, edges, fits, path, title):
    (sx, ix, *_), (sy, iy, *_) = fits[False], fits[True]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = cv2.addWeighted(bgr, 0.45, np.full_like(bgr, 255), 0.55, 0)
    col = {"wall": None, "wall_fit": (0, 140, 255)}
    for e in edges:
        h, ln, st = e["h"], e["line"], e["step"]
        if h:
            a = (int(ix + sx * st), int(iy + sy * ln))
            b = (int(ix + sx * (st + 1)), int(iy + sy * ln))
        else:
            a = (int(ix + sx * ln), int(iy + sy * st))
            b = (int(ix + sx * ln), int(iy + sy * (st + 1)))
        if e["cls"] == "wall":
            c = (200, 80, 0) if e["ext"] else (0, 150, 0)
            cv2.line(img, a, b, c, 3, cv2.LINE_AA)
        elif e["cls"] == "wall_fit":
            c = (220, 0, 220) if e.get("completed") else col["wall_fit"]
            cv2.line(img, a, b, c, 3, cv2.LINE_AA)
        elif e["f"]["int"] + e["f"]["ext"] + e["f"]["fit"] >= 0.25:
            cv2.line(img, a, b, (0, 0, 230), 1, cv2.LINE_AA)  # weak evidence, rejected
    legend = [((200, 80, 0), "外壁（455区間）"), ((0, 150, 0), "内壁（455区間）"),
              ((0, 140, 255), "建具・開口部（壁とみなした区間）"), ((220, 0, 220), "途切れを突っ切って補完した区間（要確認）"),
              ((0, 0, 230), "根拠が弱く不採用")]
    img = jp_text.legend(img, title, legend)
    cv2.imencode(".png", img)[1].tofile(path)


def extract_openings(edges, fits, cfg):
    """Runs of fitting edges (door / window counted as wall) = opening candidates, in mm and px."""
    grid = cfg["input"]["grid_mm"]
    (sx, ix, nx0, _), (sy, iy, _, ny1) = fits[False], fits[True]
    rows = {}
    for e in edges:
        if e["cls"] == "wall_fit":
            rows.setdefault((e["h"], e["line"]), []).append(e)
    out = []
    for (h, ln), lst in sorted(rows.items(), key=lambda kv: (not kv[0][0], kv[0][1])):
        lst.sort(key=lambda e: e["step"])
        run = [lst[0]]
        for e in lst[1:] + [None]:
            if e is not None and e["step"] == run[-1]["step"] + 1:
                run.append(e)
                continue
            a, b = run[0]["step"], run[-1]["step"] + 1
            if h:
                ymm = (ny1 - ln) * grid
                p1, p2 = ((a - nx0) * grid, ymm), ((b - nx0) * grid, ymm)
                px1, px2 = (ix + sx * a, iy + sy * ln), (ix + sx * b, iy + sy * ln)
            else:
                xmm = (ln - nx0) * grid
                p1, p2 = (xmm, (ny1 - b) * grid), (xmm, (ny1 - a) * grid)
                px1, px2 = (ix + sx * ln, iy + sy * a), (ix + sx * ln, iy + sy * b)
            out.append({"p1": p1, "p2": p2, "span_mm": (b - a) * grid, "px": [px1, px2],
                        "horizontal": h,
                        "kind": "bare" if all(r.get("completed") for r in run) else "fitting",
                        "exterior": all(r.get("ext") for r in run)})
            run = [e] if e is not None else []
    for i, o in enumerate(out, 1):
        o["no"] = i
    return out


def draw_openings(rgb, openings, path, title):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = cv2.addWeighted(bgr, 0.55, np.full_like(bgr, 255), 0.45, 0)
    for o in openings:
        (x1, y1), (x2, y2) = o["px"]
        c = (0, 140, 255) if o["kind"] == "fitting" else (220, 0, 220)
        pad = 9
        if o["horizontal"]:
            cv2.rectangle(img, (int(x1), int(y1 - pad)), (int(x2), int(y1 + pad)), c, 2)
            tx, ty = int((x1 + x2) / 2) - 10, int(y1 - pad - 4)
        else:
            cv2.rectangle(img, (int(x1 - pad), int(y1)), (int(x1 + pad), int(y2)), c, 2)
            tx, ty = int(x1 + pad + 3), int((y1 + y2) / 2)
        cv2.putText(img, str(o["no"]), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
    legend = [((0, 140, 255), "opening found from door/window lines"),
              ((220, 0, 220), "gap run through (no symbol seen)")]
    cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    for i, (c, t) in enumerate(legend):
        y = 75 + i * 28
        cv2.rectangle(img, (20, y - 8), (60, y + 8), c, 2)
        cv2.putText(img, t, (70, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imencode(".png", img)[1].tofile(path)


def main():
    img_path, cfg_path, prefix = sys.argv[1:4]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    rgb = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    if len(sys.argv) > 4:
        # optional: the text-erased image from clean_text.py. Walls are still judged on the
        # original (erasing text also ate wall pixels); the difference only marks where
        # lettering is, so it is not taken for fitting lines.
        global TEXT_MASK
        clean = cv2.imdecode(np.fromfile(sys.argv[4], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        diff = np.abs(rgb.astype(int) - clean.astype(int)).max(axis=2) > 40
        TEXT_MASK = cv2.dilate(diff.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    dims_path = Path(prefix + "_寸法.json")
    if dims_path.exists() and cfg["dimensions"].get("use", True):
        global DIM_EXTENT
        dims = json.load(open(dims_path, encoding="utf-8"))
        DIM_TICKS[False] = dims.get("anchor_x", [])
        DIM_TICKS[True] = dims.get("anchor_y", [])
        DIM_EXTENT = dims.get("extent_px")
    fits, gray_range = calibrate(rgb, cfg)
    if DIM_EXTENT:
        # the dimensions run to the building's outermost walls: judge every line inside them
        (sx, ix, nx0, nx1), (sy, iy, ny0, ny1) = fits[False], fits[True]
        ex = sorted(((DIM_EXTENT[0] - ix) / sx, (DIM_EXTENT[2] - ix) / sx))
        ey = sorted(((DIM_EXTENT[1] - iy) / sy, (DIM_EXTENT[3] - iy) / sy))
        fits[False] = (sx, ix, min(nx0, math.floor(ex[0] + 0.25)), max(nx1, math.ceil(ex[1] - 0.25)))
        fits[True] = (sy, iy, min(ny0, math.floor(ey[0] + 0.25)), max(ny1, math.ceil(ey[1] - 0.25)))
        # the outermost walls stand on the grid lines nearest the dimension extent (within a
        # quarter of a module; otherwise the extent is not trusted for this)
        for v, h in ((ex[0], False), (ex[1], False), (ey[0], True), (ey[1], True)):
            if abs(v - round(v)) <= 0.25:
                EXTENT_LINES.add((h, round(v)))
    names_path = Path(prefix + "_部屋名.json")
    if names_path.exists():
        (sx, ix, _, _), (sy, iy, _, _) = fits[False], fits[True]
        names = json.load(open(names_path, encoding="utf-8"))
        GENKAN_PTS[:] = [((n["x"] - ix) / sx, (n["y"] - iy) / sy) for n in names if n.get("name") == "玄関"]
    grid = cfg["input"]["grid_mm"]
    raw = judge_edges(rgb, fits, gray_range, cfg)
    edges = classify_staged(raw, cfg) if cfg["grid_detection"].get("staged", True) else classify(raw, cfg)
    walls = edges_to_walls(edges, fits, cfg)
    retype_inner_walls(walls, cfg)
    name = prefix.replace("\\", "/").split("/")[-1]
    kx, ky = grid / fits[False][0], grid / abs(fits[True][0])
    print(f"{name}: gray={gray_range} mm/px x={kx:.4f} y={ky:.4f} edges={len(edges)} "
          f"wall={sum(e['cls'] == 'wall' for e in edges)} fit={sum(e['cls'] == 'wall_fit' for e in edges)} "
          f"walls={len(walls)}")
    draw(rgb, edges, fits, prefix + "_壁の判定.png", f"{name}　壁の判定")
    # origin for downstream check images: pixel of (X=0, Y=0)
    (sx, ix, nx0, _), (sy, iy, _, ny1) = fits[False], fits[True]
    origin = (ix + sx * nx0, iy + sy * ny1)
    json.dump({"mm_per_px": (kx + ky) / 2, "origin_px": origin, "gray_range": gray_range,
               "method": "grid_v2",
               # px = slope * n + icpt for grid index n; X mm = (n - nx0)*455, Y mm = (ny1 - n)*455
               "fits": {"x": list(fits[False]), "y": list(fits[True])},
               "walls": walls},
              open(prefix + "_walls.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump([{k: v for k, v in e.items()} for e in edges],
              open(prefix + "_edges.json", "w", encoding="utf-8"), ensure_ascii=False)
    # openings are judged by opening_detect.py (pixel level); extract_openings() is the old 455-unit view


if __name__ == "__main__":
    main()
