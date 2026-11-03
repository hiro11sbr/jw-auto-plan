"""Stage 1: detect wall center lines from a raster plan image.

usage: wall_detect.py <plan.png> <config.json> <out_prefix>
writes <out_prefix>_walls.json and <out_prefix>_check.png
"""
import json
import math
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- mask
def wall_mask(rgb, cfg):
    c = cfg["wall_detection"]
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    tol = c["neutral_tolerance"]
    neutral = (np.abs(r - g) <= tol) & (np.abs(g - b) <= tol) & (np.abs(r - b) <= tol)
    v = (r + g + b) // 3
    lo, hi = c["gray_min"], c["gray_max"]
    if c["gray_auto"]:
        # strictly inside the configured range: a site plan has large neutral areas just outside
        # it (gravel ≈200 on プラン2) that would otherwise win
        hist = np.bincount(v[neutral & (v >= lo) & (v <= hi)].ravel(), minlength=256)
        peak = int(np.argmax(hist))
        hw = c["gray_auto_halfwidth"]
        lo, hi = peak - hw, peak + hw
    return neutral & (v >= lo) & (v <= hi), (lo, hi)


# ---------------------------------------------------------------- runs -> segments
def runs_1d(arr):
    """Return (start, length) of True runs in a 1D bool array."""
    a = np.concatenate(([0], arr.astype(np.int8), [0]))
    d = np.diff(a)
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    return s, e - s


def detect_direction(mask, cfg, horizontal):
    """horizontal=True finds horizontal walls (thickness measured along y)."""
    c = cfg["wall_detection"]
    tmin, tmax = c["thickness_px"]
    m = mask if horizontal else mask.T
    # samples: (pos_along, center_across, thickness)
    samples = []
    for x in range(m.shape[1]):
        s, ln = runs_1d(m[:, x])
        ok = (ln >= tmin) & (ln <= tmax)
        for st, l in zip(s[ok], ln[ok]):
            samples.append((x, st + l / 2.0, l))
    if not samples:
        return []
    samples = np.array(samples)
    # cluster locally: pieces per 1px center bin, then join pieces of neighbouring bins that overlap
    gap = c["small_gap_px"]
    pieces = []
    keys = np.floor(samples[:, 1]).astype(int)
    for k in np.unique(keys):
        sel = samples[keys == k]
        sel = sel[np.argsort(sel[:, 0])]
        for part in np.split(sel, np.flatnonzero(np.diff(sel[:, 0]) > gap) + 1):
            pieces.append([k, part[0, 0], part[-1, 0], part])
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pieces_sorted = sorted(range(len(pieces)), key=lambda i: pieces[i][0])
    for ii, i in enumerate(pieces_sorted):
        ki, ai, bi = pieces[i][:3]
        for j in pieces_sorted[ii + 1:]:
            kj, aj, bj = pieces[j][:3]
            if kj - ki > 2:
                break
            if aj <= bi + gap and bj >= ai - gap:
                parent[find(j)] = find(i)
    groups = {}
    for i in range(len(pieces)):
        groups.setdefault(find(i), []).append(pieces[i][3])
    clusters = []
    for parts in groups.values():
        cl = np.concatenate(parts)
        med = np.median(cl[:, 1])
        cl = cl[np.abs(cl[:, 1] - med) <= 2.5]
        # one sample per position (keep the one nearest the median center)
        cl = cl[np.lexsort((np.abs(cl[:, 1] - med), cl[:, 0]))]
        _, first = np.unique(cl[:, 0], return_index=True)
        clusters.append(cl[first])

    segs = []
    for cl in clusters:
        cl = cl[np.argsort(cl[:, 0])]
        xs = cl[:, 0]
        breaks = np.flatnonzero(np.diff(xs) > c["small_gap_px"])
        for part in np.split(cl, breaks + 1):
            # require mostly-filled coverage to reject sparse noise
            span = part[-1, 0] - part[0, 0] + 1
            if span < c["min_segment_px"] or len(part) < 0.6 * span:
                continue
            segs.append({
                "a": float(part[0, 0]), "b": float(part[-1, 0] + 1),
                "c": float(np.median(part[:, 1])), "t": float(np.median(part[:, 2])),
                "n": int(len(part)),
            })
    # merge segments of the same line (center within tolerance, same class)
    ext_t = c["exterior_min_thickness_px"]
    for s in segs:
        s["ext"] = s["t"] >= ext_t
    segs.sort(key=lambda s: (s["c"], s["a"]))
    merged = []
    for s in segs:
        hit = None
        for m_ in merged:
            if m_["ext"] == s["ext"] and abs(m_["c"] - s["c"]) <= c["line_tolerance_px"] \
                    and s["a"] <= m_["b"] + c["small_gap_px"] and s["b"] >= m_["a"] - c["small_gap_px"]:
                hit = m_
                break
        if hit:
            w1, w2 = hit["b"] - hit["a"], s["b"] - s["a"]
            hit["c"] = (hit["c"] * w1 + s["c"] * w2) / (w1 + w2)
            hit["t"] = max(hit["t"], s["t"])
            hit["a"], hit["b"] = min(hit["a"], s["a"]), max(hit["b"], s["b"])
        else:
            merged.append(dict(s, bridges=[]))
    for s in merged:
        s["horizontal"] = horizontal
    return merged


def bridge_openings(segs, cfg):
    """Join collinear same-class segments across openings (doors/windows/red text)."""
    c = cfg["wall_detection"]
    out = []
    groups = {}
    for s in segs:
        key = (s["horizontal"], s["ext"])
        groups.setdefault(key, []).append(s)
    for (hz, ext), lst in groups.items():
        lst.sort(key=lambda s: (s["c"], s["a"]))
        limit = c["bridge_gap_exterior_px"] if ext else c["bridge_gap_interior_px"]
        used = [False] * len(lst)
        for i, s in enumerate(lst):
            if used[i]:
                continue
            line = dict(s)
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j, t in enumerate(lst):
                    if used[j] or abs(t["c"] - line["c"]) > c["line_tolerance_px"]:
                        continue
                    gap = max(t["a"] - line["b"], line["a"] - t["b"])
                    if gap <= limit:
                        if gap > c["small_gap_px"]:
                            line["bridges"].append((min(line["b"], t["b"]), max(line["a"], t["a"])))
                        line["a"], line["b"] = min(line["a"], t["a"]), max(line["b"], t["b"])
                        line["t"] = max(line["t"], t["t"])
                        used[j] = True
                        changed = True
            out.append(line)
    return out


def connect_junctions(segs, cfg):
    """Extend/trim each end to the center of a perpendicular wall it touches."""
    tol = cfg["wall_detection"]["junction_tolerance_px"]
    hs = [s for s in segs if s["horizontal"]]
    vs = [s for s in segs if not s["horizontal"]]
    for s in segs:
        s["joins"] = 0
    for group, others in ((hs, vs), (vs, hs)):
        for s in group:
            for end in ("a", "b"):
                p = s[end]
                best = None
                for o in others:
                    # o.c is o's center along s's axis; s.c must fall inside o's span
                    if o["a"] - tol <= s["c"] <= o["b"] + tol and abs(o["c"] - p) <= o["t"] / 2 + tol:
                        d = abs(o["c"] - p)
                        if best is None or d < best[0]:
                            best = (d, o)
                if not best:
                    # free end beside a door/window: extend across the opening to the next wall
                    reach = cfg["wall_detection"]["bridge_gap_exterior_px" if s["ext"] else "bridge_gap_interior_px"]
                    sign = -1 if end == "a" else 1
                    for o in others:
                        d = (o["c"] - p) * sign
                        if 0 < d <= reach and o["a"] - tol <= s["c"] <= o["b"] + tol:
                            if best is None or d < best[0]:
                                best = (d, o)
                    if best:
                        lo_, hi_ = sorted((p, best[1]["c"]))
                        s["bridges"].append((lo_, hi_))
                if best:
                    s[end] = best[1]["c"]
                    s["joins"] += 1
                    best[1]["joins"] += 1
    # a wall that touches no other wall is furniture/counter edge, not a wall
    keep_free = cfg["wall_detection"]["keep_free_exterior_px"]
    return [s for s in segs if s["joins"] > 0 or (s["ext"] and s["b"] - s["a"] >= keep_free)]


# ---------------------------------------------------------------- scale / grid
def fit_scale(segs, cfg):
    grid = cfg["input"]["grid_mm"]
    lo, hi = cfg["input"]["mm_per_px_search"]
    cs_h = np.array([s["c"] for s in segs if s["horizontal"]])
    cs_v = np.array([s["c"] for s in segs if not s["horizontal"]])
    w_h = np.array([s["b"] - s["a"] for s in segs if s["horizontal"]])
    w_v = np.array([s["b"] - s["a"] for s in segs if not s["horizontal"]])
    best = None
    for k in np.arange(lo, hi, 0.0005):
        score = 0.0
        for cs, w in ((cs_h, w_h), (cs_v, w_v)):
            ph = 2 * np.pi * (cs * k) / grid
            score += abs(np.sum(w * np.exp(1j * ph))) / np.sum(w)
        if best is None or score > best[0]:
            best = (score, k)
    return best[1], best[0] / 2


def snap(segs, k, cfg):
    grid = cfg["input"]["grid_mm"]
    half = cfg["input"]["half_grid_mm"]
    warn = cfg["wall_detection"]["snap_warn_mm"]
    # origin: leftmost exterior vertical wall = X0, bottom-most exterior horizontal wall = Y0
    ext_v = [s for s in segs if not s["horizontal"] and s["ext"]] or [s for s in segs if not s["horizontal"]]
    ext_h = [s for s in segs if s["horizontal"] and s["ext"]] or [s for s in segs if s["horizontal"]]
    x0 = min(s["c"] for s in ext_v)
    y0 = max(s["c"] for s in ext_h)  # image y grows downward

    def to_x(px):
        return (px - x0) * k

    def to_y(py):
        return (y0 - py) * k

    def q(v, end=False):
        g = round(v / grid) * grid
        if abs(v - g) <= warn or not cfg["input"]["allow_off_grid"]:
            # no exception dimensions are read yet, so every wall goes on the 455 grid
            return g, abs(v - g), "grid"
        h = round(v / half) * half
        if abs(v - h) <= warn:
            # wall ends on a half grid are normal; only centers are reported
            return h, abs(v - h), "grid" if end else "half"
        return h, abs(v - h), "warn"

    walls = []
    for s in segs:
        if s["horizontal"]:
            y, res, kind = q(to_y(s["c"]))
            xa, _, ka = q(to_x(s["a"]), True)
            xb, _, kb = q(to_x(s["b"]), True)
            p1, p2 = (min(xa, xb), y), (max(xa, xb), y)
            raw = (to_x(s["a"]), to_y(s["c"]), to_x(s["b"]), to_y(s["c"]))
        else:
            x, res, kind = q(to_x(s["c"]))
            ya, _, ka = q(to_y(s["a"]), True)
            yb, _, kb = q(to_y(s["b"]), True)
            p1, p2 = (x, min(ya, yb)), (x, max(ya, yb))
            raw = (to_x(s["c"]), to_y(s["a"]), to_x(s["c"]), to_y(s["b"]))
        if p1 == p2:
            continue
        flags = []
        if kind == "half":
            flags.append("半グリッド")
        if kind == "warn":
            flags.append(f"グリッドずれ{res:.0f}mm")
        if "warn" in (ka, kb):
            flags.append("端部ずれ")
        walls.append({
            "type": "exterior" if s["ext"] else "interior",
            "p1": p1, "p2": p2, "raw_mm": [round(v, 1) for v in raw],
            "thickness_px": s["t"], "center_residual_mm": round(res, 1),
            "bridged_px": [[round(a, 1), round(b, 1)] for a, b in s["bridges"]],
            "flags": flags, "_px": s,
        })
    return walls, (x0, y0)


# ---------------------------------------------------------------- check image
def draw_check(rgb, walls, k, origin, path, title):
    x0, y0 = origin
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = cv2.addWeighted(base, 0.45, np.full_like(base, 255), 0.55, 0)

    def px(p):
        return int(round(p[0] / k + x0)), int(round(y0 - p[1] / k))

    for w in walls:
        col = (200, 80, 0) if w["type"] == "exterior" else (0, 150, 0)
        center_flags = [f for f in w["flags"] if f != "端部ずれ"]
        if center_flags == ["半グリッド"]:
            col = (0, 200, 230)
        elif center_flags:
            col = (0, 0, 230)
        a, b = px(w["p1"]), px(w["p2"])
        cv2.line(img, a, b, col, 2, cv2.LINE_AA)
        s = w["_px"]
        for g1, g2 in s["bridges"]:
            if s["horizontal"]:
                pa, pb = (int(g1), int(s["c"])), (int(g2), int(s["c"]))
            else:
                pa, pb = (int(s["c"]), int(g1)), (int(s["c"]), int(g2))
            cv2.line(img, pa, pb, (0, 140, 255), 4, cv2.LINE_AA)
        cv2.circle(img, a, 3, col, -1)
        cv2.circle(img, b, 3, col, -1)
    legend = [((200, 80, 0), "exterior wall center"), ((0, 150, 0), "interior wall center"),
              ((0, 140, 255), "bridged opening"), ((0, 200, 230), "half grid (227.5)"),
              ((0, 0, 230), "needs check (off grid)")]
    cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    for i, (c, t) in enumerate(legend):
        y = 75 + i * 28
        cv2.line(img, (20, y), (60, y), c, 4)
        cv2.putText(img, t, (70, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imencode(".png", img)[1].tofile(path)


def main():
    img_path, cfg_path, prefix = sys.argv[1:4]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    rgb = bgr[:, :, ::-1].copy()
    mask, gray_range = wall_mask(rgb, cfg)
    segs = detect_direction(mask, cfg, True) + detect_direction(mask, cfg, False)
    segs = bridge_openings(segs, cfg)
    segs = connect_junctions(segs, cfg)
    k, conf = fit_scale(segs, cfg)
    walls, origin = snap(segs, k, cfg)
    print(f"gray range={gray_range} scale={k:.4f} mm/px grid-fit={conf:.3f} walls={len(walls)}")
    for w in walls:
        print(f"  {w['type']:<8} {w['p1']} -> {w['p2']} t={w['thickness_px']:.0f}px "
              f"res={w['center_residual_mm']}mm bridges={len(w['bridged_px'])} {' '.join(w['flags'])}")
    draw_check(rgb, walls, k, origin, prefix + "_check.png", f"{prefix.split(chr(92))[-1]}  {k:.3f} mm/px")
    for w in walls:
        w.pop("_px")
    json.dump({"mm_per_px": k, "grid_fit": conf, "origin_px": origin, "gray_range": gray_range,
               "walls": walls}, open(prefix + "_walls.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
