"""Regions (区画) from walls + fittings, room names, and the consistency checks.

usage: regions_check.py <plan.png> <clean.png> <walls.json> <openings.json> <部屋名.json>
                        <部屋名リスト.csv> <config.json> <out_prefix>
writes <out_prefix>_区画.json and <out_prefix>_区画の確認.png

Checks (they only mark places to look at again; nothing in the drawing is changed):
  - every region must be reachable from the entrance through doors / openings
  - a UB or a toilet has exactly one door; an entrance normally has one entrance door
  - a region whose floor looks like two different floors may hide a missed wall (supporting check)
"""
import json
import math
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
import jp_text  # noqa: E402
import read_room_names as rrn  # noqa: E402
from opening_detect import Mapper  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
DOORS = {"swing", "sliding", "sliding_pocket", "ub_door", "entrance_door", "storage", "folding"}


def snap_walls(det, half):
    segs = []
    for w in det["walls"]:
        p = tuple(round(v / half) * half for v in w["p1"])
        q = tuple(round(v / half) * half for v in w["p2"])
        if p != q:
            segs.append(LineString([p, q]))
    return segs


def build_regions(det, half, min_area):
    segs = snap_walls(det, half)
    polys = [p for p in polygonize(unary_union(segs)) if p.area >= min_area]
    return polys


def raster_regions(clean, det, M, gray_range, c):
    """Regions as a person sees them: floor areas divided by the gray wall band and by the
    detected walls / fittings. Returns (label image, list of region masks' ids, footprint mask)."""
    H, W = clean.shape[:2]
    lo, hi = gray_range
    v = clean.astype(int).mean(axis=2)
    b, g, r = (clean[..., i].astype(int) for i in range(3))
    neutral = (np.abs(r - g) <= 12) & (np.abs(g - b) <= 12)
    barrier = (neutral & (v >= lo) & (v <= hi)).astype(np.uint8)
    # gray furniture / counters are islands inside rooms; only thin gray (walls) divides rooms
    k = c["wall_band_max_px"]
    thick = cv2.morphologyEx(barrier, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    barrier[thick > 0] = 0
    for w in det["walls"]:
        p = (int(round(M.x(w["p1"][0]))), int(round(M.y(w["p1"][1]))))
        q = (int(round(M.x(w["p2"][0]))), int(round(M.y(w["p2"][1]))))
        cv2.line(barrier, p, q, 1, c["wall_line_px"])
    # 框 (floor-level step, e.g. 玄関 / ホール): a pair of close parallel lines from wall to wall.
    # It divides regions but is passable.
    # building footprint = everything enclosed by the outer walls
    segs = snap_walls(det, 227.5)
    polys = list(polygonize(unary_union(segs)))
    foot = np.zeros((H, W), np.uint8)
    if polys:
        fp = unary_union(polys)
        geoms = [fp] if fp.geom_type == "Polygon" else list(fp.geoms)
        for gm in geoms:
            cv2.fillPoly(foot, [region_px(Polygon(gm.exterior), M)], 1)
    # 框 is for drawing later (it can be bent); as a region divider it caught furniture, so off by default
    kamachi = find_kamachi(clean, barrier, foot, (lo, hi), c) if c.get("kamachi_detect") else []
    for (p, q) in kamachi:
        cv2.line(barrier, p, q, 1, 3)
    space = ((foot > 0) & (barrier == 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(space, connectivity=4)
    mmpp = det["mm_per_px"]
    min_px = c["min_area_mm2"] / (mmpp * mmpp)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
    return lab, keep, stats, mmpp, kamachi, foot


def find_kamachi(clean, walls_mask, foot, gray_range, c):
    """Close parallel straight line pairs (any angle) inside the building, lying on open floor
    (not a wall band's two outlines), whose ends both reach a wall."""
    lo_g, hi_g = gray_range
    v = clean.astype(int).mean(axis=2)
    dark = (v < 110).astype(np.uint8) * 255
    segs = cv2.HoughLinesP(dark, 1, np.pi / 360, threshold=40, minLineLength=c["kamachi_min_px"], maxLineGap=3)
    if segs is None:
        return []
    segs = [tuple(int(v) for v in s) for s in np.asarray(segs).reshape(-1, 4)]  # OpenCV 4 / 5 shapes
    H, W = walls_mask.shape

    def ang(s):
        return math.atan2(s[3] - s[1], s[2] - s[0]) % math.pi

    def near_wall(x, y, r=8):
        x, y = int(x), int(y)
        return walls_mask[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)].any()

    out = []
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            da = abs(ang(a) - ang(b))
            if min(da, math.pi - da) > math.radians(3):
                continue
            # distance between the two parallel lines
            ax, ay, bx, by = a
            L = math.hypot(bx - ax, by - ay)
            if L == 0:
                continue
            nx, ny = -(by - ay) / L, (bx - ax) / L
            dist = abs((b[0] - ax) * nx + (b[1] - ay) * ny)
            if not (c["kamachi_gap_min_px"] <= dist <= c["kamachi_gap_max_px"]):
                continue
            mx, my = int((ax + bx) / 2), int((ay + by) / 2)
            if not foot[min(H - 1, max(0, my)), min(W - 1, max(0, mx))]:
                continue  # outside the building (cars, site lines)
            # between the two lines must be floor, not the gray of a wall band
            qx, qy = int(mx + nx * dist / 2), int(my + ny * dist / 2)
            if (np.sign((b[0] - ax) * nx + (b[1] - ay) * ny) < 0):
                qx, qy = int(mx - nx * dist / 2), int(my - ny * dist / 2)
            vq = clean[max(0, qy - 1):qy + 2, max(0, qx - 1):qx + 2].astype(int).mean(axis=2)
            if ((vq >= lo_g) & (vq <= hi_g)).mean() >= 0.5:
                continue
            if near_wall(mx, my, 3):
                continue  # runs along a wall: its outlines, not a 框
            if near_wall(ax, ay) and near_wall(bx, by):
                out.append(((int(ax), int(ay)), (int(bx), int(by))))
    # one entry per 框 (a pair gives two hits)
    uniq = []
    for p, q in out:
        if not any(math.hypot(p[0] - u[0][0], p[1] - u[0][1]) < 10 and math.hypot(q[0] - u[1][0], q[1] - u[1][1]) < 10
                   for u in uniq):
            uniq.append((p, q))
    return uniq


def region_px(poly, M):
    return np.array([[M.x(x), M.y(y)] for x, y in poly.exterior.coords], np.int32)


def ocr_region(img, bbox, variants, c):
    """Read the name inside one region: crop it, enlarge, OCR, match against the list."""
    x0, y0, x1, y1 = bbox
    if x1 - x0 < 20 or y1 - y0 < 15:
        return None
    crop = img[y0:y1, x0:x1].copy()
    r, g, b = (crop[..., 2].astype(int), crop[..., 1].astype(int), crop[..., 0].astype(int))
    crop[(r - g > 25) & (r - b > 25) & (np.abs(g - b) <= 30)] = 255
    best = None
    for sc in (4, 6):
        for ln in rrn.merge_fragments(rrn.ocr(crop, sc), c["merge_gap_px"]):
            m = rrn.best_match(ln["text"], variants, c["match_min_score"])
            if m and m[1] != "除外語" and (best is None or m[2] > best[2]):
                bx = (ln["box"][0] + ln["box"][2]) / 2 + x0
                by = (ln["box"][1] + ln["box"][3]) / 2 + y0
                best = (m[0], m[1], m[2], ln["text"], bx, by)
    return best


def two_floors(clean, region_mask, c):
    """Supporting check: does the floor inside look like two different floors side by side?"""
    mask = cv2.erode(region_mask.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero(mask)
    if len(xs) < 400:
        return False
    px = clean[ys, xs].astype(np.float32)
    keep = px.mean(axis=1) > 150  # floor, not furniture / lines
    px, xs, ys = px[keep], xs[keep], ys[keep]
    if len(px) < 400:
        return False
    _, lab, cen = cv2.kmeans(px, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                             3, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    share = min((lab == 0).mean(), (lab == 1).mean())
    diff = float(np.abs(cen[0] - cen[1]).max())
    if share < c["floor_min_share"] or diff < c["floor_min_color_diff"]:
        return False
    # the two colours must sit in separate parts of the region, not be mixed (wood grain)
    c0 = np.array([xs[lab == 0].mean(), ys[lab == 0].mean()])
    c1 = np.array([xs[lab == 1].mean(), ys[lab == 1].mean()])
    size = max(xs.max() - xs.min(), ys.max() - ys.min())
    return float(np.linalg.norm(c0 - c1)) >= c["floor_min_separation"] * size


SEG_STYLE = {  # (BGR colour, legend label)
    "外壁": ((200, 80, 0), "外壁"),
    "内壁": ((0, 150, 0), "内壁"),
    "壁（判定漏れ）": ((0, 200, 255), "壁（区画では境目なのに、壁の判定にない）"),
    "建具": ((255, 0, 200), "建具（開口部）"),
    "窓": ((230, 160, 0), "窓"),
    "不明": ((0, 0, 230), "境目なのに壁も建具も見つからない（要確認）"),
}
KIND_SHORT = {"swing": "開", "sliding_pocket": "引込", "sliding": "引", "entrance_door": "玄関",
              "ub_door": "UB", "storage": "収", "window": "窓", "folding": "折"}


def boundary_judgement(img, clean, det, M, lab, by_label, region_at, ops, kamachi, foot, cfg, prefix, name):
    """Walk every 227.5 segment of every grid line; where the regions on its two sides differ,
    the segment is a region boundary (= a wall centre line). Judge each one: wall / fitting /
    window / gray wall the line judgement missed / nothing found."""
    c = cfg["regions"]
    half = cfg["input"]["half_grid_mm"]
    lo, hi = det["gray_range"]
    mid_g, w_g = (lo + hi) / 2, cfg["grid_detection"]["gray_halfwidth"]
    glo, ghi = mid_g - w_g, mid_g + w_g
    xs = [v for w in det["walls"] for v in (w["p1"][0], w["p2"][0])]
    ys = [v for w in det["walls"] for v in (w["p1"][1], w["p2"][1])]
    x_lo, x_hi, y_lo, y_hi = max(0, min(xs)), max(xs), max(0, min(ys)), max(ys)
    probe = c["door_probe_px"]
    vimg = clean.astype(int).mean(axis=2)

    H, W = foot.shape

    def side_id(px, py):
        """Region id, 0 = outside the building, None = inside but on a wall / furniture."""
        r = region_at(px, py, 2)
        if r:
            return r["id"]
        xi, yi = int(round(px)), int(round(py))
        if not (0 <= xi < W and 0 <= yi < H) or not foot[yi, xi]:
            return 0
        return None

    def on_wall(h, cmm, a, b):
        for w in det["walls"]:
            (x1, y1), (x2, y2) = w["p1"], w["p2"]
            wh = abs(y1 - y2) < 1
            if wh != h:
                continue
            wc = y1 if h else x1
            lo_, hi_ = sorted((x1, x2) if h else (y1, y2))
            if abs(wc - cmm) < 60 and lo_ - 1 <= (a + b) / 2 <= hi_ + 1:
                return w["type"]
        return None

    def opening(h, cmm, a, b):
        for o in ops:
            if o["horizontal"] != h:
                continue
            oc_ = o["p1_mm"][1] if h else o["p1_mm"][0]
            lo_, hi_ = sorted((o["p1_mm"][0], o["p2_mm"][0]) if h else (o["p1_mm"][1], o["p2_mm"][1]))
            if abs(oc_ - cmm) < 80 and lo_ <= (a + b) / 2 <= hi_:
                return o["kind"]
        return None

    segs = []
    for h in (True, False):
        c_lo, c_hi = (y_lo, y_hi) if h else (x_lo, x_hi)
        a_lo, a_hi = (x_lo, x_hi) if h else (y_lo, y_hi)
        for n in range(int(round(c_lo / half)), int(round(c_hi / half)) + 1):
            cmm = n * half
            cpx = M.y(cmm) if h else M.x(cmm)
            for k in range(int(round(a_lo / half)), int(round(a_hi / half))):
                a, b = k * half, (k + 1) * half
                pa, pb = (M.x(a), M.x(b)) if h else (M.y(a), M.y(b))
                left, right = [], []
                for t in (0.2, 0.35, 0.5, 0.65, 0.8):
                    p = pa + (pb - pa) * t
                    if h:
                        left.append(side_id(p, cpx - probe))
                        right.append(side_id(p, cpx + probe))
                    else:
                        left.append(side_id(cpx - probe, p))
                        right.append(side_id(cpx + probe, p))
                la = max(set(left), key=left.count)
                ra = max(set(right), key=right.count)
                if la is None or ra is None:
                    continue  # a probe fell on a wall or furniture: not a clean boundary here
                if la == ra or left.count(la) < 3 or right.count(ra) < 3 or (la == 0 and ra == 0):
                    continue
                kind = opening(h, cmm, a, b)
                wt = on_wall(h, cmm, a, b)
                if kind == "window":
                    cls = "窓"
                elif kind:
                    cls = "建具"
                elif wt:
                    cls = "外壁" if wt == "exterior" else "内壁"
                else:
                    # gray band right on this segment: a wall the line judgement missed
                    p0, p1 = sorted((int(pa), int(pb)))
                    ci = int(round(cpx))
                    band = vimg[ci - 4:ci + 5, p0:p1] if h else vimg[p0:p1, ci - 4:ci + 5]
                    gray = float(((band >= glo) & (band <= ghi)).mean()) if band.size else 0
                    cls = "壁（判定漏れ）" if gray >= 0.4 else "不明"
                segs.append({"h": h, "c_mm": cmm, "a": a, "b": b, "cls": cls, "kind": kind,
                             "sides": [la, ra]})

    out = cv2.addWeighted(img, 0.5, np.full_like(img, 255), 0.5, 0)
    texts = []
    for s in segs:
        col = SEG_STYLE[s["cls"]][0]
        if s["h"]:
            p = (int(M.x(s["a"])), int(M.y(s["c_mm"])))
            q = (int(M.x(s["b"])), int(M.y(s["c_mm"])))
        else:
            p = (int(M.x(s["c_mm"])), int(M.y(s["a"])))
            q = (int(M.x(s["c_mm"])), int(M.y(s["b"])))
        cv2.line(out, p, q, col, 4 if s["cls"] in ("不明", "壁（判定漏れ）") else 3)
        cv2.circle(out, p, 2, (0, 0, 0), -1)
    # label each run of fitting segments once with its type
    seen = set()
    for s in segs:
        if s["cls"] == "建具" and s["kind"]:
            key = (s["h"], s["c_mm"], s["kind"], round(s["a"] / 455))
            if key in seen:
                continue
            seen.add(key)
            mx, my = (M.x((s["a"] + s["b"]) / 2), M.y(s["c_mm"])) if s["h"] else (M.x(s["c_mm"]), M.y((s["a"] + s["b"]) / 2))
            texts.append((KIND_SHORT.get(s["kind"], "?"), (int(mx) + 4, int(my) + 3), (120, 0, 120), 12))
    for (p, q) in kamachi:
        cv2.line(out, p, q, (160, 0, 160), 3)
    out = jp_text.put(out, texts)
    rows = [v for v in SEG_STYLE.values()] + [((160, 0, 160), "框（通り抜けできる境目）")]
    out = jp_text.legend(out, f"{name}　区画から見た壁と建具", rows)
    cv2.imencode(".png", out)[1].tofile(prefix + "_区画から見た壁と建具.png")
    counts = {}
    for s in segs:
        counts[s["cls"]] = counts.get(s["cls"], 0) + 1
    json.dump(segs, open(prefix + "_境目.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{name}: boundary segments (227.5) = {len(segs)} " + " ".join(f"{k}={v}" for k, v in counts.items()))


def main():
    (img_path, clean_path, walls_path, ops_path, names_path, list_path, cfg_path, prefix) = sys.argv[1:9]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    c = cfg["regions"]
    rc = cfg["room_names"]
    det = json.load(open(walls_path, encoding="utf-8"))
    ops = [o for o in json.load(open(ops_path, encoding="utf-8")) if not o["rejected"]]
    names = [n for n in json.load(open(names_path, encoding="utf-8")) if n.get("name") and n["kind"] != "除外語"]
    variants = rrn.load_names(list_path)
    M = Mapper(det, cfg["input"]["grid_mm"])
    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    clean = cv2.imdecode(np.fromfile(clean_path, np.uint8), cv2.IMREAD_COLOR)
    lo, hi = det["gray_range"]
    mid_g, w_g = (lo + hi) / 2, cfg["grid_detection"]["gray_halfwidth"]
    lab, keep, stats, mmpp, kamachi, foot = raster_regions(clean, det, M, (mid_g - w_g, mid_g + w_g), c)
    regions = []
    for k, li in enumerate(keep):
        x, y, w, h, area = stats[li]
        regions.append({"id": k + 1, "label": li, "bbox": (x, y, x + w, y + h), "area_m2": area * mmpp * mmpp / 1e6,
                        "names": [], "doors": [], "entrance": 0, "flags": []})
    by_label = {r["label"]: r for r in regions}

    def region_at(px, py, reach=4):
        """Region under a pixel (or the nearest one within a few px, text sat on furniture lines)."""
        H, W = lab.shape
        for rr in range(reach + 1):
            ys = slice(max(0, int(py) - rr), min(H, int(py) + rr + 1))
            xs = slice(max(0, int(px) - rr), min(W, int(px) + rr + 1))
            ids = [i for i in np.unique(lab[ys, xs]) if i in by_label]
            if ids:
                return by_label[ids[0]]
        return None

    # printed areas like "(9.0帖)": compared with the region's real area below
    import re
    for t in json.load(open(names_path, encoding="utf-8")):
        raw = rrn.norm(t["raw"])
        mt = re.search(r"([\d\.,．]+)帖", raw)
        if mt:
            s = mt.group(1).replace(",", ".").replace("．", ".")
            if "." not in s:
                # OCR drops the decimal point; 帖 is printed with one decimal (18.7, 9.0)
                if len(s) < 2:
                    continue  # one digit is a partial read (5.2 -> 2): not reliable
                s = s[:-1] + "." + s[-1]
            try:
                jo = float(s)
            except ValueError:
                continue
            if 0.5 <= jo <= 40:
                r = region_at(t["x"], t["y"], 8)
                if r:
                    r.setdefault("jo", []).append(jo)

    # room names read from the whole plan, then a region-by-region reading for the rest
    for n in names:
        r = region_at(n["x"], n["y"], 8)
        if r and n["name"] not in r["names"]:
            r["names"].append(n["name"])
    for r in regions:
        if not r["names"]:
            m = ocr_region(img, (r["bbox"][0] + 4, r["bbox"][1] + 4, r["bbox"][2] - 4, r["bbox"][3] - 4), variants, rc)
            if m:
                r["names"].append(m[0])
                r["name_read"] = m[3]

    # which regions each door / opening joins: look a little to each side of its middle
    adj = {r["id"]: set() for r in regions}
    adj[0] = set()  # 0 = outside
    for o in ops:
        if o["kind"] not in DOORS:
            continue
        p0, p1, cpx = o["px"]
        m_along = (p0 + p1) / 2
        sides = []
        for s in (-1, 1):
            off = c["door_probe_px"] * s
            px, py = (m_along, cpx + off) if o["horizontal"] else (cpx + off, m_along)
            r = region_at(px, py, 3)
            sides.append(r["id"] if r else 0)
        for rid in set(sides):
            if rid:
                rr = next(x for x in regions if x["id"] == rid)
                rr["doors"].append(o["kind"])
                if o["kind"] == "entrance_door":
                    rr["entrance"] += 1
        a, b = sides
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    # 框 is a passable boundary
    for (p, q) in kamachi:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        L = math.hypot(q[0] - p[0], q[1] - p[1]) or 1
        nx, ny = -(q[1] - p[1]) / L, (q[0] - p[0]) / L
        ra = region_at(mx + nx * 10, my + ny * 10, 3)
        rb = region_at(mx - nx * 10, my - ny * 10, 3)
        if ra and rb and ra is not rb:
            adj[ra["id"]].add(rb["id"])
            adj[rb["id"]].add(ra["id"])

    # reachability from the entrance (or from outside through any door)
    # start at the entrance; an upper floor has none, so start at the stairs
    start = ([r["id"] for r in regions if "玄関" in r["names"]]
             or [r["id"] for r in regions if "階段" in r["names"]] or [0])
    seen = set(start)
    q = deque(start)
    while q:
        a = q.popleft()
        for b in adj[a]:
            if b not in seen:
                seen.add(b)
                q.append(b)
    main_names = {"居室", "水回り", "玄関・動線", "収納"}
    kinds = {}
    for v, nm, kind in variants:
        kinds.setdefault(nm, kind)
    for r in regions:
        rmask = lab == r["label"]
        if r["id"] not in seen:
            r["flags"].append("通り抜けできない（建具の読み落とし？）")
        nd = len(r["doors"])
        if ("浴室" in r["names"] or "トイレ" in r["names"]) and nd != 1:
            nm = "浴室" if "浴室" in r["names"] else "トイレ"
            r["flags"].append(f"{nm}なのに建具が{nd}枚（読み取りを確認）")
        if "玄関" in r["names"] and r["entrance"] > 1:
            r["flags"].append(f"玄関ドアが{r['entrance']}枚（二世帯住宅でなければ確認）")
        # two rooms in one region: only rooms that normally have a wall/fitting between them
        rooms = list(dict.fromkeys(n for n in r["names"] if kinds.get(n) in main_names))
        # rooms that may share a region: linked through the allowed open pairs (LDK-ヌック,
        # LDK-階段 => ヌック and 階段 may share LDK's region)
        link = {}
        for a_, b_ in c["open_pairs"]:
            link.setdefault(a_, set()).add(b_)
            link.setdefault(b_, set()).add(a_)
        group, todo = set(rooms[:1]), list(rooms[:1])
        while todo:
            x_ = todo.pop()
            for y_ in link.get(x_, ()):
                if y_ in rooms and y_ not in group:
                    group.add(y_)
                    todo.append(y_)
        if len(rooms) >= 2 and group != set(rooms):
            r["flags"].append("部屋名が" + "・".join(rooms) + "（境の壁か建具の読み落とし？）")
        # region much larger than the printed 帖: a boundary was missed and rooms ran together
        if len(r.get("jo", [])) >= 2:
            r["flags"].append("広さの表示が" + "・".join(f"{j}帖" for j in r["jo"])
                              + "の2部屋分ある（境の読み落とし？）")
        if r.get("jo"):
            printed = sum(r["jo"]) * 1.62
            if r["area_m2"] > printed * c["area_over_ratio"] + c["area_over_margin_m2"]:
                r["flags"].append(f"表示の{'+'.join(str(j) for j in r['jo'])}帖（約{printed:.1f}㎡）より"
                                  f"広い{r['area_m2']:.1f}㎡（境の読み落とし？）")
        if two_floors(clean, rmask, c):
            r["flags"].append("床が2種類に見える（壁の見落とし？）")

    # check image
    out = cv2.addWeighted(img, 0.6, np.full_like(img, 255), 0.4, 0)
    rng = np.random.default_rng(3)
    overlay = out.copy()
    for r in regions:
        col = (0, 0, 255) if r["flags"] else tuple(int(v) for v in rng.integers(120, 230, 3))
        overlay[lab == r["label"]] = col
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    texts = []
    for r in regions:
        ys, xs = np.nonzero(lab == r["label"])
        # label at the pixel nearest the middle that is inside the region
        cx, cy = xs.mean(), ys.mean()
        i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        tx, ty = int(xs[i]) - 30, int(ys[i]) - 10
        label = f"{r['id']}:{'・'.join(dict.fromkeys(r['names'])) or '名前なし'} 建具{len(r['doors'])}"
        texts.append((label, (tx, ty), (0, 0, 0), 14))
        for j, fl in enumerate(r["flags"][:2]):
            texts.append(("※" + fl, (tx, ty + 16 + 14 * j), (0, 0, 200), 12))
    out = jp_text.put(out, texts)
    name = prefix.replace("\\", "/").split("/")[-1]
    out = jp_text.legend(out, f"{name}　区画の確認", [((0, 0, 255), "赤い区画＝読み落としの可能性（要確認）"),
                                                    ((180, 180, 180), "その他の区画（色は区別用）")], box=True)
    cv2.imencode(".png", out)[1].tofile(prefix + "_区画の確認.png")
    json.dump([{"id": r["id"], "names": r["names"], "name_read": r.get("name_read"),
                "doors": r["doors"], "area_m2": round(r["area_m2"], 2),
                "bbox_px": [int(v) for v in r["bbox"]], "flags": r["flags"]} for r in regions],
              open(prefix + "_区画.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    boundary_judgement(img, clean, det, M, lab, by_label, region_at, ops, kamachi, foot, cfg, prefix, name)
    print(f"{name}: regions={len(regions)} flagged={sum(bool(r['flags']) for r in regions)} 框={len(kamachi)}")
    for r in regions:
        print(f"  {r['id']:>2} {'・'.join(r['names']) or '名前なし':<14} {r['area_m2']:5.2f}㎡ "
              f"建具={r['doors']} {' / '.join(r['flags'])}")


if __name__ == "__main__":
    main()
