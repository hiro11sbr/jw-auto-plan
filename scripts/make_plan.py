"""Stage 2: wall offsets + columns -> .jww and check PNG.

usage: make_plan.py <config.json> <floor_walls.json> <floor_image.png> <out_prefix> [upper_walls.json]
"""
import json
import math
import sys
from collections import defaultdict

from pathlib import Path

import cv2
import ezjww
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
import jp_text  # noqa: E402
from place_fittings import WINDOWS as pf_WINDOWS  # noqa: E402

ENTRANCE_WORDS = ("玄関片開き", "玄関親子")
CLOSE_GRIDS, CLOSE_RATIO = 3, 0.8  # footprint gap closing (grid_detection outline_close_*)
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union

sys.stdout.reconfigure(encoding="utf-8")
EPS = 1e-6


# ---------------------------------------------------------------- walls
class Wall:
    def __init__(self, p1, p2, ext):
        (x1, y1), (x2, y2) = p1, p2
        self.h = abs(y1 - y2) < EPS
        if self.h:
            self.c, self.a, self.b = y1, min(x1, x2), max(x1, x2)
        else:
            self.c, self.a, self.b = x1, min(y1, y2), max(y1, y2)
        self.ext = ext

    def pt(self, t):
        return (t, self.c) if self.h else (self.c, t)

    def contains(self, p):
        along, across = (p[0], p[1]) if self.h else (p[1], p[0])
        return abs(across - self.c) < EPS and self.a - EPS <= along <= self.b + EPS

    def rect(self, d, cap_a=0.0, cap_b=0.0):
        """Band of half-width d; cap_a / cap_b extend it past each end (square cap)."""
        a, b = self.a - cap_a, self.b + cap_b
        if self.h:
            return box(a, self.c - d, b, self.c + d)
        return box(self.c - d, a, self.c + d, b)


def bare_lines(data, img_path, ops_path, cfg):
    """Interior walls with no gray fill along their centre and no fitting on them: a thin line
    between two walls taken for a fitting (プラン4 1F: the line between the shelves of a 物入 and
    the edge of a kitchen counter; プラン5 1F: the edges of a 玄関収納's hatching; 10/27). A real
    wall is a gray band; a fitting's gap is white but has a fitting. Returns the indices to drop."""
    from opening_detect import Mapper
    c = cfg["grid_detection"]
    lo, hi = data["gray_range"]
    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR).astype(int)
    H, W, _ = img.shape
    M = Mapper(data, cfg["input"]["grid_mm"])
    ops = []
    if Path(ops_path).exists():
        ops = json.load(open(ops_path, encoding="utf-8"))
        ops = [o for o in (ops["openings"] if isinstance(ops, dict) else ops) if not o.get("rejected")]
    out = set()
    for i, w in enumerate(data["walls"]):
        if w["type"] != "interior":
            continue
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        h = abs(y1 - y2) < EPS
        a, b = sorted((x1, x2) if h else (y1, y2))
        line = y1 if h else x1
        if any(o["horizontal"] == h and abs((o["p1_mm"][1] if h else o["p1_mm"][0]) - line) < 50
               and a - 50 <= min(o["p1_mm"][0 if h else 1], o["p2_mm"][0 if h else 1])
               and max(o["p1_mm"][0 if h else 1], o["p2_mm"][0 if h else 1]) <= b + 50 for o in ops):
            continue
        ok = n = 0
        for t in np.arange(a + 80, b - 80, 15):
            X, Y = (M.x(t), M.y(line)) if h else (M.x(line), M.y(t))
            X, Y = int(round(X)), int(round(Y))
            if not (1 <= X < W - 1 and 1 <= Y < H - 1):
                continue
            n += 1
            win = img[Y - 1:Y + 2, X - 1:X + 2].reshape(-1, 3)
            ok += sum(lo <= p.mean() <= hi and p.max() - p.min() <= 20 for p in win) >= 5
        if n and ok / n < c["bare_line_max_gray"]:
            out.add(i)
            print(f"灰色の帯のない線（建具もない）: 壁を外す {'横' if h else '縦'} {line} {a}〜{b}（灰{ok / n:.2f}）")
    return out


def load_walls(path, grid, drop=()):
    data = json.load(open(path, encoding="utf-8"))
    walls = []
    for i, w in enumerate(data["walls"]):
        if i in drop:
            continue
        # (walls fixed by the dimensions (dim_walls.py) keep their exact position: 間崩れ)
        snap = (lambda v: v) if w.get("exact") else (lambda v: round(v / grid) * grid)
        p1 = tuple(snap(v) for v in w["p1"])
        p2 = tuple(snap(v) for v in w["p2"])
        if p1 != p2 and (abs(p1[0] - p2[0]) < EPS or abs(p1[1] - p2[1]) < EPS):
            walls.append(Wall(p1, p2, w["type"] == "exterior"))
    return merge_collinear(walls), data


def drop_furniture_lines(walls, ops_path, cfg):
    """A UB door wider than any UB door is a furniture line read as a fitting (プラン3 1F x=5005:
    a counter's edge with a diagonal, 1688 wide, between two wall stubs; the user's correction
    has no wall there, 10/27). The wall between the crossing walls on either side is taken out
    (in place in walls). Returns the numbers of those openings."""
    if not Path(ops_path).exists():
        return set()
    max_w = cfg["placement"]["ub_door_max_mm"]
    out = set()
    data = json.load(open(ops_path, encoding="utf-8"))
    for o in (data["openings"] if isinstance(data, dict) else data):
        if o.get("rejected") or o["kind"] != "ub_door" or o["wall_type"] != "interior" or o["width_mm"] <= max_w:
            continue
        h = o["horizontal"]
        line = o["p1_mm"][1] if h else o["p1_mm"][0]
        a, b = sorted((o["p1_mm"][0], o["p2_mm"][0]) if h else (o["p1_mm"][1], o["p2_mm"][1]))
        cross = [w.c for w in walls if w.h != h and w.a - EPS <= line <= w.b + EPS]
        lo = max((c for c in cross if c <= a + 10), default=None)
        hi = min((c for c in cross if c >= b - 10), default=None)
        if lo is None or hi is None:
            continue
        for w in [w for w in walls if w.h == h and abs(w.c - line) < 1 and w.a < hi and w.b > lo]:
            walls.remove(w)
            for s, e in ((w.a, lo), (hi, w.b)):
                if e - s > EPS:
                    walls.append(Wall(w.pt(s), w.pt(e), w.ext))
        out.add(o["no"])
        print(f"家具の線（UBドアには広すぎる 幅{o['width_mm']}）: 壁を外す {'横' if h else '縦'} {line} {lo}〜{hi}")
    return out


def stair_foot_openings(walls, walls_path, cfg):
    """An opening beside the end of a stair's core line (the winders / the first steps) is the
    stair's open foot, not a door: the winders' diagonals fanning out from the stair wall's end
    were read as a 折戸's V (プラン5 1F x=910, UP; 10/27). Its wall part is taken out (in place
    in walls). Returns the numbers of those openings. A door under the stair stands beside the
    run of the core, not beyond its end."""
    ops_path = walls_path.replace("_walls.json", "_openings.json")
    edges_path = walls_path.replace("_walls.json", "_edges.json")
    if not (Path(ops_path).exists() and Path(edges_path).exists()):
        return set()
    det = json.load(open(walls_path, encoding="utf-8"))
    (_sx, _ix, nx0, _), (_sy, _iy, _, ny1) = det["fits"]["x"], det["fits"]["y"]
    grid, half = cfg["input"]["grid_mm"], cfg["input"]["half_grid_mm"]
    reach = cfg["placement"]["stair_foot_reach_mm"]
    cores = {}  # (h, line mm) -> [along mm of each core edge's ends]
    for e in json.load(open(edges_path, encoding="utf-8")):
        if not e.get("stair"):
            continue
        if e["h"]:
            line, t = (ny1 - e["line"]) * grid, (e["step"] - nx0) * grid
            cores.setdefault((True, line), []).extend((t, t + grid))
        else:
            line, t = (e["line"] - nx0) * grid, (ny1 - e["step"]) * grid
            cores.setdefault((False, line), []).extend((t, t - grid))
    data = json.load(open(ops_path, encoding="utf-8"))
    out = set()
    for o in (data["openings"] if isinstance(data, dict) else data):
        # (only a 折戸, the kind the winders' fan is taken for: a sliding door beside the top of a
        # stair is real, プラン3 2F)
        if o.get("rejected") or o["kind"] not in cfg["placement"]["stair_foot_kinds"] or o["wall_type"] != "interior":
            continue
        h = o["horizontal"]
        line = o["p1_mm"][1] if h else o["p1_mm"][0]
        a, b = sorted((o["p1_mm"][0], o["p2_mm"][0]) if h else (o["p1_mm"][1], o["p2_mm"][1]))
        for (ch, cl), ts in cores.items():
            c0, c1 = min(ts), max(ts)
            # parallel, within reach of the core line, and running past one end of the core
            # (プラン5: the core ends at about 5020, the opening runs 4674〜5430)
            if ch != h or not 0 < abs(cl - line) <= reach:
                continue
            if not (a < c0 - 100 < b or a < c1 + 100 < b):
                continue
            lo = math.floor((a + 10) / half) * half
            hi = math.ceil((b - 10) / half) * half
            for w in [w for w in walls if w.h == h and abs(w.c - line) < 1 and w.a < hi and w.b > lo]:
                walls.remove(w)
                for s, e in ((w.a, lo), (hi, w.b)):
                    if e - s > EPS:
                        walls.append(Wall(w.pt(s), w.pt(e), w.ext))
            out.add(o["no"])
            print(f"階段の上り口（建具なし・壁を開ける）: {'横' if h else '縦'} {line} {lo}〜{hi}（{o['kind']}）")
            break
    return out


def merge_collinear(walls):
    """Union overlapping/touching walls on the same line."""
    lines = defaultdict(list)
    for w in walls:
        lines[(w.h, w.c)].append(w)
    out = []
    for (h, c), lst in lines.items():
        lst.sort(key=lambda w: w.a)
        cur = lst[0]
        for w in lst[1:]:
            if w.a <= cur.b + EPS:
                cur.b = max(cur.b, w.b)
                cur.ext = cur.ext or w.ext
            else:
                out.append(cur)
                cur = w
        out.append(cur)
    return out


def nodes_of(walls):
    """Every endpoint and crossing, with its arms (directions of wall leaving the point)."""
    pts = set()
    for w in walls:
        pts.add(w.pt(w.a))
        pts.add(w.pt(w.b))
    hs = [w for w in walls if w.h]
    vs = [w for w in walls if not w.h]
    for h in hs:
        for v in vs:
            if h.a - EPS <= v.c <= h.b + EPS and v.a - EPS <= h.c <= v.b + EPS:
                pts.add((v.c, h.c))
    nodes = {}
    for p in pts:
        arms = set()
        for w in walls:
            if not w.contains(p):
                continue
            t = p[0] if w.h else p[1]
            if t > w.a + EPS:
                arms.add("W" if w.h else "S")
            if t < w.b - EPS:
                arms.add("E" if w.h else "N")
        nodes[p] = arms
    return nodes


def node_kind(arms):
    n = len(arms)
    if n == 1:
        return "end"
    if n == 2:
        return "straight" if arms in ({"E", "W"}, {"N", "S"}) else "L"
    return {3: "T", 4: "X"}.get(n, "none")


# ---------------------------------------------------------------- footprint / offsets
def footprint_of(walls):
    # outline of all enclosed areas, so an exterior wall misread as interior still closes the outline
    lines = [LineString([w.pt(w.a), w.pt(w.b)]) for w in walls]
    polys = list(polygonize(unary_union(lines)))
    if not polys:
        raise SystemExit("exterior walls do not form a closed outline")
    fp = unary_union(polys)
    if fp.geom_type == "MultiPolygon":
        fp = max(fp.geoms, key=lambda g: g.area)
    fp = Polygon(fp.exterior)
    # an outline that does not close all round (プラン4 1F 玄関/porch, プラン3 2F) leaves only
    # one block as the footprint: close gaps up to CLOSE_GRIDS x 455 and take that shape when it
    # is clearly larger (same rule as grid_detect.settle_outline)
    R = CLOSE_GRIDS * 455 / 2
    closed = unary_union(lines).buffer(R, join_style=2)
    if closed.geom_type == "MultiPolygon":
        closed = max(closed.geoms, key=lambda g: g.area)
    closed = Polygon(closed.exterior).buffer(-R, join_style=2)
    if closed.geom_type == "MultiPolygon":
        closed = max(closed.geoms, key=lambda g: g.area)
    if not closed.is_empty and fp.area < CLOSE_RATIO * closed.area:
        snap = lambda v: round(v / 227.5) * 227.5  # noqa: E731  back on the grid
        fp = Polygon([(snap(x), snap(y)) for x, y in closed.exterior.coords]).buffer(0)
    return fp


def classify_boundary(walls, fp):
    ring = fp.exterior
    for w in walls:
        seg = LineString([w.pt(w.a), w.pt(w.b)])
        w.ext = ring.buffer(1).contains(seg)


def rings_to_edges(geom):
    edges = []
    polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for p in polys:
        if p.is_empty:
            continue
        for ring in [p.exterior, *p.interiors]:
            cs = list(ring.coords)
            edges += [(cs[i], cs[i + 1]) for i in range(len(cs) - 1)]
    return edges


def simplify_edges(edges):
    """Merge consecutive collinear axis-aligned edges and drop zero-length ones."""
    hs, vs = defaultdict(list), defaultdict(list)
    for (x1, y1), (x2, y2) in edges:
        if abs(y1 - y2) < EPS and abs(x1 - x2) > EPS:
            hs[round(y1, 3)].append(sorted((x1, x2)))
        elif abs(x1 - x2) < EPS and abs(y1 - y2) > EPS:
            vs[round(x1, 3)].append(sorted((y1, y2)))
    out = []
    for groups, horiz in ((hs, True), (vs, False)):
        for c, spans in groups.items():
            spans.sort()
            cur = spans[0]
            for s in spans[1:]:
                if s[0] <= cur[1] + EPS:
                    cur = [cur[0], max(cur[1], s[1])]
                else:
                    out.append((horiz, c, cur))
                    cur = s
            out.append((horiz, c, cur))
    return [((a, c), (b, c)) if h else ((c, a), (c, b)) for h, c, (a, b) in out]


def drop_end_caps(edges, walls, nodes, d):
    """Remove the short cap edge across a free wall end (材料の小口線は描かない)."""
    caps = []
    for w in walls:
        for t in (w.a, w.b):
            p = w.pt(t)
            if node_kind(nodes.get(p, set())) == "end":
                caps.append((w.h, p))
    keep = []
    for e in edges:
        (x1, y1), (x2, y2) = e
        drop = False
        for h, (px, py) in caps:
            if h and abs(x1 - px) < EPS and abs(x2 - px) < EPS and \
                    min(y1, y2) >= py - d - EPS and max(y1, y2) <= py + d + EPS:
                drop = True
            if not h and abs(y1 - py) < EPS and abs(y2 - py) < EPS and \
                    min(x1, x2) >= px - d - EPS and max(x1, x2) <= px + d + EPS:
                drop = True
        if not drop:
            keep.append(e)
    return keep


def ring_faces(walls, fp, min_len=400):
    """{(horizontal, line, side)}: the outdoor faces of the exterior walls along the outline (a
    wall counts by its part on the outline: one wall can run on past it, プラン3 1F x=3185)."""
    ring = fp.exterior.buffer(1)
    out = set()
    for w in walls:
        part = LineString([w.pt(w.a), w.pt(w.b)]).intersection(ring)
        for seg in getattr(part, "geoms", [part]):
            if seg.is_empty or seg.length < min_len:
                continue
            m = seg.centroid
            for s in (1, -1):
                if not fp.contains(Point(m.x, m.y + s * 200) if w.h else Point(m.x + s * 200, m.y)):
                    out.add((w.h, round(w.c, 1), s))
    return out


def offset_lines(walls, fp, nodes, cfg, spaces=(), space_walls=None):
    """fp: the interior outline (the building outline less outside spaces). spaces: [(polygon,
    name)] outside spaces inside the building outline; space_walls: {wall: side of the outside
    space (+1/-1)} for walls between an outside space and the outdoors."""
    off = cfg["offsets_mm"]
    lay = cfg["layers"]["wall"]
    space_walls = space_walls or {}
    inner = [w for w in walls if not w.ext and w not in space_walls]
    result = []  # (edge, color, label)
    # room side: column face (52.5) and board face (65) for exterior inside + interior walls
    # Ends are extended by d (square cap): an outside corner closes cleanly instead of notching
    # at the wall center. At a free end the board face (65) wraps the end (reference drawing corrections
    # 2026-10-22); the column face (52.5) stops at the end, where the end column covers it.
    for d, color, label, cap_free in ((off["column_face"], lay["structure_color"], "柱面", False),
                                      (off["board_face_inside"], lay["finish_color"], "石膏ボード面", True)):
        rooms = fp.buffer(-d, join_style="mitre")
        if inner:
            rects = []
            for w in inner:
                caps = []
                for t in (w.a, w.b):
                    free = node_kind(nodes.get(w.pt(t), set())) == "end"
                    caps.append(d if (cap_free or not free) else 0.0)
                rects.append(w.rect(d, *caps))
            rooms = rooms.difference(unary_union(rects))
        edges = simplify_edges(rings_to_edges(rooms))
        if not cap_free:
            edges = drop_end_caps(edges, inner, nodes, d)
        result += [(e, color, label) for e in edges]
    # outside of exterior walls
    ext = off["exterior_side"]
    porch = off["porch_side"]
    outside = [(off["column_face"], porch["column_face"], lay["structure_color"], "柱面(外)"),
               (ext["sheathing"], porch["sheathing"], lay["structure_color"], "面材"),
               (ext["added_insulation"], porch["sheathing"], lay["structure_color"], "付加断熱"),
               (ext["ventilation"], porch["ventilation"], lay["structure_color"], "通気層"),
               (ext["exterior_face"], porch["exterior_face"], lay["finish_color"], "外壁面")]
    pieces = porch_walls(walls, fp)
    # outside-space walls (garage / porch / balcony perimeter inside the building outline)
    space_ws = [w for w in walls if w in space_walls]
    garages = unary_union([p for p, k in spaces if k == "ガレージ"]) if spaces else None
    faces = ring_faces(walls, fp)
    gb = off["exterior_side"]["ventilation"] + cfg["outside_space"]["garage_board_mm"]

    def face_d(h, c, mid, s, d, dp, label):
        # 付加断熱 rule (user 10/24): an outside-space wall in line with an exterior wall of the
        # outline keeps the added insulation on the face that carries on that wall's outdoor face
        # (no step in the finished face); otherwise none (60 thinner). Per face: プラン3 1F, the
        # garage face of the garage/porch wall carries on the 玄関 wall's garage face, the porch
        # face is thin (user's correction 10/27). In a garage the finish is the board (gb).
        full = (h, round(c, 1), s) in faces
        if label == "外壁面" and garages is not None and not garages.is_empty:
            pt = Point(mid, c + s * 200) if h else Point(c + s * 200, mid)
            if garages.contains(pt):  # (no added insulation: 79.5 + 12.5 = 92, プラン3 1F x=0)
                return gb if full else porch["ventilation"] + cfg["outside_space"]["garage_board_mm"]
        return d if full else dp

    done = []  # edges already drawn: a porch face with no insulation coincides with the sheathing
    for d, dp, color, label in outside:
        g = fp.buffer(d, join_style="mitre")
        if label == "外壁面" and garages is not None and not garages.is_empty:
            # inside a garage the last finish is 12.5 gypsum board, not the 15 mm siding
            g = g.difference(garages).union(fp.buffer(gb, join_style="mitre"))
        for h, c, a, b, s_porch in pieces:
            d_out = face_d(h, c, (a + b) / 2, -s_porch, d, dp, label)
            # a wall across the far end: stop on its outer face
            sgn = 1 if b > a else -1
            meet = next((w2 for w2 in walls if w2.h != h and abs(w2.c - b) < 1 and w2.a - 1 <= c <= w2.b + 1), None)
            d_end = face_d(meet.h, meet.c, c, sgn, d, dp, label) if meet is not None else None
            g = g.union(porch_rect(h, c, a, b, s_porch, d_out, dp, cap=label != "柱面(外)", d_end=d_end))
        for w in space_ws:
            mid = (w.a + w.b) / 2
            d_lo = face_d(w.h, w.c, mid, -1, d, dp, label)
            d_hi = face_d(w.h, w.c, mid, 1, d, dp, label)
            d_out = max(d_lo, d_hi)
            lo, hi = w.c - d_lo, w.c + d_hi
            t0, t1 = w.a - d_out, w.b + d_out
            g = g.union(box(t0, lo, t1, hi) if w.h else box(lo, t0, hi, t1))
        # the outer ring, plus the rings round outside spaces closed in by the building (a
        # garage's inside faces)
        keep = Polygon(g.exterior) if g.geom_type == "Polygon" else unary_union([Polygon(p.exterior) for p in g.geoms])
        edges = simplify_edges(rings_to_edges(keep) + (rings_to_edges(g) if spaces else []))
        if label == "柱面(外)":
            edges = [e for e in edges if not any(on_free_end(e, p) for p in pieces)]
        edges = [e for e in edges if not covered(e, done)]
        done += edges
        result += [(e, color, label) for e in edges]
    return result


def outside_spaces(walls, fp, names_path, det, cfg):
    """[(polygon, name)]: closed areas inside the building outline holding a label of an outside
    space (ガレージ・ポーチ・バルコニー)."""
    from opening_detect import Mapper
    kinds = cfg["outside_space"]["names"]
    M = Mapper(det, cfg["input"]["grid_mm"])
    pts = [(Point(M.xmm(n["x"]), M.ymm(n["y"])), n["name"])
           for n in json.load(open(names_path, encoding="utf-8")) if n.get("name") in kinds]
    lines = [LineString([w.pt(w.a), w.pt(w.b)]) for w in walls]
    faces = list(polygonize(unary_union(lines)))
    # a garage open on its shutter side (no wall there) closes on the building's extent read from
    # the dimension lines, which take the garage in (プラン3 1F: a 2730 shutter left the garage out
    # of the outline, so its windows faced the garage and its 窓台 rule did not apply, 10/27)
    ext_faces = []
    dims_path = names_path.replace("_部屋名.json", "_寸法.json")
    if Path(dims_path).exists():
        ex = json.load(open(dims_path, encoding="utf-8")).get("extent_px")
        if ex:
            (xa, xb), (ya, yb) = sorted((M.xmm(ex[0]), M.xmm(ex[2]))), sorted((M.ymm(ex[1]), M.ymm(ex[3])))
            g = cfg["input"]["half_grid_mm"]  # (the ticks read 40 mm off: back on the grid)
            rect = box(*(round(v / g) * g for v in (xa, ya, xb, yb)))
            ext_faces = list(polygonize(unary_union(lines + [rect.exterior])))
    out = []
    for pt, name in pts:
        if fp.contains(pt):
            face = next((f for f in faces if f.contains(pt)), None)
        elif name in cfg["outside_space"]["garage_names"]:
            # (only a face against the building: a garage, not an open yard inside the extent)
            face = next((f for f in ext_faces if f.contains(pt) and f.buffer(1).intersects(fp)), None)
        else:
            continue
        if face is not None and not any(face.equals(f) for f, _n in out):
            out.append((face, name))
    return out


def porch_walls(walls, fp):
    """Walls that run on outside the building outline, e.g. the porch side wall (ポーチ壁):
    [(horizontal, centre, attach t, free end t, porch side +1/-1)]. The porch is the side that
    faces another part of the building."""
    ring = fp.buffer(1)
    out = []
    for w in walls:
        rest = LineString([w.pt(w.a), w.pt(w.b)]).difference(ring)
        for seg in getattr(rest, "geoms", [rest]):
            if seg.is_empty or seg.length < 400:
                continue
            (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
            t1, t2 = (x1, x2) if w.h else (y1, y2)
            p1, p2 = Point(x1, y1), Point(x2, y2)
            if p1.distance(fp) < 5:
                attach, free = t1, t2
            elif p2.distance(fp) < 5:
                attach, free = t2, t1
            else:
                continue
            attach = round(attach / 227.5) * 227.5  # back to the wall-centre crossing
            mid = (attach + free) / 2

            def faces_building(s, w=w, mid=mid):
                ray = (LineString([(mid, w.c + s * 200), (mid, w.c + s * 6000)]) if w.h else
                       LineString([(w.c + s * 200, mid), (w.c + s * 6000, mid)]))
                return ray.intersects(fp)

            out.append((w.h, w.c, attach, free, 1 if faces_building(1) else -1))
    return out


def porch_rect(h, c, a, b, s_porch, d_out, d_porch, cap=True, d_end=None):
    """Layer band of a porch wall: full exterior layers (d_out) on the outer face, the thinner
    porch build-up (d_porch, no added insulation) on the porch face, and the free end wrapped by
    the outer layers (user's correction, 2026-10-22). d_end: an end that meets another wall stops
    on that wall's face instead (プラン3 1F, user's correction 10/27)."""
    lo_c, hi_c = (c - d_porch, c + d_out) if s_porch < 0 else (c - d_out, c + d_porch)
    step = (d_out if d_end is None else d_end) if cap else 0.0
    t0, t1 = (b - step, a) if b < a else (a, b + step)
    return box(t0, lo_c, t1, hi_c) if h else box(lo_c, t0, hi_c, t1)


def on_free_end(e, piece):
    """Is edge e the line across the free end of a porch wall (the end column covers it)?"""
    h, c, a, b, _s = piece
    (x1, y1), (x2, y2) = e
    if h:
        return abs(x1 - b) < EPS and abs(x2 - b) < EPS
    return abs(y1 - b) < EPS and abs(y2 - b) < EPS


def covered(e, edges):
    """Is edge e lying on top of (inside) one of the edges?"""
    (x1, y1), (x2, y2) = e
    for (u1, v1), (u2, v2) in edges:
        if abs(y1 - y2) < EPS and abs(v1 - v2) < EPS and abs(y1 - v1) < EPS:
            if min(u1, u2) - EPS <= min(x1, x2) and max(x1, x2) <= max(u1, u2) + EPS:
                return True
        if abs(x1 - x2) < EPS and abs(u1 - u2) < EPS and abs(x1 - u1) < EPS:
            if min(v1, v2) - EPS <= min(y1, y2) and max(y1, y2) <= max(v1, v2) + EPS:
                return True
    return False


# ---------------------------------------------------------------- columns
def upper_positions(upper_walls, line_h, c):
    """Positions along a lower wall line where the upper floor has a wall crossing or ending."""
    if not upper_walls:
        return None
    pos = set()
    for u in upper_walls:
        if u.h != line_h:
            # perpendicular upper wall: its center crosses this line if c lies in its span
            if u.a - EPS <= c <= u.b + EPS:
                pos.add(u.c)
        elif abs(u.c - c) < EPS:
            pos.update((u.a, u.b))
    return pos


def place_columns(walls, nodes, cfg, upper_walls, fittings=()):
    """fittings: [(horizontal, line, lo, hi, part word)] frame spans of placed doors/windows. A column never
    overlaps a fitting; where a fitting leaves a gap over the pitch, columns go to its sides
    (the grid lines just outside the frame) and the rest is filled as usual (user, 2026-10-22)."""
    pitch = cfg["column"]["pitch_mm"]
    hc = cfg["column"]["size_mm"] / 2
    grid = cfg["input"]["grid_mm"]
    cols = {}  # point -> reason
    entrance = [(fh, line, lo, hi) for fh, line, lo, hi, word in fittings if word in ENTRANCE_WORDS]
    no_sides = set()
    for p, arms in nodes.items():
        k = node_kind(arms)
        if k in ("end", "L", "T", "X"):
            # no column on the entrance door, not even at a wall junction: the columns there stand
            # at the 1820 pitch clear of the door, the irregular rest is done by hand (user 10/23)
            if any((p[0] if fh else p[1]) + hc > lo + EPS and (p[0] if fh else p[1]) - hc < hi - EPS
                   and abs((p[1] if fh else p[0]) - line) < 1 for fh, line, lo, hi in entrance):
                continue
            # nor on a fitting's frame (プラン3 1F: the W1454's pocket runs past the stub wall at
            # 5915,8645; the user's correction has no column there, 10/27)
            hit = [i for i, (fh, line, lo, hi, _w) in enumerate(fittings)
                   if (p[0] if fh else p[1]) + hc > lo + EPS and (p[0] if fh else p[1]) - hc < hi - EPS
                   and abs((p[1] if fh else p[0]) - line) < 1]
            if hit:
                print(f"建具に被る柱を置かない: {p} ({k})")
                no_sides.update(hit)
                continue
            cols[p] = k
    defaults = []  # remainder placed on the default side without upper-floor evidence
    for w in walls:
        spans = [(lo, hi) for fh, line, lo, hi, _word in fittings if fh == w.h and abs(line - w.c) < 1]
        # no side columns for the entrance door, a garage shutter, or a fitting that took a
        # junction column's place (not needed there, user 10/27)
        side_spans = [(lo, hi) for i, (fh, line, lo, hi, word) in enumerate(fittings)
                      if fh == w.h and abs(line - w.c) < 1 and word not in ENTRANCE_WORDS + ("シャッター",)
                      and i not in no_sides]

        def clear(t, spans=spans):
            return all(t + hc <= lo + EPS or t - hc >= hi - EPS for lo, hi in spans)

        stops = sorted({(p[0] if w.h else p[1]) for p in cols if w.contains(p)})
        for s, e in zip(stops, stops[1:]):
            if e - s > pitch + EPS:
                for lo, hi in side_spans:
                    if s < lo and hi < e:
                        for t in (math.floor((lo - hc + EPS) / grid) * grid, math.ceil((hi + hc - EPS) / grid) * grid):
                            if s + EPS < t < e - EPS:
                                cols.setdefault(w.pt(t), "fitting")
        stops = sorted({(p[0] if w.h else p[1]) for p in cols if w.contains(p)})
        for s, e in zip(stops, stops[1:]):
            L = e - s
            if L <= pitch + EPS:
                continue
            n = int((L + EPS) // pitch)
            r = L - n * pitch
            from_a = [t for t in (s + i * pitch for i in range(1, n + 1)) if t < e - EPS and clear(t)]
            from_b = [t for t in (e - i * pitch for i in range(1, n + 1)) if t > s + EPS and clear(t)]
            choice, marked = from_a, False
            if r > EPS:
                up = upper_positions(upper_walls, w.h, w.c)
                score_a = sum(any(abs(t - u) < EPS for u in up) for t in from_a) if up else 0
                score_b = sum(any(abs(t - u) < EPS for u in up) for t in from_b) if up else 0
                if score_b > score_a:
                    choice = from_b
                elif score_a == score_b:
                    # no upper-floor evidence: the side whose columns stand on the 910 grid
                    # (columns go on the 910 grid; user 10/27, プラン3 1F north wall 10920)
                    on910 = lambda ts: sum(abs(t / (2 * grid) - round(t / (2 * grid))) < 1e-6 for t in ts)  # noqa: E731
                    if len(from_b) == len(from_a) and on910(from_b) > on910(from_a):
                        choice = from_b
                    marked = True
            for t in choice:
                cols.setdefault(w.pt(t), "pitch")
            if marked:
                defaults.append(w.pt((s + e) / 2))
    return cols, defaults


# ---------------------------------------------------------------- output
def write_jww(path, walls, offsets, cols, cfg, center, placements=(), lib=None, extras=()):
    S = cfg["layers"]["scale"]
    L = cfg["layers"]
    d = ezjww.new(paper_size=3)
    g = d.options["layer_groups"][L["group"]]
    g["name"] = L["group_name"]
    g["scale"] = S
    for i, n in enumerate(L["names"]):
        g["layers"][i]["name"] = n
    msp = d.modelspace()
    cx, cy = center

    def line(p, q, layer, color, style):
        msp.add_line(((p[0] - cx) / S, (p[1] - cy) / S), ((q[0] - cx) / S, (q[1] - cy) / S),
                     jwwattribs={"layer": layer, "pen_color": color, "pen_style": style})

    wl, cl, fr = L["wall"], L["center"], L["frame"]
    for w in walls:
        line(w.pt(w.a), w.pt(w.b), cl["layer"], cl["color"], cl["style"])
    for (p, q), color, _ in offsets:
        line(p, q, wl["layer"], color, wl["style"])
    for (p, q), layer, color, style in extras:  # wall returns and 窓台 at windows
        line(p, q, layer, color, style)
    h = cfg["column"]["size_mm"] / 2
    for (x, y) in cols:
        c4 = [(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)]
        for i in range(4):
            line(c4[i], c4[(i + 1) % 4], wl["layer"], wl["structure_color"], wl["style"])
        line(c4[0], c4[2], wl["layer"], wl["structure_color"], wl["style"])
        line(c4[1], c4[3], wl["layer"], wl["structure_color"], wl["style"])
    # fittings: one block definition per part used, one reference per opening
    if placements:
        by_name = {v[0]: v[1] for v in (lib or {}).values()}
        for name in dict.fromkeys(pl["block"] for pl in placements):
            d.add_block(name, by_name[name])
        for pl in placements:
            msp.add_block_ref(pl["block"], ((pl["pos"][0] - cx) / S, (pl["pos"][1] - cy) / S),
                              scale_x=float(pl["sx"]), scale_y=float(pl["sy"]), rotation=float(pl["rot"]),
                              jwwattribs={"layer": wl["layer"]})
    # print alignment marks (paper mm)
    pm = cfg["print_marks_mm"]

    def pline(p, q):
        msp.add_line(p, q, jwwattribs={"layer": fr["layer"], "pen_color": fr["color"], "pen_style": fr["style"]})

    for X, Y in pm["l_corners"]:
        for sx in (1, -1):
            for sy in (1, -1):
                pline((sx * X, sy * Y), (sx * (X - pm["arm"]), sy * Y))
                pline((sx * X, sy * Y), (sx * X, sy * (Y - pm["arm"])))
    W, H = pm["outline"]
    pts = [(-W, -H), (W, -H), (W, H), (-W, H)]
    for i in range(4):
        pline(pts[i], pts[(i + 1) % 4])
    d.saveas(path)


def frame_crossings(pl, lib, S, h, c_):
    """Along-wall positions where the placed part's frame lines cross the wall face line at c_
    (lines across the wall that reach that face): its jambs, and on a 片引戸 the 中方立 on the
    face it reaches."""
    ents = [e for e in lib[pl["block"]][1] if e["type"] == "LINE"]
    out = []
    for (x1, y1), (x2, y2) in block_lines_mm(ents, pl, S):
        t1, t2, u1, u2 = (x1, x2, y1, y2) if h else (y1, y2, x1, x2)
        if abs(t1 - t2) < 0.5 and min(u1, u2) - 0.5 <= c_ <= max(u1, u2) + 0.5:
            out.append((t1 + t2) / 2)
    return out


def cut_openings(offsets, placements, cfg, reach=160, lib=None):
    """Wall face lines do not run through a fitting: cut them over each placed part's span.
    At a window the room-side board face (65) stops 12.5 inside the frame, where it turns back
    to the window (the 窓台 reaches 12.5 into the wall each side). At a door each face line stops
    at the first frame line it meets (a 片引戸's pocket face runs on to the 中方立, the other face
    stops at the pocket-end jamb), and where both board faces of a wall end on the same frame
    face a board line closes the wall end there (user's correction, プラン3 1F 10/26)."""
    board = cfg["offsets_mm"]["board_face_inside"]
    embed = cfg["placement"]["window_sill"]["embed_mm"]
    S = cfg["layers"]["scale"]
    out, ends = [], []
    for (p, q), color, label in offsets:
        h = abs(p[1] - q[1]) < EPS
        segs = [(p, q)]
        for pl in placements:
            if pl["horizontal"] != h:
                continue
            c_ = p[1] if h else p[0]
            if abs(c_ - pl["line"]) > reach:
                continue
            lo, hi = pl["span"]
            if pl["word"] in pf_WINDOWS and abs((pl["line"] - c_) * pl["outward"] - board) < 0.5:
                lo, hi = lo + embed, hi - embed
            cross = []
            if pl["word"] not in pf_WINDOWS and lib and pl["block"] in lib:
                cross = [t for t in frame_crossings(pl, lib, S, h, c_) if lo - 1 <= t <= hi + 1]
            if len(cross) >= 2 and max(cross) - min(cross) > 100:
                lo, hi = min(cross), max(cross)
            elif pl["word"] == "片引戸":
                # the pocket half is inside the wall: the wall face lines run on over it up to the
                # 中方立 (middle of the part); only the doorway half is cut (user 10/24). The part's
                # pocket is at its local right, model high end when sx = +1
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if pl["sx"] == 1 else (mid, hi)
            nxt = []
            for (a, b) in segs:
                ta, tb = sorted((a[0], b[0]) if h else (a[1], b[1]))
                if tb <= lo or ta >= hi:
                    nxt.append((a, b))
                    continue
                for s, e in ((ta, lo), (hi, tb)):
                    if e - s > 1:
                        nxt.append(((s, c_), (e, c_)) if h else ((c_, s), (c_, e)))
                        if pl["word"] not in pf_WINDOWS and abs(abs(c_ - pl["line"]) - board) < 0.5:
                            ends.append((id(pl), h, pl["line"], e if e == lo else s, c_, color, label))
            segs = nxt
        out += [(s, color, label) for s in segs]
    done = set()
    for i, (k, h, line, t, c1, color, label) in enumerate(ends):
        for k2, h2, _l, t2, c2, _c, _lb in ends[i + 1:]:
            if k2 == k and h2 == h and abs(t2 - t) < 0.5 and (c1 - line) * (c2 - line) < 0:
                key = (k, round(t, 1))
                if key not in done:
                    done.add(key)
                    out.append((((t, c1), (t, c2)) if h else ((c1, t), (c2, t)), color, label))
    return out


INTERIOR_DOORS = ("片開き", "片引戸", "引違い戸", "両開き", "折戸")


def jamb_closures(offsets, placements, cols, walls, cfg):
    """How the wall lines end at an interior door's jambs (例1 1F 修正完了, user 10/30):
    - board face (65): a line across the wall at the frame's end closes it (also where the end
      meets a crossing wall's board face, which then runs straight across);
    - column face (52.5): stops 12.5 short of the frame's end and is closed by a line across; where
      a column's box ends there, the box is the end and nothing runs past it toward the frame;
    - a crossing wall's column face runs on through a junction left without a column.
    A 片引戸's pocket end keeps its own closing (cut_openings)."""
    off = cfg["offsets_mm"]
    col, board = off["column_face"], off["board_face_inside"]
    hc = cfg["column"]["size_mm"] / 2
    gap = board - col  # 12.5
    lines = [[p, q, c, lb] for (p, q), c, lb in offsets]
    lay = cfg["layers"]["wall"]
    finish, struct = lay["finish_color"], lay["structure_color"]
    col_pts = list(cols)

    def along(pt, h):
        return pt[0] if h else pt[1]

    def acr(pt, h):
        return pt[1] if h else pt[0]

    def mk(h, t1, t2, c):
        """A segment from t1 to t2 along a wall of direction h, at c across."""
        return ((t1, c), (t2, c)) if h else ((c, t1), (c, t2))

    def across_line(h, t, c1, c2):
        """A segment across a wall of direction h at t along it, from c1 to c2."""
        return ((t, c1), (t, c2)) if h else ((c1, t), (c2, t))

    def faces_at(h, L, dist, label, t, d):
        """Lines of `label` running along the wall at L±dist that reach t from the wall side."""
        out = []
        for ln in lines:
            p, q, _c, lb = ln
            if lb != label or abs(acr(p, h) - acr(q, h)) > 0.5 or abs(abs(acr(p, h) - L) - dist) > 0.5:
                continue
            a_, b_ = sorted((along(p, h), along(q, h)))
            if (d == -1 and a_ < t + 0.5 and b_ > t - 200) or (d == 1 and b_ > t - 0.5 and a_ < t + 200):
                out.append(ln)
        return out

    def run_end(h, L, t, d):
        """Where the walls on line L, chained end to end, stop going from t in direction d."""
        pos = t
        moved = True
        while moved:
            moved = False
            for w2 in walls:
                if w2.h == h and abs(w2.c - L) < EPS and w2.a - 1 <= pos <= w2.b + 1:
                    far = w2.b if d == 1 else w2.a
                    if (far - pos) * d > 1:
                        pos, moved = far, True
        return pos

    added = []
    for pl in placements:
        if pl["word"] not in INTERIOR_DOORS:
            continue
        h, L = pl["horizontal"], pl["line"]
        lo, hi = pl["span"]
        w = next((w for w in walls if w.h == h and abs(w.c - L) < EPS and w.a - 1 <= lo and hi <= w.b + 1), None)
        if w is None or w.ext:
            continue
        for t, d in ((lo, -1), (hi, 1)):
            # the wall goes on beyond this end (d: direction from the frame toward the wall)
            if not (w.a + 1 < t if d == -1 else t < w.b - 1):
                continue
            # a door centred in its bay (778 in 910: 1 mm each side) ends 1 short of the crossing
            # wall's board face: the wall's lines end at that face (例1 2F x=4485)
            t0, tee = t, False
            for w2 in walls:
                f = w2.c - d * board  # the crossing wall's face toward the frame
                if w2.h != h and w2.a - 1 <= L <= w2.b + 1 and abs(f - t) <= 2:
                    t = f
                    tee = tee or abs(w2.a - L) < 1 or abs(w2.b - L) < 1  # it ends on this wall (T / L)
            if t != t0:
                # the board faces along the wall that stopped at the frame's end stop at that face too
                for ln in lines:
                    p, q, _c, lb = ln
                    if lb != "石膏ボード面" or abs(acr(p, h) - acr(q, h)) > 0.5 or abs(abs(acr(p, h) - L) - board) > 0.5:
                        continue
                    if abs(along(p, h) - t0) < 0.5:
                        ln[0] = mk(h, t, t, acr(p, h))[0]
                    elif abs(along(q, h) - t0) < 0.5:
                        ln[1] = mk(h, t, t, acr(q, h))[1]
            covered = any(lb == "石膏ボード面" and abs(along(p, h) - t) < 0.5 and abs(along(q, h) - t) < 0.5
                          and min(acr(p, h), acr(q, h)) <= L - board + 0.5
                          and max(acr(p, h), acr(q, h)) >= L + board - 0.5 for p, q, _c, lb in lines)
            if pl["word"] == "片引戸":
                # its pocket runs inside the wall: where a face runs on across the frame's end (the
                # pocket end) or cut_openings already closed the wall (at the 中方立), leave it as is
                runs_on = any(lb == "石膏ボード面" and abs(acr(p, h) - acr(q, h)) < 0.5
                              and abs(abs(acr(p, h) - L) - board) < 0.5
                              and min(along(p, h), along(q, h)) < t - 1 < t + 1 < max(along(p, h), along(q, h))
                              for p, q, _c, lb in lines)
                if covered or runs_on:
                    continue
            t2 = t + d * gap
            has_col = any(abs(acr(cp, h) - L) < 1 and abs(along(cp, h) - (t2 + d * hc)) < 1 for cp in col_pts)
            # board face: a line across the wall at the frame's end. Where a column stands right at
            # the jamb its box closes the wall instead, unless a crossing wall's board face meets the
            # end there (that face then runs straight across: 例1 1F x=1820 at 2795 / 3575)
            # (only where the wall beyond is no more than that junction: it ends there (an L) or the
            # next frame starts on its far side (例1 1F x=1820, two sliding doors either side of the
            # 5460 T). Where it runs on past it, the column's box closes it: 例1 1F x=3640 at 1820)
            reach_end = run_end(h, L, t, d)
            nxt = [s for pl2 in placements if pl2 is not pl and pl2["horizontal"] == h and abs(pl2["line"] - L) < 1
                   for s in pl2["span"] if (s - t) * d > 0]
            beyond = min([abs(reach_end - t)] + [abs(s - t) for s in nxt])
            # (a crossing wall that ends on this wall at the frame's end: its face runs straight
            # across whatever lies beyond: 例1 1F x=1820 at 3575 / 6305, user 10/30)
            crossing = (beyond <= 2 * board + 1 or tee) and any(
                w2.h != h and abs(abs(w2.c - t) - board) < 0.5 and w2.a - 1 <= L <= w2.b + 1 for w2 in walls)
            if not covered and (crossing or not has_col):
                added.append((across_line(h, t, L - board, L + board), finish, "石膏ボード面"))
            elif covered and has_col and not crossing:
                # cut_openings closed it (both faces end on the frame), but the column there does
                lines = [ln for ln in lines if not (
                    ln[3] == "石膏ボード面" and abs(along(ln[0], h) - t) < 0.5 and abs(along(ln[1], h) - t) < 0.5
                    and abs(abs(acr(ln[0], h) - acr(ln[1], h)) - 2 * board) < 1)]
            # column face: 12.5 short of the frame end, closed across unless a column box ends there
            for ln in faces_at(h, L, col, "柱面", t, d):
                p, q, _c, _lb = ln
                a_, b_ = sorted((along(p, h), along(q, h)))
                if (d == -1 and b_ > hi + 0.5) or (d == 1 and a_ < lo - 0.5):
                    continue  # runs on past the opening: not this end's line
                a_, b_ = (a_, min(b_, t2)) if d == -1 else (max(a_, t2), b_)
                ln[0], ln[1] = mk(h, a_, b_, acr(p, h))
            lines = [ln for ln in lines if math.dist(ln[0], ln[1]) > 0.5]
            if not has_col:
                added.append((across_line(h, t2, L - col, L + col), struct, "柱面"))
    out = [((p, q), c, lb) for p, q, c, lb in lines] + added
    # a crossing wall's column face runs on through a junction without a column: fill a 105 gap
    # between two collinear column-face pieces when no column stands there
    fills = []
    face = [((p, q), c) for (p, q), c, lb in out if lb == "柱面"]
    for (p1, q1), c1 in face:
        for (p2, q2), _c2 in face:
            for hh in (True, False):
                if not (abs(acr(p1, hh) - acr(q1, hh)) < 0.5 and abs(acr(p2, hh) - acr(q2, hh)) < 0.5
                        and abs(acr(p1, hh) - acr(p2, hh)) < 0.5):
                    continue
                e1 = max(along(p1, hh), along(q1, hh))
                s2 = min(along(p2, hh), along(q2, hh))
                if abs(s2 - e1 - 2 * hc) < 0.5:
                    mid = (e1 + s2) / 2
                    cc = acr(p1, hh)
                    if any(abs(along(cp, hh) - mid) < 1 and abs(abs(acr(cp, hh) - cc) - hc) < 1 for cp in col_pts):
                        continue
                    if any(w.h != hh and abs(w.c - mid) < 1 and w.a - 1 <= cc <= w.b + 1 for w in walls):
                        fills.append((((e1, cc), (s2, cc)) if hh else ((cc, e1), (cc, s2)), c1, "柱面"))
    return out + fills


def koshikabe_lines(kos, walls, fp, cfg, voids=()):
    """腰壁 from koshikabe_detect.py: 笠木 band ±75 from the centre, the centre line, and at a
    wall junction short ticks joining the 笠木 to the wall's board face. A free end: the 笠木
    ends where the drawing shows it (snapped to grid ±75 when within 20 mm); a junction: the 笠木
    stops at the column face (user's correction 2026-10-22)."""
    kc = cfg["koshikabe"]
    k, col, board = kc["kasagi_mm"], cfg["offsets_mm"]["column_face"], cfg["offsets_mm"]["board_face_inside"]
    cl = cfg["layers"]["center"]
    sl = (kc["layer"], kc["color"], kc["style"])
    grid = cfg["input"]["half_grid_mm"]
    out, drawn, lshape = [], [], []

    def side_by_side(p, q):
        """Two 腰壁 running side by side close together over most of their length: a 笠木 is one
        150 band, so this is a wide hatched area (a window's 格子 or a storage beside the 玄関,
        プラン3 1F: drawn as two 笠木 like a continuous 窓台, user 10/27)."""
        if p is q or p["horizontal"] != q["horizontal"] or abs(p["line_mm"] - q["line_mm"]) > kc["pair_max_gap_mm"]:
            return False
        ov = min(p["b_mm"], q["b_mm"]) - max(p["a_mm"], q["a_mm"])
        return ov >= 0.7 * min(p["b_mm"] - p["a_mm"], q["b_mm"] - q["a_mm"])
    paired = [kb for kb in kos if kb.get("source") != "階段まわり"
              and any(side_by_side(kb, o) for o in kos if o.get("source") != "階段まわり")]
    for kb in paired:
        print(f"腰壁ではない（2本並んだハッチング）: {'横' if kb['horizontal'] else '縦'} {kb['line_mm']} {round(kb['a_mm'])}〜{round(kb['b_mm'])}")
    kos = [kb for kb in kos if kb not in paired]
    long_ = [kb for kb in kos if kb["b_mm"] - kb["a_mm"] >= kc["min_len_mm"]]

    def ends_of(kb):
        h, c, a, b = kb["horizontal"], kb["line_mm"], kb["a_mm"], kb["b_mm"]
        return [(t, c) if h else (c, t) for t in (a, b)]

    def joins_long(kb):
        """A short piece turning a corner off the end of a long 腰壁: the L of a stair's 笠木
        (the spine bends round the foot of the flight to the wall, プラン2 2F, user 10/24)."""
        return any(lg["horizontal"] != kb["horizontal"] and
                   min(math.dist(p, q) for p in ends_of(kb) for q in ends_of(lg)) <= kc["corner_join_mm"]
                   for lg in long_)
    for kb in kos:
        h, c, a, b = kb["horizontal"], kb["line_mm"], kb["a_mm"], kb["b_mm"]
        mid_pt = ((a + b) / 2, c) if h else (c, (a + b) / 2)
        # the edge between a 吹抜 and the hall is a 腰壁 (user 10/27), short as it may be
        # (プラン3 2F: 425 from the wall to the stair)
        by_void = any(math.dist(mid_pt, v) <= kc["void_reach_mm"] for v in voids)
        if b - a < kc["min_len_mm"] and not (b - a >= kc["corner_min_len_mm"] and (joins_long(kb) or by_void)):
            continue
        mid = Point(((a + b) / 2, c) if h else (c, (a + b) / 2))
        if not fp.buffer(-1).contains(mid):
            continue

        def junction(t, h=h, c=c):
            """Wall centre of a wall this end meets (crossing it, or ending on the same line)."""
            best = None
            for w in walls:
                if w.h != h and w.a - EPS <= c <= w.b + EPS and abs(w.c - t) < 150:
                    best = w.c if best is None or abs(w.c - t) < abs(best - t) else best
                if w.h == h and abs(w.c - c) < EPS:
                    for e in (w.a, w.b):
                        if abs(e - t) < 150:
                            best = e if best is None or abs(e - t) < abs(best - t) else best
            return best

        # a stair's 笠木 L (a piece corner-joined to another): drawn as one continuous L, and at a
        # wall it runs 65 past the grid line, as low walls stand on short columns (user 10/25;
        # the kitchen 腰壁 is the exception and keeps stopping at the column face)
        partners = [lg for lg in kos if lg is not kb and lg["horizontal"] != h and
                    lg["b_mm"] - lg["a_mm"] >= kc["corner_min_len_mm"] and
                    min(math.dist(p, q) for p in ends_of(kb) for q in ends_of(lg)) <= kc["corner_join_mm"]
                    and (kb in long_ or lg in long_)]
        ends = []
        for t, sgn in ((a, -1), (b, 1)):  # sgn: direction from the 腰壁 toward this end
            j = junction(t)
            corner = next((lg for lg in partners if abs(lg["line_mm"] - t) <= kc["corner_join_mm"]), None)
            if corner is not None:
                cc = corner["line_mm"] + sgn * k  # out to the partner's outer 笠木 face
                ends.append((cc, corner["line_mm"], False))
            elif j is not None and partners:
                ends.append((j + sgn * kc["stair_past_grid_mm"], j, True))
            elif j is not None:
                ends.append((j - sgn * col, j, True))  # (笠木 end, centre-line end, at a wall)
            else:
                g = round((t - sgn * k) / grid) * grid + sgn * k
                kt = g if abs(g - t) <= 20 else round(t / 5) * 5
                ends.append((kt, kt - sgn * k, False))
        (ka, ca, ja), (kb_, cb, jb) = ends
        if partners:
            # collect the band; the L is drawn as one outline after the loop
            lshape.append((box(ka, c - k, kb_, c + k) if h else box(c - k, ka, c + k, kb_)))
            out.append((((ca, c), (cb, c)) if h else ((c, ca), (c, cb)), cl["layer"], cl["color"], cl["style"]))
            drawn.append({"horizontal": h, "line_mm": c, "a_mm": ka, "b_mm": kb_})
            continue

        def seg(t1, t2, across):
            return ((t1, across), (t2, across)) if h else ((across, t1), (across, t2))

        def cross(t, u1, u2):
            return ((t, u1), (t, u2)) if h else ((u1, t), (u2, t))

        out.append((seg(ka, kb_, c - k), *sl))
        out.append((seg(ka, kb_, c + k), *sl))
        for kt, at_wall in ((ka, ja), (kb_, jb)):
            if at_wall:  # ticks from the 笠木 to the wall's board face
                out.append((cross(kt, c - k, c - board), *sl))
                out.append((cross(kt, c + board, c + k), *sl))
            else:
                out.append((cross(kt, c - k, c + k), *sl))
        out.append((seg(ca, cb, c), cl["layer"], cl["color"], cl["style"]))
        drawn.append({"horizontal": h, "line_mm": c, "a_mm": ka, "b_mm": kb_})
    if lshape:
        u = unary_union(lshape)
        for e in simplify_edges(rings_to_edges(u)):
            out.append((e, *sl))
    return out, drawn


def garage_door_window(pl, spaces, cfg):
    """A window standing in for a door to the garage: in the insulated wall between a room and
    the garage, opening like a door (縦すべり), 600 wide or more. It gets no 窓台 (user 10/27,
    プラン3 1F 土間収納→ガレージ)."""
    gs = [p for p, k in spaces if k in cfg["outside_space"]["garage_names"]]
    if not gs or "縦すべり" not in pl["word"] or pl["width"] < cfg["placement"]["window_sill"]["garage_door_min_mm"]:
        return False
    mid = (pl["span"][0] + pl["span"][1]) / 2
    t = pl["line"] + pl["outward"] * 300
    probe = Point(mid, t) if pl["horizontal"] else Point(t, mid)
    return any(g.contains(probe) for g in gs)


def entrance_casing(offsets, placements, cfg):
    """額縁 at the entrance door's jambs on the room side (user's correction プラン3 1F, 10/27;
    provisional sizes, details to come): the wall stops 21 short of the frame and is closed
    there (structure line from the added insulation's face to the column face); the casing, 21
    thick, runs from the frame into the room 13 proud of the board face; the outer lines of the
    wall stop with it. Returns (offsets, extra lines)."""
    off = cfg["offsets_mm"]
    lay = cfg["layers"]["wall"]
    ec = cfg["placement"]["entrance_casing"]
    t_, proud, reach = ec["thickness_mm"], ec["proud_mm"], ec["frame_reach_mm"]
    col, board = off["column_face"], off["board_face_inside"]
    ins = off["exterior_side"]["added_insulation"]
    extras = []
    for pl in placements:
        if pl["word"] not in ENTRANCE_WORDS:
            continue
        h, L, si = pl["horizontal"], pl["line"], -pl["outward"]  # si: the room side
        lo, hi = pl["span"]

        def seg(a, b, h=h):  # (along, across) -> model
            return ((a[0], a[1]), (b[0], b[1])) if h else ((a[1], a[0]), (b[1], b[0]))
        for e, d in ((lo, -1), (hi, 1)):
            x = e - d * t_  # the casing's frame-side edge (the casing stands in the frame's end)
            # the outer lines (outdoor side) that stop inside the casing stop at the wall's end
            new = []
            for (p, q), color, label in offsets:
                (pa, pc_), (qa, qc) = ((p[0], p[1]), (q[0], q[1])) if h else ((p[1], p[0]), (q[1], q[0]))
                if abs(pc_ - qc) < EPS and (pc_ - L) * si < 0 and abs(pc_ - L) <= 200:
                    if min(e, x) - 2 < pa < max(e, x) + 2:
                        pa = e
                    if min(e, x) - 2 < qa < max(e, x) + 2:
                        qa = e
                    (p, q) = seg((pa, pc_), (qa, qc))
                new.append(((p, q), color, label))
            offsets = new
            st = lay["style"]
            extras.append((seg((e, L - si * ins), (e, L + si * col)), lay["layer"], lay["structure_color"], st))
            extras.append((seg((e, L + si * col), (e, L + si * (board + proud))), lay["layer"], lay["finish_color"], st))
            extras.append((seg((e, L + si * (board + proud)), (x, L + si * (board + proud))), lay["layer"], lay["finish_color"], st))
            extras.append((seg((x, L + si * (board + proud)), (x, L - si * reach)), lay["layer"], lay["finish_color"], st))
        print(f"玄関ドアの額縁（厚{t_}・暫定）: {pl['block']}")
    return offsets, extras


def window_extras(placements, cfg, spaces=()):
    """Wall returns and 窓台 at each window: [((p, q), layer, color, style)] in model mm.
    Returns run from the room-side face to the window frame's interior face: the column-face /
    下地 line (52.5, color 1) at the frame edge, the board face (65, color 5) 12.5 inside it.
    窓台: 12.5 into the walls each side (= frame width), 10 in front of the board face."""
    off = cfg["offsets_mm"]
    lay = cfg["layers"]["wall"]
    sill = cfg["placement"]["window_sill"]
    col, board, embed = off["column_face"], off["board_face_inside"], sill["embed_mm"]
    front = board + sill["projection_mm"]
    out = []
    for pl in placements:
        if pl["word"] not in pf_WINDOWS:
            continue
        no_sill = garage_door_window(pl, spaces, cfg)
        L, s, (lo, hi) = pl["line"], pl["outward"], pl["span"]
        face = L + s * pl["inner_mm"]  # the frame's interior face (model, across the wall)

        def along(t, a, b, h=pl["horizontal"]):  # a segment running along the wall at across t
            return ((a, t), (b, t)) if h else ((t, a), (t, b))

        def across(t, a, b, h=pl["horizontal"]):  # a segment across the wall at along t
            return ((t, a), (t, b)) if h else ((a, t), (b, t))

        st = lay["style"]
        for t in (lo, hi):
            out.append((across(t, L - s * col, face), lay["layer"], lay["structure_color"], st))
        for t in (lo + embed, hi - embed):
            out.append((across(t, L - s * board, face), lay["layer"], lay["finish_color"], st))
        if no_sill:
            print(f"窓台なし（ガレージへのドア代わりの窓）: {pl['block']}")
            continue
        sl = (sill["layer"], sill["color"], sill["style"])
        out.append((along(L - s * front, lo, hi), *sl))
        for t in (lo, hi):
            out.append((across(t, L - s * board, L - s * front), *sl))
    return out


def block_lines_mm(entities, pl, S):
    """Lines of a placed block in model mm (local paper mm -> scale, mirror, rotate, move)."""
    import math
    ang = math.radians(pl["rot"])
    ca, sa = math.cos(ang), math.sin(ang)
    out = []

    def tr(u, v):
        u, v = u * S * pl["sx"], v * S * pl["sy"]
        return (pl["pos"][0] + u * ca - v * sa, pl["pos"][1] + u * sa + v * ca)

    for e in entities:
        if e["type"] == "LINE":
            out.append((tr(e["start_x"], e["start_y"]), tr(e["end_x"], e["end_y"])))
        elif e["type"] == "ARC":
            n = 12
            pts = [tr(e["center_x"] + e["radius"] * math.cos(e["start_angle"] + e["arc_angle"] * i / n),
                      e["center_y"] + e["radius"] * math.sin(e["start_angle"] + e["arc_angle"] * i / n))
                   for i in range(n + 1)]
            out += list(zip(pts, pts[1:]))
    return out


def kamachi_lines(ops, names, mapper, cfg, walls=None):
    """框 (上がり框): two lines parallel to its grid line, 52.5 and 157.5 toward the 玄関 (the
    user's own drawing, provisional 2026-10-23), between the wall faces at its ends."""
    pc = cfg["placement"]
    half = cfg["input"]["half_grid_mm"]
    face = cfg["offsets_mm"]["board_face_inside"]
    jamb = pc["jamb_search_mm"]
    # the 土間 side: the 玄関, or a 土間収納 off it (プラン3 1F: the 框 between the ホール and the
    # 土間収納 was offset to the ホール side, toward the farther 玄関 label, narrowing the ホール, 10/27)
    genkan = [(mapper.xmm(n["x"]), mapper.ymm(n["y"])) for n in names if n.get("name") in ("玄関", "土間")]
    found = []  # (h, line, s, ga, gb)
    for o in ops:
        if o.get("kind") != "kamachi" or not genkan:
            continue
        h = o["horizontal"]
        line = round((o["p1_mm"][1] if h else o["p1_mm"][0]) / half) * half
        a, b = sorted((o["p1_mm"][0], o["p2_mm"][0]) if h else (o["p1_mm"][1], o["p2_mm"][1]))
        ga, gb = round((a - jamb) / half) * half, round((b + jamb) / half) * half
        mid = (a + b) / 2
        g = min(genkan, key=lambda p: math.dist(p, (mid, line) if h else (line, mid)))
        s = 1 if (g[1] if h else g[0]) > line else -1  # the 玄関 side
        ex = o.get("exact")
        if any(f[0] == h and f[1] == line and f[3] < gb and f[4] > ga for f in found):
            continue  # (found both as an opening and as two lines: once)
        found.append((h, line, s, ga, gb, ex))
    out = []
    outer = max(pc["kamachi_mm"])
    for h, line, s, ga, gb, ex in found:
        for d in pc["kamachi_mm"]:
            c = line + s * d
            lo, hi = ex if ex else (ga + face, gb - face)
            if d == outer:
                # an L of two 框 meeting at a corner: the outer lines run on to meet each other
                # (プラン3 1F 玄関/ホール/土間収納, user's correction 10/27)
                for h2, line2, s2, ga2, gb2, _ex in found:
                    if h2 != h and line in (ga2, gb2) and (line2 == ga or line2 == gb):
                        end = line2 + s2 * outer
                        lo, hi = (end, hi) if line2 == ga else (lo, end)
            seg = LineString([(lo, c), (hi, c)] if h else [(c, lo), (c, hi)])
            if walls:
                seg = seg.difference(unary_union([w.rect(face) for w in walls
                                                  if not (w.h == h and abs(w.c - line) < 1)]))  # (broken where it crosses a wall, user's correction)
            for part in getattr(seg, "geoms", [seg]):
                if part.length > 1:
                    p, q = part.coords[0], part.coords[-1]
                    out.append(((p, q), pc["kamachi_layer"], pc["kamachi_color"], pc["kamachi_style"]))
    return out


def open_kamachi(img_path, walls, names, M, cfg):
    """框 on a grid line with no wall (プラン3 1F x=5005 between the 玄関 and the ホール: no wall
    line, so no opening was found there; 10/27). The drawing shows the 框 as two continuous dark
    lines 52.5 and 157.5 from the grid line on the 玄関 side, running from wall to wall. Returns
    pseudo openings (kind "kamachi") for kamachi_lines."""
    pc = cfg["placement"]
    half = cfg["input"]["half_grid_mm"]
    tol, dark, min_frac = pc["open_kamachi_tol_mm"], pc["open_kamachi_dark"], pc["open_kamachi_min_frac"]
    face = cfg["offsets_mm"]["board_face_inside"]
    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    labels = [(M.xmm(n["x"]), M.ymm(n["y"])) for n in names if n.get("name") in ("玄関", "土間")]
    out = []

    def dark_at(h, c, t):
        # a dark pixel within ±tol across the line at (along t, across c)
        ok = False
        for dd in np.arange(-tol, tol + 0.1, 4):
            x, y = (M.x(t), M.y(c + dd)) if h else (M.x(c + dd), M.y(t))
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H and img[yi, xi] < dark:
                ok = True
                break
        return ok

    for lx, ly in labels:
        for h in (True, False):
            near, along = (ly, lx) if h else (lx, ly)
            # grid lines within 3 bays of the label, on each side
            base = round(near / (2 * half)) * 2 * half
            for c in (base + k * 2 * half for k in range(-4, 5)):
                if abs(c - near) < half:
                    continue
                s = 1 if near > c else -1  # the 玄関 side
                # the bay along the line: the walls crossing / ending on it either side of the label
                cross = sorted(w.c for w in walls if w.h != h and w.a - 1 <= c <= w.b + 1)
                lo = max((v for v in cross if v <= along), default=None)
                hi = min((v for v in cross if v >= along), default=None)
                if lo is None or hi is None or hi - lo < 2 * half:
                    continue
                if any(w.h == h and abs(w.c - c) < 1 and w.a < hi - 1 and w.b > lo + 1 for w in walls):
                    continue  # a wall stands there (its 框 is found as an opening)
                ts = np.arange(lo + face, hi - face, 10)
                both = [all(dark_at(h, c + s * d, t) for d in pc["kamachi_mm"]) for t in ts]
                # the longest run of the two lines (gaps up to 50 mm: text, joints): a 框 may stop
                # at a cabinet instead of a wall (プラン5 1F: the 下駄箱 at its lower end, 10/27)
                runs, start, gap = [], None, 0
                for i, ok in enumerate(both + [False] * 6):
                    if ok:
                        start = i if start is None else start
                        gap, last = 0, i
                    elif start is not None:
                        gap += 1
                        if gap > 5:
                            runs.append((start, last))
                            start, gap = None, 0
                if not runs:
                    continue
                r0, r1 = max(runs, key=lambda r: r[1] - r[0])
                if (r1 - r0) * 10 < max(min_frac * (hi - lo - 2 * face), pc["open_kamachi_min_mm"]):
                    continue
                e0 = lo + face if r0 * 10 <= 100 else ts[r0]
                e1 = hi - face if (len(ts) - 1 - r1) * 10 <= 100 else ts[r1]
                o = {"kind": "kamachi", "horizontal": h, "exact": [float(e0), float(e1)],
                     "p1_mm": [lo, c] if h else [c, lo], "p2_mm": [hi, c] if h else [c, hi]}
                if o not in out:
                    out.append(o)
                    print(f"壁のない所の框: {'横' if h else '縦'} {c} {e0:.0f}〜{e1:.0f}")
    return out


def draw_fittings(img_path, det, offsets, placements, skipped, lib, cfg, out, title, extras=(), cols=None):
    cols = cols or {}
    k = det["mm_per_px"]
    x0, y0 = det["origin_px"]
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.addWeighted(bgr, 0.4, np.full_like(bgr, 255), 0.6, 0)

    def px(p):
        return int(round(p[0] / k + x0)), int(round(y0 - p[1] / k))

    for (p, q), color, _ in offsets:
        cv2.line(img, px(p), px(q), (160, 90, 160) if color == 5 else (90, 90, 90), 1, cv2.LINE_AA)
    for (p, q), _l, color, _s in extras:
        cv2.line(img, px(p), px(q), (160, 90, 160) if color == 5 else (0, 140, 0) if color == 3 else (90, 90, 90),
                 1, cv2.LINE_AA)
    h = cfg["column"]["size_mm"] / 2
    for (x, y), _why in cols.items():
        cv2.rectangle(img, px((x - h, y - h)), px((x + h, y + h)), (0, 0, 0), 1)
    by_name = {v[0]: v[1] for v in lib.values()}
    names = []
    for pl in placements:
        for p, q in block_lines_mm(by_name[pl["block"]], pl, cfg["layers"]["scale"]):
            cv2.line(img, px(p), px(q), (200, 0, 160), 1, cv2.LINE_AA)
        cv2.circle(img, px(pl["pos"]), 3, (0, 0, 220), -1)
        # part name (kind + catalogue width); windows whose code was not read are flagged
        short = pl["block"].split("_", 2)[2].replace("_W", " ")
        if pl["kind"] == "window" and not pl.get("label"):
            short += "（記号読めず）"
        x, y = px(pl["pos"])
        names.append((short, (x + 4, y + 4), (160, 60, 0), 13))
    img = jp_text.put(img, names)
    for o in skipped:
        p0, p1, c = o["px"]
        c = int(round(c))
        if o["horizontal"]:
            cv2.rectangle(img, (p0, c - 8), (p1, c + 8), (0, 160, 230), 2)
        else:
            cv2.rectangle(img, (c - 8, p0), (c + 8, p1), (0, 160, 230), 2)
    img = jp_text.legend(img, title, [((200, 0, 160), "置いた建具（部品集のブロック）"),
                                     ((0, 0, 220), "ブロックの基準点（左の柱の中心）"),
                                     ((0, 140, 0), "窓台・腰壁の笠木"),
                                     ((0, 0, 0), "柱（建具には被せない）"),
                                     ((160, 60, 0), "部品名と幅（窓は図面の記号から種類を判定）"),
                                     ((0, 160, 230), "部品がまだない建具")])
    cv2.imencode(".png", img)[1].tofile(out)


def draw_check(img_path, det, walls, offsets, cols, defaults, out, title):
    k = det["mm_per_px"]
    x0, y0 = det["origin_px"]
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.addWeighted(bgr, 0.45, np.full_like(bgr, 255), 0.55, 0)

    def px(p):
        return int(round(p[0] / k + x0)), int(round(y0 - p[1] / k))

    for (p, q), color, _ in offsets:
        cv2.line(img, px(p), px(q), (160, 90, 160) if color == 5 else (90, 90, 90), 1, cv2.LINE_AA)
    for w in walls:
        cv2.line(img, px(w.pt(w.a)), px(w.pt(w.b)), (200, 80, 0) if w.ext else (0, 150, 0), 2, cv2.LINE_AA)
    h = 52.5
    for (x, y), why in cols.items():
        a, b = px((x - h, y - h)), px((x + h, y + h))
        cv2.rectangle(img, a, b, (0, 0, 0) if why != "pitch" else (0, 0, 200), 2)
        cv2.line(img, a, b, (0, 0, 0), 1)
        cv2.line(img, (a[0], b[1]), (b[0], a[1]), (0, 0, 0), 1)
    for p in defaults:
        cv2.circle(img, px(p), 12, (255, 0, 255), 2)
    legend = [((200, 80, 0), "外壁の壁芯"), ((0, 150, 0), "内壁の壁芯"),
              ((90, 90, 90), "躯体・下地の線（線色1）"), ((160, 90, 160), "仕上げの線（線色5）"),
              ((0, 0, 0), "柱（端部・L字・T字・十字）"), ((0, 0, 200), "柱（1820刻み）"),
              ((255, 0, 255), "余りを既定側に置いた区間（要確認）")]
    img = jp_text.legend(img, title, legend)
    cv2.imencode(".png", img)[1].tofile(out)


def main():
    cfg_path, walls_path, img_path, prefix = sys.argv[1:5]
    upper_path = sys.argv[5] if len(sys.argv) > 5 else None
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    grid = cfg["input"]["half_grid_mm"]  # walls may sit on 227.5 lines (closets)
    walls, det = load_walls(walls_path, grid)
    bare = bare_lines(det, img_path, walls_path.replace("_walls.json", "_openings.json"), cfg)
    if bare:
        walls, det = load_walls(walls_path, grid, bare)
    furniture_ops = drop_furniture_lines(walls, walls_path.replace("_walls.json", "_openings.json"), cfg)
    furniture_ops |= stair_foot_openings(walls, walls_path, cfg)
    fp = footprint_of(walls)
    # outside spaces inside the building outline (garage, porch, balcony; user 10/24): the
    # interior outline leaves them out, their walls get the outside-space build-up.
    # (found before the stray lines are dropped: a garage open on its shutter side lies outside
    # fp, and its own walls are what close it, プラン3 1F 10/27)
    building = fp
    names_path = walls_path.replace("_walls.json", "_部屋名.json")
    spaces = outside_spaces(walls, fp, names_path, det, cfg) if Path(names_path).exists() else []
    shutters = []
    for poly, kind in spaces:
        # the garage's shutter side (no wall line, closed on the dimension extent): the wall runs
        # on across it (user 10/27, overriding the open gap of 10/24)
        if kind not in cfg["outside_space"]["garage_names"]:
            continue
        drawn = unary_union([LineString([w.pt(w.a), w.pt(w.b)]) for w in walls]).buffer(1)
        cs = list(poly.exterior.coords)
        for p, q in zip(cs, cs[1:]):
            rest = LineString([p, q]).difference(drawn)
            for seg in getattr(rest, "geoms", [rest]):
                if seg.is_empty or seg.length < cfg["input"]["grid_mm"]:
                    continue
                (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
                if abs(x1 - x2) > EPS and abs(y1 - y2) > EPS:
                    continue
                snap_ = lambda v: round(v / grid) * grid  # noqa: E731
                sw = Wall((snap_(x1), snap_(y1)), (snap_(x2), snap_(y2)), False)
                walls.append(sw)
                # (an opening: no column on it, user's correction 10/26)
                hc = cfg["column"]["size_mm"] / 2
                shutters.append((sw.h, sw.c, sw.a + hc + 1, sw.b - hc - 1, "シャッター"))
                print(f"シャッターの所も壁を連続: {kind} ({snap_(x1)}, {snap_(y1)})〜({snap_(x2)}, {snap_(y2)})")
    if spaces:
        walls = merge_collinear(walls)
        # (a garage closed on the dimension extent lies outside fp: the building takes it in)
        building = unary_union([fp] + [p for p, _k in spaces])
        if building.geom_type == "MultiPolygon":
            building = max(building.geoms, key=lambda g: g.area)
        building = Polygon(building.exterior)
    # stray lines outside the building (eaves, terrace edges) are not walls
    # (a wall that only runs on past the outline, like a porch side wall, is kept whole)
    inside = building.buffer(1)
    walls = [w for w in walls
             if LineString([w.pt(w.a), w.pt(w.b)]).intersection(inside).length >= 0.5 * (w.b - w.a)]
    if spaces:
        rest = fp.difference(unary_union([p for p, _k in spaces]))
        if rest.geom_type == "MultiPolygon":
            rest = max(rest.geoms, key=lambda g: g.area)
        if rest.geom_type == "Polygon" and not rest.is_empty:
            fp = Polygon(rest.exterior)
        print("外部空間:", [(k, round(p.area / 1e6, 1)) for p, k in spaces])
    classify_boundary(walls, fp)
    space_walls = {}
    for w in walls:
        if w.ext:
            continue
        mid = (w.a + w.b) / 2
        sides = {}
        for s in (1, -1):
            pt = Point(mid, w.c + s * 200) if w.h else Point(w.c + s * 200, mid)
            sides[s] = "in" if fp.contains(pt) else ("space" if any(p.contains(pt) for p, _k in spaces) else "out")
        if "in" not in sides.values() and "space" in sides.values():
            space_walls[w] = 1 if sides[1] == "space" else -1
    nodes = nodes_of(walls)
    offsets = offset_lines(walls, fp, nodes, cfg, spaces, space_walls)
    upper = load_walls(upper_path, grid)[0] if upper_path else None
    minx, miny, maxx, maxy = building.bounds
    center = ((minx + maxx) / 2, (miny + maxy) / 2)

    # fittings from the parts library, placed at the detected openings (before the columns:
    # a column never overlaps a fitting)
    placements, skipped, lib, extras = [], [], {}, []
    ops_path = walls_path.replace("_walls.json", "_openings.json")
    root = Path(__file__).resolve().parent.parent
    lib_paths = sorted(root.glob(cfg["placement"]["library"].replace("\\", "/")))
    if Path(ops_path).exists() and lib_paths:
        import place_fittings as pf
        from opening_detect import Mapper
        lib = pf.load_library([str(p) for p in lib_paths])
        ops = json.load(open(ops_path, encoding="utf-8"))
        for o in (ops["openings"] if isinstance(ops, dict) else ops):
            if o.get("no") in furniture_ops:
                o["rejected"] = "家具の線"
        def outside(h, line, mid):
            probe = Point(mid, line + 300) if h else Point(line + 300, mid)
            return -1 if building.contains(probe) else 1

        rooms = list(polygonize(unary_union([LineString([w.pt(w.a), w.pt(w.b)]) for w in walls])))

        def larger_side(h, line, mid):
            def area(pt):
                return next((r.area for r in rooms if r.contains(pt)), float("inf"))
            plus = Point(mid, line + 300) if h else Point(line + 300, mid)
            minus = Point(mid, line - 300) if h else Point(line - 300, mid)
            return 1 if area(plus) >= area(minus) else -1

        wall_segs = [(w.pt(w.a), w.pt(w.b)) for w in walls]

        def perp_at(h, line, pos, dirn, tol=150):
            """Is there a wall meeting the wall line near where a W1644's 中方立 would stand (845 from
            the wall at `pos`, the end away from the pocket, toward the pocket)? The frame on the side
            without the pocket is over 780, so such a wall leaves no room for the W1644 (user 10/28-29:
            a wall there -> W1454, none -> W1644)."""
            m = pos + dirn * 845
            for (x1, y1), (x2, y2) in wall_segs:
                if h and abs(x1 - x2) < 1 and abs(x1 - m) <= tol and min(y1, y2) - 1 <= line <= max(y1, y2) + 1:
                    return True
                if not h and abs(y1 - y2) < 1 and abs(y1 - m) <= tol and min(x1, x2) - 1 <= line <= max(x1, x2) + 1:
                    return True
            return False

        placements, skipped = pf.plan(ops, lib, cfg, Mapper(det, cfg["input"]["grid_mm"]), outside, larger_side, perp_at)
        # a window in an outside-space wall with no added insulation (a garage's outer wall not in
        # line with the house's exterior walls): its reference is the sheathing's outer face, 60
        # in from where the added insulation's would be (プラン3 1F ガレージ FIX, user 10/27)
        on_ring = ring_faces(walls, fp)
        ext_side = cfg["offsets_mm"]["exterior_side"]
        thin = ext_side["added_insulation"] - ext_side["sheathing"]
        for pl in placements:
            if pl["word"] not in pf.OUTWARD or (pl["horizontal"], round(pl["line"], 1), pl["outward"]) in on_ring:
                continue
            mid = (pl["span"][0] + pl["span"][1]) / 2
            if not any(w.h == pl["horizontal"] and abs(w.c - pl["line"]) < 1 and w.a - 1 <= mid <= w.b + 1
                       for w in space_walls):
                continue
            x, y = pl["pos"]
            pl["pos"] = (x, y - pl["outward"] * thin) if pl["horizontal"] else (x - pl["outward"] * thin, y)
            pl["inner_mm"] -= thin
            print(f"付加断熱なしの壁の窓（基準を{thin}内へ）: {pl['block']}")
        # a window end close to an outside corner of the exterior walls (入隅): the frame end
        # (the window's reference mark) keeps 50 off the other wall's exterior face (user's note on
        # プラン3 1F, 10/26: W1540 moved 8330 -> 8394.5 = face 8344.5 + 50)
        gap = cfg["placement"]["window_corner_gap_mm"]
        for pl in placements:
            if pl["word"] not in pf_WINDOWS:
                continue
            h, L, s = pl["horizontal"], pl["line"], pl["outward"]
            lo, hi = pl["span"]
            move = 0.0
            # (inside a garage the last finish is the 12.5 board, 152 from the centre, not the siding)
            mid = (lo + hi) / 2
            probe = Point(mid, L + s * 300) if h else Point(L + s * 300, mid)
            face_d = ext_side["exterior_face"]
            if any(k in cfg["outside_space"]["garage_names"] and p.contains(probe) for p, k in spaces):
                face_d = ext_side["ventilation"] + cfg["outside_space"]["garage_board_mm"]
            for w in walls:
                if not w.ext or w.h == h or not (w.a - 1 <= L + s * 200 <= w.b + 1):
                    continue  # (a perpendicular exterior wall running on outside the window's wall)
                if lo - 400 < w.c <= lo and lo < w.c + face_d + gap:
                    move = max(move, w.c + face_d + gap - lo)
                if hi <= w.c < hi + 400 and hi > w.c - face_d - gap:
                    move = min(move, w.c - face_d - gap - hi)
            if move:
                x, y = pl["pos"]
                pl["pos"] = (x + move, y) if h else (x, y + move)
                pl["span"] = (lo + move, hi + move)
                print(f"入隅に近い窓を外壁面から{gap}離す（{move:+.1f}）: {pl['block']}")
        offsets = cut_openings(offsets, placements, cfg, lib=lib)
        offsets, casing = entrance_casing(offsets, placements, cfg)
        extras = window_extras(placements, cfg, spaces) + casing
        names_path = walls_path.replace("_walls.json", "_部屋名.json")
        if Path(names_path).exists():
            names = json.load(open(names_path, encoding="utf-8"))
            M = Mapper(det, cfg["input"]["grid_mm"])
            open_k = open_kamachi(img_path, walls, names, M, cfg)
            extras += kamachi_lines(list(ops) + open_k, names, M, cfg, walls)
    kos_path = walls_path.replace("_walls.json", "_腰壁.json")
    kos = []
    if Path(kos_path).exists():
        voids = []
        npath = walls_path.replace("_walls.json", "_部屋名.json")
        if upper_path is None and Path(npath).exists():  # (2F: on 1F 「上部吹抜」 is the floor above)
            from opening_detect import Mapper
            Mv = Mapper(det, cfg["input"]["grid_mm"])
            voids = [(Mv.xmm(n["x"]), Mv.ymm(n["y"])) for n in json.load(open(npath, encoding="utf-8"))
                     if n.get("name") == "吹抜"]
        k_lines, kos = koshikabe_lines(json.load(open(kos_path, encoding="utf-8")), walls, fp, cfg, voids)
        extras += k_lines
    fits = [(pl["horizontal"], pl["line"], *pl["span"], pl["word"]) for pl in placements] + shutters
    cols, defaults = place_columns(walls, nodes, cfg, upper, fits)
    if placements:
        offsets = jamb_closures(offsets, placements, cols, walls, cfg)
    write_jww(prefix + ".jww", walls, offsets, cols, cfg, center, placements, lib, extras)
    # model mm = paper mm * scale + center; score_walls.py needs this to read corrected drawings back
    json.dump({"center_mm": center, "scale": cfg["layers"]["scale"]},
              open(prefix + "_meta.json", "w", encoding="utf-8"))
    name = prefix.replace("\\", "/").split("/")[-1]
    draw_check(img_path, det, walls, offsets, cols, defaults, prefix + "_壁と柱.png", f"{name}　壁と柱")
    draw_fittings(img_path, det, offsets, placements, skipped, lib, cfg, prefix + "_建具の配置.png",
                  f"{name}　建具の配置", extras, cols)
    print(f"{name}: fittings placed={len(placements)} no part yet={len(skipped)}")
    off_end = [pl for pl in placements if pl.get("end_gap_mm", 0) > cfg["placement"]["end_gap_max_mm"]]
    for pl in off_end:
        print(f"  要確認：建具の端が図面の開口の端から {pl['end_gap_mm']}mm ずれている {pl['block']} 位置{pl['pos']}")
    kinds = defaultdict(int)
    for why in cols.values():
        kinds[why] += 1
    print(f"{name}: walls={len(walls)} (exterior {sum(w.ext for w in walls)}) footprint={fp.bounds} "
          f"offset lines={len(offsets)} columns={len(cols)} {dict(kinds)} remainder-default={len(defaults)}")


if __name__ == "__main__":
    main()
