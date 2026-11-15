"""Stage 2: wall offsets + columns -> .jww and check PNG.

usage: make_plan.py <config.json> <floor_walls.json> <floor_image.png> <out_prefix> [upper_walls.json]
"""
import json, os
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
BOXES = []  # 玄関収納 bodies (x0, y0, x1, y1), mm: the 框 shows through them as dotted
ENT_RUNS = []  # 玄関収納 found in the drawing (entrance_storage.detect): dicts
FURN = []  # stretches taken out of the walls as furniture: (horizontal, line_mm, a_mm, b_mm)


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
        self.ext_in = ext  # as the input called it (before the outline reading)

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
            FURN.append((h, line, a, b))
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


def furniture_runs(walls, det, img_path, base_path, cfg):
    """Furniture read as a wall: a stretch of an interior wall (300 mm or more) with no gray band
    and no fitting on it. The user's answers of 11/5 (質問4): a narrow strip beside it (冷蔵庫・
    カップボード・玄関収納, プラン4 1F・プラン5 1F) is furniture, so the stretch is taken out of the
    wall; a line with the same room on both sides (プラン5 1F, in the LDK) is a 腰壁; a stretch
    between two ordinary rooms, e.g. the entrance of a UB (a closed room), stays as it is.
    Takes the stretches out of walls in place and returns the 腰壁 entries to add."""
    from opening_detect import Mapper
    fc = cfg["furniture_runs"]
    regions_path, ops_path = base_path + "_区画.json", base_path + "_openings.json"
    if not (Path(regions_path).exists() and Path(ops_path).exists()):
        return []
    regions = json.load(open(regions_path, encoding="utf-8"))
    ops = json.load(open(ops_path, encoding="utf-8"))
    ops = ops["openings"] if isinstance(ops, dict) else ops
    M = Mapper(det, cfg["input"]["grid_mm"])
    lo, hi = det["gray_range"]
    g = cv2.cvtColor(cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    gray = (g >= lo) & (g <= hi)

    def region_at(X, Y):
        best = None
        for r in regions:
            x0, y0, x1, y1 = r["bbox_px"]
            if x0 <= X <= x1 and y0 <= Y <= y1 and (best is None or (x1 - x0) * (y1 - y0) < best[0]):
                best = ((x1 - x0) * (y1 - y0), r)
        return best[1] if best else None

    kos, cuts = [], []
    for w in walls:
        if w.ext:
            continue
        ts = np.arange(w.a + 60, w.b - 60, 12.5)
        if not len(ts):
            continue
        pts = [(M.x(t), M.y(w.c)) if w.h else (M.x(w.c), M.y(t)) for t in ts]
        on = [gray[max(int(round(y)) - 2, 0):int(round(y)) + 3, max(int(round(x)) - 2, 0):int(round(x)) + 3].mean() > 0.3
              for x, y in pts]
        runs, st = [], None
        for t, v in zip(list(ts) + [None], on + [True]):
            if not v and st is None:
                st = t
            elif v and st is not None:
                runs.append((st, t if t is not None else ts[-1]))
                st = None
        i0 = 0 if w.h else 1
        for a, b in runs:
            if b - a < fc["min_len_mm"]:
                continue
            fitted = False
            for o in ops:
                c = o["p1_mm"][1] if w.h else o["p1_mm"][0]
                if o.get("rejected") or o["horizontal"] != w.h or abs(c - w.c) > 100:
                    continue
                oa, ob = sorted((o["p1_mm"][i0], o["p2_mm"][i0]))
                fitted = fitted or (oa - 150 <= a and ob + 150 >= b)
            if fitted:
                continue
            mid, sides = (a + b) / 2, []
            for sg in (1, -1):
                d = fc["probe_mm"] * sg
                X, Y = (M.x(mid), M.y(w.c + d)) if w.h else (M.x(w.c + d), M.y(mid))
                sides.append(region_at(X, Y))
            same = sides[0] is not None and sides[0] is sides[1]
            narrow = any(r is None or r["area_m2"] < fc["narrow_region_m2"] for r in sides)
            what = "腰壁" if same else ("家具" if narrow else None)
            if what is None:
                continue
            cuts.append((w, a, b))
            if what == "家具":
                FURN.append((w.h, w.c, a, b))
            if what == "腰壁":
                kos.append({"horizontal": w.h, "line_mm": w.c, "a_mm": a, "b_mm": b, "len_px": 40,
                            "source": "家具の列"})
            print(f"{what}の線を壁から外す（灰色の帯なし・建具なし）: {'横' if w.h else '縦'} {w.c} {round(a)}〜{round(b)}")
    for w, a, b in cuts:
        if w not in walls:
            continue
        walls.remove(w)
        # (the stretch is cut from the nearest end of the sampled run out to the wall faces)
        for s, e in ((w.a, a - 60), (b + 60, w.b)):
            if e - s > 100:
                walls.append(Wall(w.pt(s), w.pt(e), w.ext))
    return kos


def furniture_symbols(det, img_path, base_path, cfg, walls=()):
    """Draw the furniture that was taken out of the walls (user 11/7: 記号まで; 参考図面ではレイヤー6が
    家具・設備、レイヤー5が建具。レイヤー分けは会社ごとなので設定 furniture_draw)。
    The body sits against the wall's face read from the image (the line found is grid-snapped and can
    be 100 mm off the real furniture): the gray band's near edge across the stretch, and the end walls'
    faces along it. 玄関収納 (beside 玄関/ホール/土間): 800 hatched + 400 tall (diagonals) = 1200,
    depth 400. A stretch of 910 + 900 units (3000 or more): 冷蔵庫 (700 square, dotted) at the lower
    end + カップボード (depth 455, 900 units, hatched). Returns extras lines (mm)."""
    from opening_detect import Mapper
    fd = cfg.get("furniture_draw")
    rp = base_path + "_区画.json"
    if not (fd and FURN and Path(rp).exists()):
        return []
    regions = json.load(open(rp, encoding="utf-8"))
    M = Mapper(det, cfg["input"]["grid_mm"])
    lo, hi = det["gray_range"]
    g = cv2.cvtColor(cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    gray = (g >= lo) & (g <= hi)

    def region_at(X, Y):
        best = None
        for r in regions:
            x0, y0, x1, y1 = r["bbox_px"]
            if x0 <= X <= x1 and y0 <= Y <= y1 and (best is None or (x1 - x0) * (y1 - y0) < best[0]):
                best = ((x1 - x0) * (y1 - y0), r)
        return best[1] if best else None

    out, seen = [], set()
    lay, col, sty, dot = fd["layer"], fd["color"], fd["style"], fd["dot_style"]
    lay_b = fd.get("built_in_layer", lay)  # 造り付け（カップボード・玄関収納）: fixed, not 据置 furniture
    cur = [lay]
    for h, c, a, b in FURN:
        if (h, c, round(a), round(b)) in seen or b - a < 300:
            continue
        seen.add((h, c, round(a), round(b)))
        mid = (a + b) / 2
        sides = []
        for sg in (1, -1):
            d = 150 * sg
            X, Y = (M.x(mid), M.y(c + d)) if h else (M.x(c + d), M.y(mid))
            sides.append(region_at(X, Y))
        thin = [r is not None and r["area_m2"] < cfg["furniture_runs"]["narrow_region_m2"] and not r.get("names")
                for r in sides]
        if not any(thin) or b - a < 600:
            continue  # no unnamed narrow strip beside it (or too short): not furniture to draw
        sg = 1 if thin[0] else -1  # the wall of the strip lies on this side
        names = " ".join(n for r in sides if r for n in r.get("names", []))
        entrance = any(k in names for k in ("玄関", "ホール", "土間"))
        depth = fd["entrance_depth_mm"] if entrance else fd["cupboard_depth_mm"]

        def on_gray(t, d):
            X, Y = (M.x(t), M.y(c + sg * d)) if h else (M.x(c + sg * d), M.y(t))
            X, Y = int(round(X)), int(round(Y))
            if not (1 <= Y < gray.shape[0] - 1 and 1 <= X < gray.shape[1] - 1):
                return False
            return gray[Y - 1:Y + 2, X - 1:X + 2].mean() > 0.5

        # the wall's near face: along the stretch, the farthest start of a gray run (>= 100 mm thick)
        # within reach (the appliance grays on the way are nearer)
        hits = []
        for t in np.arange(a + 150, b - 150, 100):
            best = None
            for d in np.arange(0, depth + 250, 12.5):
                if on_gray(t, d) and not on_gray(t, d - 12.5) and all(on_gray(t, d + k * 12.5) for k in range(1, 8)):
                    best = d
            if best is not None:
                hits.append(best)
        face = float(np.median(hits)) if len(hits) >= 3 else float(depth)
        dm = face - depth / 2  # mid-depth of the body

        def end_face(t0, step):  # end of the first gray run (>= 4 samples) met walking in from outside
            t, run, miss = t0, 0, 0
            for _ in range(80):
                if on_gray(t, dm):
                    run, miss = run + 1, 0
                elif run >= 4:
                    miss += 1
                    if miss >= 4:  # (thin dark outline lines split a band: only a real gap ends it)
                        return t - step * miss
                else:
                    run = 0
                t += step
            return None

        a_end = end_face(a - 100, 12.5)
        b_end = end_face(b + 100, -12.5)
        a0 = a_end if a_end is not None and abs(a_end - a) < 250 else a
        b0 = b_end if b_end is not None and abs(b_end - b) < 250 else b
        # snap to the model: the faces are exactly 65 from the wall centre lines (the image measure was
        # 10-30 mm off, the furniture sank into a wall or stood off it; user 11/8)
        board = cfg["offsets_mm"]["board_face_inside"]
        back = [w_ for w_ in walls if w_.h == h and abs(w_.c - (c + sg * face)) < 150 and w_.a - 150 <= a and b <= w_.b + 150]
        if back:
            w_ = min(back, key=lambda q: abs(q.c - (c + sg * face)))
            face = sg * (w_.c - c) - board
            dm = face - depth / 2
        across_ok = lambda w_, t: w_.h != h and w_.a - 200 <= c <= w_.b + 200 and abs(w_.c - t) < 200
        lo_w = [w_ for w_ in walls if across_ok(w_, a)]
        hi_w = [w_ for w_ in walls if across_ok(w_, b)]
        if lo_w:
            a0 = min(lo_w, key=lambda q: abs(q.c - a)).c + board
        if hi_w:
            b0 = min(hi_w, key=lambda q: abs(q.c - b)).c - board

        def pt(t, d):  # d: distance from the wall face, away from the wall
            return (t, c + sg * (face - d)) if h else (c + sg * (face - d), t)

        def box(t0, t1, dep, style, d0=0):  # dep = far depth; d0 = near depth (0 = on the wall face)
            for p, q in (((t0, d0), (t1, d0)), ((t0, dep), (t1, dep)), ((t0, d0), (t0, dep)), ((t1, d0), (t1, dep))):
                out.append(((pt(*p), pt(*q)), cur[0], col, style))

        def hatch(t0, t1, dep):  # the top's hatching: lines along the long side, 50 mm apart (user 11/7)
            d = fd["hatch_mm"]
            while d < dep - 1:
                out.append(((pt(t0, d), pt(t1, d)), cur[0], col, sty))
                d += fd["hatch_mm"]

        if entrance:
            cur[0] = lay_b
            low, tall = fd["entrance_low_mm"], fd["entrance_tall_mm"]
            xs = [pt(a0, 0)[0], pt(a0 + low + tall, depth)[0]]
            ys = [pt(a0, 0)[1], pt(a0 + low + tall, depth)[1]]
            BOXES.append((min(xs), min(ys), max(xs), max(ys)))
            box(a0, a0 + low, depth, sty)
            hatch(a0, a0 + low, depth)
            box(a0 + low, a0 + low + tall, depth, sty)
            out.append(((pt(a0 + low, 0), pt(a0 + low + tall, depth)), cur[0], col, sty))  # one diagonal only (user 11/7)
            print(f"家具を描く: 玄関収納 {'横' if h else '縦'} {c} {round(a0)}〜{round(a0 + low + tall)}"
                  f"（{low}ハッチ＋{tall}対角線、奥行き{depth}、壁の面から）")
            continue
        n_units = (b0 - a - 910) / fd["unit_mm"]
        if b - a >= 3000 and abs(n_units - round(n_units)) < 0.12:
            cur[0] = lay
            g0 = fd["fridge_gap_mm"]  # 50 mm clear of the walls on every side
            box(a0 + g0, a0 + g0 + fd["fridge_mm"], g0 + fd["fridge_mm"], dot, g0)
            c0 = a + 910
            cur[0] = lay_b
            box(c0, b0, depth, sty)
            n = max(1, round((b0 - c0) / fd["unit_mm"]))
            for k in range(1, n):
                t = c0 + (b0 - c0) * k / n
                out.append(((pt(t, 0), pt(t, depth)), cur[0], col, sty))
            hatch(c0, b0, depth)
            print(f"家具を描く: 冷蔵庫 {'横' if h else '縦'} {c} {round(a0 + g0)}〜{round(a0 + g0 + fd['fridge_mm'])}（{fd['fridge_mm']}角、壁から{g0}）"
                  f"＋カップボード {round(c0)}〜{round(b0)}（奥行き{depth}、壁の面から）")
        elif b - a < 1000:
            cur[0] = lay
            g0 = fd["fridge_gap_mm"]
            box(a0 + g0, a0 + g0 + fd["fridge_mm"], g0 + fd["fridge_mm"], dot, g0)
            print(f"家具を描く: 冷蔵庫 {'横' if h else '縦'} {c} {round(a0 + g0)}〜{round(a0 + g0 + fd['fridge_mm'])}")
        else:
            cur[0] = lay_b
            box(a0, b0, depth, sty)
            hatch(a0, b0, depth)
            print(f"家具を描く: カップボード {'横' if h else '縦'} {c} {round(a0)}〜{round(b0)}（奥行き{depth}、壁の面から）")
    return out


def find_entrance_runs(det, img_path, base_path, cfg):
    """The drawing's 玄関収納 (found once; the door shift and the drawing both use it)."""
    key = (img_path, base_path)
    if ENT_RUNS and ENT_RUNS[0] == key:
        return ENT_RUNS[1]
    names_p, ops_p = base_path + "_部屋名.json", base_path + "_openings.json"
    runs = []
    if cfg.get("furniture_draw") and Path(names_p).exists() and Path(ops_p).exists():
        import entrance_storage as ES
        names = json.load(open(names_p, encoding="utf-8"))
        ops = json.load(open(ops_p, encoding="utf-8"))
        ops = ops["openings"] if isinstance(ops, dict) else ops
        runs = ES.detect(img_path, det, names, ops, cfg)
    ENT_RUNS[:] = [key, runs]
    return runs


def entrance_storage_symbols(det, img_path, base_path, cfg, walls=()):
    """玄関収納を、図面の絵から見つけて描く（user 11/8〜: 壁を外した所だけでなく、図面にあるものを全部）。
    見つけ方は scripts/entrance_storage.py（400の倍数のユニット、低＝ハッチング、トール＝対角線1本）。
    場所は壁の面（壁芯から65）に合わせ、奥行き400。すでに furniture_symbols が描いた所（BOXES）は飛ばす。
    Returns extras lines (mm)."""
    fd = cfg.get("furniture_draw")
    names_p, ops_p = base_path + "_部屋名.json", base_path + "_openings.json"
    if not (fd and Path(names_p).exists() and Path(ops_p).exists()):
        return []
    import entrance_storage as ES
    runs = find_entrance_runs(det, img_path, base_path, cfg)
    board = cfg["offsets_mm"]["board_face_inside"]
    depth, unit = fd["entrance_depth_mm"], ES.UNIT
    lay_b, col, sty = fd.get("built_in_layer", fd["layer"]), fd["color"], fd["style"]
    out = []
    for r in runs:
        h, c = bool(r["horizontal"]), r["c_mm"]
        # the wall the body leans on: parallel, its face (65 from the centre) about 400 from the body's far side
        cand = [w for w in walls if w.h == h and w.a - 300 <= (r["a_mm"] + r["b_mm"]) / 2 <= w.b + 300
                and 150 <= abs(w.c - c) <= 400]
        if cand:
            w = min(cand, key=lambda q: abs(abs(q.c - c) - (board + depth / 2)))
            sg = 1 if w.c > c else -1  # the wall lies on this side
            face = w.c - sg * board  # the wall's board face (model)
        else:
            # no wall in the model there (it was taken out, e.g. under a stair): the drawing's gray band is the wall
            w = None
            sg = r["wall_side"]
            face = round((c + sg * depth / 2) / 25.0) * 25.0
        far = face - sg * depth
        # along the wall: a unit end meets a wall across it (centre +- 65) when one is near
        a0 = r["a_mm"]
        for q in walls:
            if q.h != h and q.a - 200 <= c <= q.b + 200 and abs(q.c - a0) < 250:
                a0 = q.c + board if q.c < a0 + 125 and abs(q.c + board - a0) < 250 else a0
        if a0 == r["a_mm"]:
            a0 = round(a0 / 25.0) * 25.0
        spans, t = [], a0
        for kind, t0, t1 in r["units"]:
            ln = max(unit, round((t1 - t0) / unit) * unit)
            spans.append((kind, t, t + ln))
            t += ln
        x0, x1 = a0, t
        pt = (lambda along, across: (along, across)) if h else (lambda along, across: (across, along))
        bx = [pt(x0, face), pt(x1, far)]
        box_ = (min(bx[0][0], bx[1][0]), min(bx[0][1], bx[1][1]), max(bx[0][0], bx[1][0]), max(bx[0][1], bx[1][1]))
        if any(min(box_[2], b[2]) - max(box_[0], b[0]) > 50 and min(box_[3], b[3]) - max(box_[1], b[1]) > 50 for b in BOXES):
            continue  # (already drawn from the stretch taken out of the wall)
        BOXES.append(box_)
        for kind, t0, t1 in spans:
            for p, q in (((t0, face), (t1, face)), ((t0, far), (t1, far)), ((t0, face), (t0, far)), ((t1, face), (t1, far))):
                out.append(((pt(*p), pt(*q)), lay_b, col, sty))
            if kind == "低":
                d = fd["hatch_mm"]
                while d < depth - 1:
                    across = face - sg * d
                    out.append(((pt(t0, across), pt(t1, across)), lay_b, col, sty))
                    d += fd["hatch_mm"]
            else:
                out.append(((pt(t0, face), pt(t1, far)), lay_b, col, sty))  # one diagonal only (user 11/7)
        print(f"玄関収納を描く（図面から）: {'横' if h else '縦'} 壁{('%g' % w.c) if w else 'なし（図面の灰色の帯）'} 面{face:g} {x0:g}〜{x1:g} "
              + "＋".join(f"{'ハッチ' if k == '低' else '対角線'}{b - a:g}" for k, a, b in spans))
    return out


def dot_under_boxes(extras, cfg):
    """The 框 shows through a 玄関収納 on the plan, but Jw_cad does not draw it through: only the part
    overlapping the storage is dotted (user 11/7). Splits the 框 lines at the boxes."""
    pc, fd = cfg["placement"], cfg.get("furniture_draw")
    if not (BOXES and fd):
        return extras
    from shapely.geometry import box as sbox
    out = []
    for (p, q), layer, color, style in extras:
        if (layer, color, style) != (pc["kamachi_layer"], pc["kamachi_color"], pc["kamachi_style"]):
            out.append(((p, q), layer, color, style))
            continue
        seg = LineString([p, q])
        inside = unary_union([sbox(*b) for b in BOXES])
        for part, dotted in ((seg.difference(inside), False), (seg.intersection(inside), True)):
            for g in getattr(part, "geoms", [part]):
                if g.length > 1 and g.geom_type == "LineString":
                    out.append(((g.coords[0], g.coords[-1]), layer, color, fd["dot_style"] if dotted else style))
    return out


def stud_lines(walls, cols, fits, cfg, placements=()):
    """間柱 (user 11/7): 105 x 45 inside the walls, on the 455 grid, where no 柱 stands, with one
    diagonal. 105 runs across the wall, 45 along it. Not inside an opening (a door's / window's
    span) and not beside a column.
    Beside a door: where the bay holds room for a stud beyond the frame (bay - 130 - frame width
    >= 45), a stud stands against the frame's free end: 735 in 910, 1190 in 1365, 1645 in 1820
    fit exactly with a stud at one column (1820 - 130 - 45 = 1645); a door with a few mm clearance
    (1644) or much room is drawn with the stud pushed to the door side (user 11/7).
    Returns extras lines (mm)."""
    sc = cfg.get("stud")
    if not sc:
        return []
    grid = cfg["input"]["grid_mm"]
    lay = cfg["layers"]["wall"]
    out = []
    la, dp = sc["along_mm"] / 2, sc["depth_mm"] / 2
    near = cfg["column"]["size_mm"] / 2 + la + sc["column_gap_mm"]
    board = cfg["offsets_mm"]["board_face_inside"]
    placed = []  # (wall id, t)

    def draw(w, t):
        if w.h:
            r = [(t - la, w.c - dp), (t + la, w.c - dp), (t + la, w.c + dp), (t - la, w.c + dp)]
        else:
            r = [(w.c - dp, t - la), (w.c + dp, t - la), (w.c + dp, t + la), (w.c - dp, t + la)]
        for i in range(4):
            out.append(((r[i], r[(i + 1) % 4]), lay["layer"], lay["structure_color"], lay["style"]))
        out.append(((r[0], r[2]), lay["layer"], lay["structure_color"], lay["style"]))  # one diagonal
        placed.append((id(w), t))

    seen = set()
    for w in walls:
        # a wall whose start sits on a 227.5 line (a 間崩し): the studs follow the wall's own grid
        phase = w.a % grid
        phase = phase if abs(phase - cfg["input"]["half_grid_mm"]) < 1.0 else 0.0
        t = math.ceil((w.a - phase - 1e-6) / grid) * grid + phase
        while t <= w.b + 1e-6:
            x, y = w.pt(t)
            key = (round(x), round(y))
            ok = key not in seen
            seen.add(key)
            if ok and any(math.hypot(x - cx, y - cy) < near for cx, cy in cols):
                ok = False
            if ok and any(f[0] == w.h and abs(f[1] - w.c) < 100 and f[2] - sc["opening_gap_mm"] <= t <= f[3] + sc["opening_gap_mm"]
                          for f in fits):
                ok = False
            if ok and (t - w.a < near or w.b - t < near):
                ok = False  # (a wall end carries a column)
            if ok:
                draw(w, t)
            t += grid
    # beside the doors
    for pl in placements:
        if pl["word"] not in INTERIOR_DOORS:
            continue
        h, L = pl["horizontal"], pl["line"]
        lo, hi = pl["span"]
        w = next((w for w in walls if w.h == h and abs(w.c - L) < EPS and w.a - 1 <= lo and hi <= w.b + 1), None)
        if w is None:
            continue
        along = [(c[0] if h else c[1]) for c in cols if abs((c[1] if h else c[0]) - L) < 1]
        left = [t for t in along if t <= lo + 1]
        right = [t for t in along if t >= hi - 1]
        if not left or not right:
            continue
        gl, gr = lo - (max(left) + board), (min(right) - board) - hi
        # beside a 両開き the stud stands 12.5 off the frame: the 石膏ボード goes between (user 11/12, プラン4)
        off_ = sc.get("double_door_gap_mm", 12.5) if pl["word"] == "両開き" else 0
        for gap, t_c in ((gl, lo - la - off_), (gr, hi + la + off_)):
            # (the stud may touch the column face: the 12.5 board beside the column is not in its way, user 11/12)
            if gap + (12.5 if off_ else 0) >= sc["along_mm"] + off_ - sc["door_gap_tol_mm"]:
                if any(pid == id(w) and abs(t - t_c) < sc["along_mm"] for pid, t in placed):
                    continue
                draw(w, t_c)
    return out


def fill_wall_gaps(walls, ops_path, cfg):
    """壁の小さなすき間をうめる (user 11/11, プラン3 2F: トイレの上の壁に455のすき間が残り、外周が
    閉じていなかった。そこは天井点検口の記号が壁の線に重なっていて、壁が読めなかった所)。
    同じ線上にある内壁と内壁のすき間が max_mm 以下で、そこに建具（見つかったもの）がなければ、壁でつなぐ。
    戸のない出入口は910なので、455以下なら壁として扱う。"""
    mx = cfg.get("wall_gap_fill", {}).get("max_mm")
    if not mx:
        return 0
    ops = []
    if Path(ops_path).exists():
        data = json.load(open(ops_path, encoding="utf-8"))
        ops = [o for o in (data["openings"] if isinstance(data, dict) else data) if not o.get("rejected")]
    n = 0
    for w in list(walls):
        if w.ext:
            continue
        for v in walls:
            if v is w or v.ext or v.h != w.h or abs(v.c - w.c) > 1 or v.a < w.b - EPS:
                continue
            gap = v.a - w.b
            if gap < EPS or gap > mx + EPS:
                continue
            hit = False
            for o in ops:
                if (o["p1_mm"][1] if w.h else o["p1_mm"][0]) is None:
                    continue
                oc = o["p1_mm"][1] if w.h else o["p1_mm"][0]
                oa, ob = sorted((o["p1_mm"][0], o["p2_mm"][0]) if w.h else (o["p1_mm"][1], o["p2_mm"][1]))
                if abs(oc - w.c) < 1 and oa < v.a and ob > w.b:
                    hit = True
            if hit:
                continue
            walls.append(Wall(w.pt(w.b), w.pt(v.a), False))
            n += 1
            print(f"壁のすき間をうめる（{gap:g}）: {'横' if w.h else '縦'} {w.c:g}  {w.b:g}〜{v.a:g}")
            break
    if n:
        walls[:] = merge_collinear(walls)
    return n


def close_exterior(walls, cfg):
    """外壁は必ず外周を閉じる (user 11/7: プラン3 2F, the WIC's underside on the balcony side was
    open). An exterior wall's end that touches no other exterior wall is a loose end; two loose ends facing
    each other on one line (within max_gap_mm) are joined by an exterior wall, when the new wall
    crosses no wall. Returns the new walls: they are added after the outline is taken and the walls
    are classified (closing first would change the outline and turn the stub walls out toward a
    balcony into interior walls, プラン3 2F, user 11/7)."""
    c = cfg.get("exterior_close")
    if not c:
        return []
    added = []
    lines = [LineString([w.pt(w.a), w.pt(w.b)]) for w in walls]
    loose = []
    for i, w in enumerate(walls):
        if not w.ext:
            continue
        for t in (w.a, w.b):
            pt = Point(*w.pt(t))
            if not any(j != i and walls[j].ext and lines[j].distance(pt) < 1.0 for j in range(len(walls))):
                loose.append((w.pt(t), w))
    done = set()
    for i, (p, wp) in enumerate(loose):
        for j in range(i + 1, len(loose)):
            q, wq = loose[j]
            horizontal = abs(p[1] - q[1]) < 1.0
            if not (horizontal or abs(p[0] - q[0]) < 1.0) or wp is wq:
                continue
            if horizontal == wp.h or horizontal == wq.h:
                continue  # the closing wall runs across the two walls' ends, not along them
            gap = abs(p[0] - q[0]) if horizontal else abs(p[1] - q[1])
            if gap < 1.0 or gap > c["max_gap_mm"] or (i, j) in done:
                continue
            seg = LineString([p, q])
            if any(seg.intersection(l).length > 0 or (seg.crosses(l)) for l in lines):
                continue
            added.append(Wall(p, q, True))
            lines.append(seg)
            done.add((i, j))
            print(f"外壁を閉じる（端がつながっていない2点を結ぶ）: {p} - {q}")
    return added


def furring_offsets(offsets, placements, cols, cfg):
    """壁のふかし (user 11/8): a frame deeper than 155 (the 192 of the 3枚連動片引戸) is set with its
    slide side's face 12.5 proud of the board face; the other side (the 中方立 side) stands out further
    and the wall is built out there. The wall lies on both sides of the door (never across it): the
    board-face line on the 中方立 side moves out to where the frame stands 12.5 proud of it, from each
    frame end to the next column line, with returns back to the board face at the column lines. The
    column-face line (52.5) is not moved, so the board face and the 52.5 line are drawn apart."""
    fc = cfg.get("furring")
    if not fc or not placements:
        return offsets
    board = cfg["offsets_mm"]["board_face_inside"]
    proud = fc["proud_mm"]
    lay = cfg["layers"]["wall"]
    out = list(offsets)
    for pl in placements:
        wc = fc["words"].get(pl["word"])
        if not wc:
            continue
        depth = wc["depth_mm"]
        h, L = pl["horizontal"], pl["line"]
        lo, hi = pl["span"]
        d_plus = pl["sy"] if h else -pl["sy"]  # model direction of the block's local +y (the 中方立 side)
        new_face = -(board + proud) + depth - proud
        c_old, c_new = L + d_plus * board, L + d_plus * new_face
        along = [(c[0] if h else c[1]) for c in cols if abs((c[1] if h else c[0]) - L) < 1]
        left = [t for t in along if t <= lo + 1]
        right = [t for t in along if t >= hi - 1]
        # the 中方立 stands post_mm from the frame's left end (the block's local left: the model's low end when
        # the block runs along +, else the high end); the built-out wall runs from the 中方立 to the column line
        pw_ = wc.get("post_w_mm", 24.0)
        post = (lo + wc["post_mm"] + pw_) if pl["sx"] == 1 else (hi - wc["post_mm"] - pw_)  # (stop before the post: its far face)
        zones = [(max(left) if left else lo, lo), (post, min(right) if right else hi)] if pl["sx"] == 1 else                 [(max(left) if left else lo, post), (hi, min(right) if right else hi)]
        kept = []
        for (p, q), c, lb in out:
            on_old = lb == "石膏ボード面" and abs((p[1] if h else p[0]) - c_old) < 0.5 and abs((q[1] if h else q[0]) - c_old) < 0.5
            if not on_old:
                kept.append(((p, q), c, lb))
                continue
            a_, b_ = sorted((p[0], q[0]) if h else (p[1], q[1]))
            cuts = [(a_, b_)]
            for za, zb in zones:  # the part of this line inside a zone beside the door is replaced
                nxt = []
                for s_, e_ in cuts:
                    if e_ <= za or s_ >= zb:
                        nxt.append((s_, e_))
                        continue
                    if s_ < za:
                        nxt.append((s_, za))
                    if e_ > zb:
                        nxt.append((zb, e_))
                cuts = nxt
            for s_, e_ in cuts:
                if e_ - s_ > 0.5:
                    kept.append((((s_, c_old), (e_, c_old)) if h else ((c_old, s_), (c_old, e_)), c, lb))
            for za, zb in zones:
                s_, e_ = max(za, a_), min(zb, b_)
                if e_ - s_ > 0.5:
                    kept.append((((s_, c_new), (e_, c_new)) if h else ((c_new, s_), (c_new, e_)), c, lb))
        out = kept
        # the board surface between the 中方立 and the frame's far end: a plain line; and the board thickness
        # (12.5) behind each built-out line (user 11/8)
        d_end = (post, hi) if pl["sx"] == 1 else (lo, post)
        if d_end[1] - d_end[0] > 0.5:
            out.append((((d_end[0], c_old), (d_end[1], c_old)) if h else ((c_old, d_end[0]), (c_old, d_end[1])), lay["finish_color"], "石膏ボード面"))
        c_thick = c_new - d_plus * 12.5  # the board's thickness lies behind its surface, toward the wall centre
        for za, zb in zones:
            if zb - za > 0.5:
                out.append((((za, c_thick), (zb, c_thick)) if h else ((c_thick, za), (c_thick, zb)), lay["structure_color"], "ボード厚み"))
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


SPUR_MAX = [0.0]


def cfg_spur_max_mm():
    return SPUR_MAX[0]


def classify_boundary(walls, fp):
    ring = fp.exterior
    for w in walls:
        seg = LineString([w.pt(w.a), w.pt(w.b)])
        w.ext = ring.buffer(1).contains(seg)
    # an exterior wall that lies along the outline but sticks out past it for one short stretch (a
    # stub toward a balcony, at most spur_max long) is exterior as a whole: the 910 stub under the
    # ホール's balcony side of プラン3 2F was drawn as an interior wall (user 11/7). Walls that run
    # inside the outline, or lie wholly off it, keep the old reading.
    if cfg_spur_max_mm():
        inside = fp.buffer(-1)
        band = ring.buffer(1)
        for w in walls:
            if w.ext or not w.ext_in:
                continue
            seg = LineString([w.pt(w.a), w.pt(w.b)])
            if inside.intersects(seg) and seg.intersection(inside).length >= 1.0:
                continue
            on = seg.intersection(band)
            if on.is_empty or on.length < 1.0:
                continue  # wholly off the outline: old reading
            off = seg.length - on.length
            if 800.0 <= off <= cfg_spur_max_mm() + 1.0:  # (one 910 module: a stub, not a short return)
                w.ext = True
                print(f"外壁の短い突き出し（{off:.0f}mm）も外壁にする: {w.pt(w.a)} - {w.pt(w.b)}")


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


def shift_entrance_doors(placements, walls, cfg, runs=()):
    """玄関ドアは、1820の間（壁の長さがちょうど1820）にある時、吊り元側へ200ずらす。吊り元側に玄関収納が
    あれば戸先側へ200ずらす。開き勝手（吊り元の向き）は変えない（user 11/10, プラン5 1F）。
    玄関収納 = 取り除いた家具のうち、玄関まわりで壁に直角に付くもの（FURN の run）。"""
    sh = cfg["placement"].get("entrance_shift_mm")
    if not sh:
        return
    bay = cfg["placement"].get("entrance_bay_mm", 1820)
    for pl in placements:
        if pl["word"] not in ENTRANCE_WORDS or not pl.get("hinge_known"):
            continue
        h, line = pl["horizontal"], pl["line"]
        lo, hi = pl["span"]
        w = next((w for w in walls if w.h == h and abs(w.c - line) < 1 and w.a - 1 <= lo and hi <= w.b + 1), None)
        if w is None or abs((w.b - w.a) - bay) > 5:
            continue
        hinge_dir = 1 if pl["sx"] == -1 else -1  # the hinge end lies toward +along when the block runs toward -along
        hinge_end = w.b if hinge_dir == 1 else w.a
        stor = list(FURN) + [(r["horizontal"], r["c_mm"], r["a_mm"], r["b_mm"]) for r in runs]
        storage = any(fh != h and abs(c - hinge_end) <= 500 and min(fa, fb) - 600 <= line <= max(fa, fb) + 600
                      for fh, c, fa, fb in stor)
        d = -hinge_dir if storage else hinge_dir
        hc = cfg["column"]["size_mm"] / 2
        if not storage:
            # 吊り元側の枠の端を、壁端の柱の面から sh（200）離す（user 11/15。「200ずらす」だと、ドアの幅によっては
            # 柱に食い込んだ：プラン1 1F の W1126）
            gap = cfg["placement"].get("entrance_hinge_gap_mm", sh)  # 150（user 11/15。200から変更）
            target = (w.b - hc - gap) if hinge_dir > 0 else (w.a + hc + gap)
            move = target - hi if hinge_dir > 0 else target - lo
        else:
            move = d * sh
            # 戸先側へずらす時も、壁端の柱の面（壁端から65）に枠が食い込まない範囲までにする
            edge = hc + 12.5
            room = (w.b - edge - hi) if d > 0 else (lo - (w.a + edge))
            if room < sh:
                move = d * max(0.0, room)
        x, y = pl["pos"]
        pl["pos"] = (x + move, y) if h else (x, y + move)
        pl["span"] = (lo + move, hi + move)
        if storage:
            print(f"玄関ドアを戸先側へ{abs(move):g}ずらす（{bay}の間、吊り元側に玄関収納あり）: {lo:.1f} -> {lo + move:.1f}")
        else:
            print(f"玄関ドアの吊り元側の枠を、柱の面から{gap:g}離す（{bay}の間、吊り元側に玄関収納なし）: {lo:.1f} -> {lo + move:.1f}")


def place_columns(walls, nodes, cfg, upper_walls, fittings=(), fp=None, mullions=()):
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
            # 片引戸の中方立の付近（開口側の半分）にも置かない（user 11/13、プラン5 1F (2730,10920)）
            if any(abs((p[1] if fh else p[0]) - line) < 1 and lo - hc < (p[0] if fh else p[1]) < hi + hc
                   for fh, line, lo, hi in mullions):
                print(f"片引戸の中方立のそばに柱を置かない: {p} ({k})")
                continue
            cols[p] = k
    # 外壁の出隅から910の位置は必ず柱にする (user 11/12). A convex corner = the point just inside
    # the angle between its two walls lies in the footprint. Not where a fitting would be in the way.
    if fp is not None:
        DIRS = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}
        for p, kind in list(cols.items()):
            arms = nodes.get(p)
            if kind != "L" or not arms or len(arms) != 2:
                continue
            d1, d2 = (DIRS[a] for a in arms)
            if not fp.contains(Point(p[0] + 20 * (d1[0] + d2[0]), p[1] + 20 * (d1[1] + d2[1]))):
                continue  # 入隅
            for d in (d1, d2):
                q = (p[0] + 910 * d[0], p[1] + 910 * d[1])
                h = d[1] == 0
                w = next((w for w in walls if w.ext and w.h == h and w.contains(p) and w.contains(q)), None)
                if w is None:
                    continue
                t = q[0] if h else q[1]
                if any(fh == h and abs(line - w.c) < 1 and t + hc > lo + EPS and t - hc < hi - EPS
                       for fh, line, lo, hi, _w in fittings):
                    continue
                if q not in cols:
                    cols[q] = "出隅910"
    defaults = []  # remainder placed on the default side without upper-floor evidence
    for w in walls:
        spans = [(lo, hi) for fh, line, lo, hi, _word in fittings if fh == w.h and abs(line - w.c) < 1]
        # no side columns for the entrance door, a garage shutter, or a fitting that took a
        # junction column's place (not needed there, user 10/27)
        side_spans = [(lo, hi) for i, (fh, line, lo, hi, word) in enumerate(fittings)
                      if fh == w.h and abs(line - w.c) < 1 and word not in ENTRANCE_WORDS + ("シャッター",)
                      and i not in no_sides]

        # 直交する壁の建具の枠にも柱を重ねない（プラン3 2F：片引戸の脇の柱 (2730,7280) が、
        # 交わる壁の縦すべり窓の枠に乗っていた。自動検査 [D] で発見、11/13）
        # 引戸は除く（戸袋は壁の中。プラン1 1F の正解図は、片引戸の戸袋の壁に両開きの脇の柱 (3640,7735) がある）
        cross = [(line, lo, hi) for fh, line, lo, hi, word in fittings
                 if fh != w.h and "引戸" not in word and "引違い戸" not in word]
        cross += [(line, lo, hi) for fh, line, lo, hi in mullions if fh != w.h]

        def clear(t, spans=spans, cross=cross, w=w):
            if not all(t + hc <= lo + EPS or t - hc >= hi - EPS for lo, hi in spans):
                return False
            return not any(abs(t - line) < hc and w.c + hc > lo + EPS and w.c - hc < hi - EPS
                           for line, lo, hi in cross)

        # 室内建具の両脇には、グリッド（455）の交点に柱を置く（間柱ではなく柱）。建具の幅が小さくても
        # 910のグリッドからずらさない：650・735・780も910の柱間に置く（user 11/13）
        for i, (fh, line, lo, hi, word) in enumerate(fittings):
            if fh != w.h or abs(line - w.c) >= 1 or word not in INTERIOR_DOORS:
                continue
            for t in (math.floor((lo - hc + EPS) / grid) * grid, math.ceil((hi + hc - EPS) / grid) * grid):
                if w.contains(w.pt(t)) and clear(t):
                    cols.setdefault(w.pt(t), "fitting")
        stops = sorted({(p[0] if w.h else p[1]) for p in cols if w.contains(p)})
        for s, e in zip(stops, stops[1:]):
            if e - s > pitch + EPS:
                for lo, hi in side_spans:
                    if s < lo and hi < e:
                        for t in (math.floor((lo - hc + EPS) / grid) * grid, math.ceil((hi + hc - EPS) / grid) * grid):
                            if s + EPS < t < e - EPS and clear(t):
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
                    # a leftover of 455 (e.g. 2275 = 1820 + 455) leaves 455 on one side and 1820 on
                    # the other: unbalanced. The last bay (pitch + 455) is split 910 / 1365 instead
                    # (user 11/12). 910 from the end that puts the column on the 910 grid.
                    if abs(r - grid) < EPS:
                        base_a = [s + i * pitch for i in range(1, n)]
                        cand_a = base_a + [e - (pitch + r) + 910]  # 910 from the last full column
                        cand_b = base_a + [e - 910]                # 1365 from it
                        cand = [t for t in (cand_a if on910(cand_a) >= on910(cand_b) else cand_b)
                                if s + EPS < t < e - EPS and clear(t)]
                        if len(cand) == len(choice):
                            choice = cand
            for t in choice:
                cols.setdefault(w.pt(t), "pitch")
            if marked:
                defaults.append(w.pt((s + e) / 2))
    return cols, defaults


# ---------------------------------------------------------------- output
def _generic_ub(kind):
    """UB の部品がない時（配布用）の簡単な形：外形・内側・浴槽・ドアの枠（用紙mm、1/50）。
    向きは部品集と同じ（浴槽が左の壁。beside はドアが下の壁の右寄り、facing は右の壁の下寄り）"""
    S = 50.0
    def rect(x0, y0, x1, y1):
        return [{"type": "LINE", "start_x": a / S, "start_y": b / S, "end_x": c / S, "end_y": d / S,
                 "base": {"pen_color": 1, "pen_style": 1}}
                for a, b, c, d in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0))]
    if kind == "beside":
        es = rect(32, 47, 1668, 1683) + rect(50, 65, 1650, 1665) + rect(90, 120, 780, 1610) + rect(900, 0, 1640, 47)
    else:
        es = rect(32, 17, 1668, 1653) + rect(50, 35, 1650, 1635) + rect(90, 90, 780, 1580) + rect(1668, 45, 1715, 790)
    return es


def ub_layout(ub_path, cfg):
    """読み取った UB（<階>_UB.json）に、UB の部品を置く（user 11/16）。部品の外形の中心を UB の壁芯の
    真ん中に置き、浴槽の壁・ドアの壁・ドアの寄りが合うように回す／裏返す。
    返す：(置き方のリスト [{block, ents, pos, rot, sx, sy}], ドアの開口 [{horizontal, line, span, word}])"""
    if not Path(ub_path).exists():
        return [], []
    ubs = json.load(open(ub_path, encoding="utf-8"))
    if not ubs:
        return [], []
    up = cfg.get("ub_part", {})
    S = cfg["layers"]["scale"]
    blocks = {}
    lib_p = Path(__file__).resolve().parent.parent / up.get("library", "").replace("\\", "/") if up.get("library") else None
    if lib_p and lib_p.is_file():
        blocks = {b["name"]: b["entities"] for b in ezjww.read_document(str(lib_p))["block_defs"]}
    canon = {"beside": {"tub": [-1, 0], "door": [0, -1], "door_end": [1, 0]},
             "facing": {"tub": [-1, 0], "door": [1, 0], "door_end": [0, -1]}}
    out, cuts = [], []
    for u in ubs:
        door = u.get("door")
        kind = "facing" if door and door[0] == -u["tub"][0] and door[1] == -u["tub"][1] else "beside"
        spec_ = {**canon[kind], **up.get(kind, {})}
        name = spec_.get("block")
        ents = blocks.get(name) if name else None
        if ents is None:
            name, ents = f"UB_{kind}", _generic_ub(kind)
        # 部品の外形（1000mm 以上の線の範囲）と、その中心
        long_ = [e for e in ents if e["type"] == "LINE"
                 and math.hypot(e["end_x"] - e["start_x"], e["end_y"] - e["start_y"]) * S >= 1000]
        xs = [v * S for e in long_ for v in (e["start_x"], e["end_x"])]
        ys = [v * S for e in long_ for v in (e["start_y"], e["end_y"])]
        bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
        cl = ((bx0 + bx1) / 2, (by0 + by1) / 2)
        best = None
        for rot in (0, 90, 180, 270):
            for sx in (1, -1):
                a = math.radians(rot)
                f = lambda v, a=a, sx=sx: (round(v[0] * sx * math.cos(a) - v[1] * math.sin(a)),  # noqa: E731
                                           round(v[0] * sx * math.sin(a) + v[1] * math.cos(a)))
                ok = f(spec_["tub"]) == tuple(u["tub"])
                if door:
                    ok = ok and f(spec_["door"]) == tuple(door) and f(spec_["door_end"]) == tuple(u["door_end"])
                if ok and best is None:
                    best = (rot, sx, f)
        if best is None:
            print(f"UB @{u['center_mm']}：部品の向きが合わない（浴槽{u['tub']} ドア{door}）。置かない")
            continue
        rot, sx, f = best
        a = math.radians(rot)

        def tp(x, y, a=a, sx=sx):
            return (x * sx * math.cos(a) - y * math.sin(a), x * sx * math.sin(a) + y * math.cos(a))
        c_ = tp(*cl)
        pos = (u["center_mm"][0] - c_[0], u["center_mm"][1] - c_[1])
        out.append({"block": name, "ents": ents, "pos": pos, "rot": rot, "sx": sx, "sy": 1, "kind": kind})
        print(f"UB の部品: {name} 中心{tuple(u['center_mm'])} 回転{rot} 裏返し{'あり' if sx < 0 else 'なし'}")
        if not door:
            continue
        # ドアの開口：部品の、ドアの側で外形の外に出ている線（ドアの枠）の範囲。壁はそこで切る
        dl = spec_["door"]
        outside = []
        for e in ents:
            if e["type"] != "LINE":
                continue
            pts = [(e["start_x"] * S, e["start_y"] * S), (e["end_x"] * S, e["end_y"] * S)]
            # （外形の角の印＝かぎの線は外す：外形の端から 10 以内）
            if all(((x - bx1 > 1) if dl[0] > 0 else (bx0 - x > 1) if dl[0] < 0 else (y - by1 > 1) if dl[1] > 0 else (by0 - y > 1))
                   and ((by0 + 10 < y < by1 - 10) if dl[0] else (bx0 + 10 < x < bx1 - 10)) for x, y in pts):
                outside += pts
        h = bool(u["door_horizontal"])
        if outside:
            ws = [tp(x, y) for x, y in outside]
            ts = [(pos[0] + p[0]) if h else (pos[1] + p[1]) for p in ws]
            span = (min(ts), max(ts))
        else:
            span = tuple(u["door_span_mm"])
        if u.get("door_line_mm") is None:
            continue
        cuts.append({"horizontal": h, "line": u["door_line_mm"], "span": span, "word": "UB開口", "block": None,
                     "sx": 1, "sy": 1, "outward": 0})
        # 開口部の中心線（室内の建具は全部、ドア芯から200ずつ、user 11/15）
        m_, L_ = (span[0] + span[1]) / 2, u["door_line_mm"]
        out[-1]["centre"] = (((m_, L_ - 200), (m_, L_ + 200)) if h else ((L_ - 200, m_), (L_ + 200, m_)))
        print(f"UB のドアの開口: {'横' if h else '縦'} 線{u['door_line_mm']:g} {span[0]:.1f}〜{span[1]:.1f}")
    return out, cuts


def _generic_kitchen():
    """キッチンの部品がない時（配布用）の簡単な形：天板 2550×650・コンロ・シンク（用紙mm、1/50）。
    向きは部品集と同じ（コンロが左の端、背中が上）"""
    S = 50.0
    es = []
    for x0, y0, x1, y1 in ((0, 0, 2550, 650), (150, 70, 750, 600), (1500, 80, 2350, 580)):
        for a, b, c, d in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
            es.append({"type": "LINE", "start_x": a / S, "start_y": b / S, "end_x": c / S, "end_y": d / S,
                       "base": {"pen_color": 1, "pen_style": 1}})
    for cx_, cy_ in ((300, 250), (600, 250), (450, 470)):
        es.append({"type": "CIRCLE", "center_x": cx_ / S, "center_y": cy_ / S, "radius": 90 / S,
                   "base": {"pen_color": 1, "pen_style": 1}})
    return es


def kitchen_layout(k_path, cfg, walls=()):
    """読み取ったキッチン（<階>_キッチン.json）に、キッチンの部品を置く（user 11/16）。
    コンロの端を、その先の壁の面（石膏ボードの面）に付ける。作業する側の天板の端を元の図の天板の端に
    そろえる（背中に壁があれば、背中を壁の面に付ける）。返す：[{block, ents, pos, rot, sx, sy, kind}]"""
    if not Path(k_path).exists():
        return []
    ks = json.load(open(k_path, encoding="utf-8"))
    if not ks:
        return []
    kp = cfg.get("kitchen_part", {})
    S = cfg["layers"]["scale"]
    board = cfg["offsets_mm"]["board_face_inside"]
    name, ents = kp.get("block"), None
    lib_p = Path(__file__).resolve().parent.parent / kp.get("library", "").replace("\\", "/") if kp.get("library") else None
    if name and lib_p and lib_p.is_file():
        ents = {b["name"]: b["entities"] for b in ezjww.read_document(str(lib_p))["block_defs"]}.get(name)
    if ents is None:
        name, ents = "キッチン_I型_W2550", _generic_kitchen()
    stove0, back0 = kp.get("stove", [-1, 0]), kp.get("back", [0, 1])
    # 部品の天板の外形（長い線の範囲）
    long_ = [e for e in ents if e["type"] == "LINE"
             and math.hypot(e["end_x"] - e["start_x"], e["end_y"] - e["start_y"]) * S >= 500]
    xs = [v * S for e in long_ for v in (e["start_x"], e["end_x"])]
    ys = [v * S for e in long_ for v in (e["start_y"], e["end_y"])]
    corners = [(x, y) for x in (min(xs), max(xs)) for y in (min(ys), max(ys))]
    depth = max(ys) - min(ys)
    out = []
    for k in ks:
        best = None
        for rot in (0, 90, 180, 270):
            for sx in (1, -1):
                a = math.radians(rot)
                f = lambda v, a=a, sx=sx: (round(v[0] * sx * math.cos(a) - v[1] * math.sin(a)),  # noqa: E731
                                           round(v[0] * sx * math.sin(a) + v[1] * math.cos(a)))
                if best is None and f(stove0) == tuple(k["stove"]) and f(back0) == tuple(k["back"]):
                    best = (rot, sx)
        if best is None:
            continue
        rot, sx = best
        a = math.radians(rot)
        tc = [(x * sx * math.cos(a) - y * math.sin(a), x * sx * math.sin(a) + y * math.cos(a)) for x, y in corners]
        h = k["horizontal"]
        ti, ui = (0, 1) if h else (1, 0)
        t_lo, t_hi = min(p[ti] for p in tc), max(p[ti] for p in tc)
        u_lo, u_hi = min(p[ui] for p in tc), max(p[ui] for p in tc)
        ss = k["stove"][ti]
        bs = k["back"][ui]
        dt = k["stove_face_mm"] - (t_hi if ss > 0 else t_lo)
        if k.get("back_wall_mm") is not None:
            back_face = k["back_wall_mm"] - bs * board
            du = back_face - (u_hi if bs > 0 else u_lo)
        else:
            du = k["front_mm"] - (u_lo if bs > 0 else u_hi)
        pos = (dt, du) if h else (du, dt)
        out.append({"block": name, "ents": ents, "pos": pos, "rot": rot, "sx": sx, "sy": 1, "kind": "kitchen"})
        print(f"キッチンの部品: {name} {'横' if h else '縦'} コンロ{tuple(k['stove'])} 背中{tuple(k['back'])}"
              f" 回転{rot} 裏返し{'あり' if sx < 0 else 'なし'}（奥行き{depth:.0f}、元の図の天板 {k['depth_mm']}）")
    return out


def write_jww(path, walls, offsets, cols, cfg, center, placements=(), lib=None, extras=(), toilets=(), ubs=()):
    S = cfg["layers"]["scale"]
    L = cfg["layers"]
    d = ezjww.new(paper_size=3)
    import jww_env
    jww_env.apply_env(d)  # 環境設定 参考図面の .JWF（色・線幅・線種・レイヤグループ）
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
    # 石膏ボードなしの仕様（draw_board: false）：寸法・建具はボード12.5の時のまま、石膏ボードの線だけ描かない（user 11/15）
    no_board = not cfg.get("draw_board", True)
    for (p, q), color, lb in offsets:
        if no_board and lb == "石膏ボード面":
            continue
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
        # 建具 (windows / doors) sit on a layer of their own, apart from the 柱 and walls: a range
        # selection of a part's base point (a column centre) must not pick the column up too
        # (user 11/12, プラン4 1F; 参考図面 keeps 建具 on layer 5, columns on others)
        fl = L.get("fitting", wl)["layer"]
        if L.get("fitting_as_block", False):
            import copy
            for name in dict.fromkeys(pl["block"] for pl in placements):
                ents = copy.deepcopy(by_name[name])
                for e in ents:
                    e["base"]["layer"] = fl
                d.add_block(name, ents)
            for pl in placements:
                msp.add_block_ref(pl["block"], ((pl["pos"][0] - cx) / S, (pl["pos"][1] - cy) / S),
                                  scale_x=float(pl["sx"]), scale_y=float(pl["sy"]), rotation=float(pl["rot"]),
                                  jwwattribs={"layer": fl})
        else:
            # no blocks (グループ化なし, user 11/12): the part's lines and arcs are written out as
            # plain entities, moved / mirrored / rotated into place
            import math
            for pl in placements:
                ang = math.radians(pl["rot"])
                ca, sa = math.cos(ang), math.sin(ang)
                sx, sy = pl["sx"], pl["sy"]

                def tr(u, v, pl=pl, ca=ca, sa=sa, sx=sx, sy=sy):
                    u, v = u * S * sx, v * S * sy
                    return (pl["pos"][0] + u * ca - v * sa, pl["pos"][1] + u * sa + v * ca)
                for e in by_name[pl["block"]]:
                    if e["base"]["pen_color"] == 6 and pl["word"] in INTERIOR_DOORS:
                        continue  # 室内建具の青（色6）の線は要らない（user 11/13）。水色（色1）だけ
                    at = {"layer": fl, "pen_color": e["base"]["pen_color"], "pen_style": e["base"]["pen_style"]}
                    if e["type"] == "LINE":
                        p1, p2 = tr(e["start_x"], e["start_y"]), tr(e["end_x"], e["end_y"])
                        msp.add_line(((p1[0] - cx) / S, (p1[1] - cy) / S), ((p2[0] - cx) / S, (p2[1] - cy) / S),
                                     jwwattribs=at)
                    elif e["type"] == "ARC":
                        c = tr(e["center_x"], e["center_y"])
                        a0, sw = e["start_angle"], e["arc_angle"]  # radians, CCW
                        a1 = a0 + sw
                        # direction of an angle after the mirror and the rotation
                        mv = lambda a: math.atan2(sy * math.sin(a), sx * math.cos(a)) + ang  # noqa: E731
                        n0, n1 = mv(a0), mv(a1)
                        if sx * sy < 0:  # a mirror reverses the sweep: the end becomes the start
                            n0, n1 = n1, n0
                        st = round(math.degrees(n0), 4) % 360.0
                        st = 0.0 if st >= 359.9999 else st
                        swd = round(math.degrees(n1 - n0), 4) % 360.0 or 360.0
                        msp.add_arc(((c[0] - cx) / S, (c[1] - cy) / S), e["radius"] * abs(sx), st, swd, jwwattribs=at)
    # 開口部の中心線：出入りできる開口に、壁と直角に、ドア芯から200ずつ（計400）。建具と同じ色（user 11/15）
    for (p, q), color in opening_centre_lines(placements, lib, S):
        msp.add_line(((p[0] - cx) / S, (p[1] - cy) / S), ((q[0] - cx) / S, (q[1] - cy) / S),
                     jwwattribs={"layer": L.get("fitting", wl)["layer"], "pen_color": color, "pen_style": 1})
    # 便器：検出した向き・位置に置く（user 11/13）。形は cfg["toilet_part"]（部品の json のパス、
    # ROOT から）。"generic" か、ファイルがなければ汎用の形（配布用、11/14）
    if toilets:
        import math
        tp = cfg.get("toilet_part", "generic")
        tp_path = Path(__file__).resolve().parent.parent / tp
        part = generic_toilet() if tp == "generic" or not tp_path.exists() else json.load(open(tp_path, encoding="utf-8"))
        fd_ = cfg["furniture_draw"]
        at_ = {"layer": fd_["layer"], "pen_color": fd_["color"], "pen_style": fd_["style"]}
        half = part["長さ_mm"] / 2
        for tl in toilets:
            fx, fy = tl["front"]
            ox, oy = tl["tip_mm"][0] - fx * half, tl["tip_mm"][1] - fy * half  # 便器の中心
            rot = math.atan2(fy, fx)

            def T(u, v, ox=ox, oy=oy, fx=fx, fy=fy):
                return (((ox + u * fx - v * fy) - cx) / S, ((oy + u * fy + v * fx) - cy) / S)
            for e in part["entities"]:
                if e["t"] == "L":
                    msp.add_line(T(e["p"][0], e["p"][1]), T(e["p"][2], e["p"][3]), jwwattribs=at_)
                elif e["t"] == "A":
                    st = round(math.degrees(e["a0"] + rot), 4) % 360.0
                    st = 0.0 if st >= 359.9999 else st
                    msp.add_arc(T(*e["c"]), e["r"] / S, st, round(math.degrees(e["sw"]), 4) % 360.0 or 360.0, jwwattribs=at_)
                else:
                    msp.add_circle(T(*e["c"]), e["r"] / S, jwwattribs=at_)
    # UB（ユニットバス）：読み取った向きに部品を置く（家具・設備のレイヤー、user 11/16）
    if ubs:
        fd_ = cfg["furniture_draw"]
        for ub in ubs:
            if ub.get("centre"):
                (p, q) = ub["centre"]
                msp.add_line(((p[0] - cx) / S, (p[1] - cy) / S), ((q[0] - cx) / S, (q[1] - cy) / S),
                             jwwattribs={"layer": fd_["layer"], "pen_color": fd_["color"], "pen_style": 1})
            ang = math.radians(ub["rot"])
            ca, sa = math.cos(ang), math.sin(ang)
            sx, sy = ub["sx"], ub["sy"]

            def tu(u, v, ub=ub, ca=ca, sa=sa, sx=sx, sy=sy):
                u, v = u * S * sx, v * S * sy
                return ((ub["pos"][0] + u * ca - v * sa - cx) / S, (ub["pos"][1] + u * sa + v * ca - cy) / S)
            for e in ub["ents"]:
                at = {"layer": fd_["layer"], "pen_color": fd_["color"], "pen_style": e["base"].get("pen_style", 1)}
                if e["type"] == "LINE":
                    msp.add_line(tu(e["start_x"], e["start_y"]), tu(e["end_x"], e["end_y"]), jwwattribs=at)
                elif e["type"] == "CIRCLE":
                    msp.add_circle(tu(e["center_x"], e["center_y"]), e["radius"], jwwattribs=at)
                elif e["type"] == "ARC":
                    a0, sw = e["start_angle"], e["arc_angle"]
                    fl_ = e.get("flatness", 1.0) or 1.0
                    if abs(fl_ - 1.0) > 1e-6:  # 楕円の弧：点を結んだ線で描く
                        tl = e.get("tilt_angle", 0.0) or 0.0
                        n = max(8, int(abs(sw) * 20))
                        pts = []
                        for j in range(n + 1):
                            t = a0 + sw * j / n
                            ux, uy = e["radius"] * math.cos(t), e["radius"] * fl_ * math.sin(t)
                            pts.append(tu(e["center_x"] + ux * math.cos(tl) - uy * math.sin(tl),
                                          e["center_y"] + ux * math.sin(tl) + uy * math.cos(tl)))
                        for p, q in zip(pts, pts[1:]):
                            msp.add_line(p, q, jwwattribs=at)
                        continue
                    mv = lambda a, sx=sx, sy=sy, ang=ang: math.atan2(sy * math.sin(a), sx * math.cos(a)) + ang  # noqa: E731
                    n0, n1 = mv(a0), mv(a0 + sw)
                    if sx * sy < 0:
                        n0, n1 = n1, n0
                    st = round(math.degrees(n0), 4) % 360.0
                    st = 0.0 if st >= 359.9999 else st
                    swd = round(math.degrees(n1 - n0), 4) % 360.0 or 360.0
                    msp.add_arc(tu(e["center_x"], e["center_y"]), e["radius"], st, swd, jwwattribs=at)
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


def walk_in_window(code, word=None):
    """出入りできる窓か：窓の記号（例 06020T・16020H）の種類が縦すべり系（T を含む）か引違い（H）で、高さ 1800 以上（user 11/15）。
    高さは記号の4・5桁目（16020 の「20」＝2000）。18 以上なら 1800 以上。
    記号が読めず高さが分からない時は、縦すべり系・引違いの窓なら引く（迷ったら引く、user 11/16）"""
    if code and len(code) >= 6 and code[:5].isdigit():
        return ("T" in code[5:] or "H" in code[5:]) and int(code[3:5]) >= 18
    return bool(word) and ("縦すべり" in word or word == "引違い窓")


def opening_centre_lines(placements, lib, S, half=200.0):
    """出入りできる開口（室内建具・玄関ドアの全部と、出入りできる窓）の中心線（user 11/15）。
    開口の幅の真ん中に、壁と直角に、ドア芯（枠の見込みの真ん中）から両側へ half ずつ。
    片引戸は戸袋でない側（部品の局所で左＝枠と中方立の間）、アウトセット片引戸は枠の外から外の真ん中。
    返り値 [((p, q), 線色)]（モデル mm）"""
    import math
    from place_fittings import outset_jambs
    entry = {v[0]: v for v in (lib or {}).values()}
    out = []
    for pl in placements:
        is_win = pl["kind"] == "window" if pl.get("kind") else pl["word"] in pf_WINDOWS
        if is_win and not walk_in_window(pl.get("label"), pl.get("word")):
            continue
        en = entry.get(pl["block"])
        if en is None:
            continue
        ents, word, W, off = en[1], en[3], float(en[4]), float(en[5])
        x0, x1 = off, off + W
        perp = []  # 壁と直角の線（局所 mm）: (x, y_lo, y_hi, 線色)
        for e in ents:
            if e["type"] != "LINE":
                continue
            ax, ay, bx, by = (v * S for v in (e["start_x"], e["start_y"], e["end_x"], e["end_y"]))
            if abs(ax - bx) < 0.5 and abs(ay - by) >= 60:
                perp.append(((ax + bx) / 2, min(ay, by), max(ay, by), e["base"]["pen_color"]))
        jamb = [t for t in perp if x0 - 1 <= t[0] <= x0 + 25]
        if jamb:
            j = max(jamb, key=lambda t: t[2] - t[1])
            yc, color = (j[1] + j[2]) / 2, j[3]
        else:
            yc, color = 0.0, 1
        if word == "アウトセット片引戸":
            jj = outset_jambs(ents, S)
            xc = (jj[0] + jj[1]) / 2 if jj else x0 + W / 2
        elif "縦すべり" in word and "FIX" in word:
            # 縦すべり+FIX：出入りできるのは縦すべりの障子の側だけ。開いた障子の線（枠の見込みから
            # 50以上はみ出す線）がある側の、枠の内側から方立の面までの真ん中（user 11/15）
            p40 = [((e["start_x"] + e["end_x"]) * S / 2) for e in ents if e["type"] == "LINE"
                   and abs(e["start_x"] - e["end_x"]) * S < 0.5 and abs(e["start_y"] - e["end_y"]) * S >= 40]
            perp = [(x, 0, 0, 0) for x in p40]  # 方立・枠の内側を探すため、短い縦の線（40以上）も見る
            mull = [t[0] for t in perp if x0 + 0.25 * W <= t[0] <= x0 + 0.75 * W]
            half_d = (jamb and (j[2] - j[1]) / 2) or 65.0
            sash = []
            for e in ents:
                if e["type"] == "LINE":
                    for ex, ey in ((e["start_x"], e["start_y"]), (e["end_x"], e["end_y"])):
                        if abs(ey * S - yc) > half_d + 50:
                            sash.append((e["start_x"] + e["end_x"]) * S / 2)
            if mull and sash:
                mid = (min(mull) + max(mull)) / 2
                if sum(sash) / len(sash) < mid:
                    ins = [t[0] for t in perp if x0 <= t[0] <= x0 + 80]
                    xc = ((max(ins) if ins else x0) + min(mull)) / 2
                else:
                    ins = [t[0] for t in perp if x1 - 80 <= t[0] <= x1]
                    xc = (max(mull) + (min(ins) if ins else x1)) / 2
            else:
                xc = x0 + W / 2
        elif word == "片引戸":
            mull = [t[0] for t in perp if x0 + 0.3 * W <= t[0] <= x0 + 0.7 * W]
            inner = [t[0] for t in perp if x0 <= t[0] <= x0 + 40]
            xc = (max(inner) + min(mull)) / 2 if mull and inner else x0 + W / 4
        else:
            xc = x0 + W / 2
        ang = math.radians(pl["rot"])
        ca, sa = math.cos(ang), math.sin(ang)

        def tr(u, v, pl=pl, ca=ca, sa=sa):
            u, v = u * pl["sx"], v * pl["sy"]
            return (pl["pos"][0] + u * ca - v * sa, pl["pos"][1] + u * sa + v * ca)
        out.append(((tr(xc, yc - half), tr(xc, yc + half)), color))
    return out


def generic_toilet(length=736.0, width=386.0):
    """汎用の便器（メーカーなし）：便器の中心が原点、前（とがった側）が +x。後ろにタンク、前にだ円の便座"""
    import math
    hl, hw = length / 2, width / 2
    tank = 180.0
    ents = []
    pts = [(-hl, -hw), (-hl + tank, -hw), (-hl + tank, hw), (-hl, hw)]
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        ents.append({"t": "L", "p": [a[0], a[1], b[0], b[1]]})
    a_, b_ = (length - tank) / 2 + 2, hw - 15  # だ円（タンクの前から先端まで）
    c0 = -hl + tank - 2 + a_
    n = 40
    for k in range(n):
        t0, t1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        ents.append({"t": "L", "p": [c0 + a_ * math.cos(t0), b_ * math.sin(t0), c0 + a_ * math.cos(t1), b_ * math.sin(t1)]})
    return {"名前": "汎用の便器", "長さ_mm": length, "幅_mm": width, "基準": "便器の中心", "前": "+x", "entities": ents}


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
            elif pl["word"] in ("玄関片開き", "玄関親子") and lib and pl["block"] in lib:
                # the frame's outer ears stand past the span: the exterior layer lines (通気層・外壁面)
                # stop at the first frame line they meet instead of running into the ears
                # (user 11/12, プラン4 1F 玄関ドア: the two 外壁材 lines stuck out)
                ear = frame_crossings(pl, lib, S, h, c_)
                lo = min([lo] + [t for t in ear if lo - 60 <= t < lo])
                hi = max([hi] + [t for t in ear if hi < t <= hi + 60])
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


def join_kos(kos, extra, gap=200):
    """The 腰壁 stretches taken from furniture_runs join a 腰壁 already found on the same line when
    they meet it (プラン5 1F: the counter's edge runs on into the 腰壁 of the kitchen); the rest are
    added as they are."""
    kos = [dict(k) for k in kos]
    for e in extra:
        for k in kos:
            if k["horizontal"] == e["horizontal"] and abs(k["line_mm"] - e["line_mm"]) < 1 and                     e["a_mm"] - gap <= k["b_mm"] and e["b_mm"] + gap >= k["a_mm"]:
                k["a_mm"], k["b_mm"] = min(k["a_mm"], e["a_mm"]), max(k["b_mm"], e["b_mm"])
                break
        else:
            kos.append(e)
    return kos


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
                # (笠木 end, centre-line end, at a wall): both stop at the board face, not inside
                # the 石膏ボード (user 11/12, プラン3: the 腰壁 went through the board)
                # (the 芯 line still runs to the wall centre, as the user's own drawing has it)
                ends.append((j - sgn * board, j, False))  # closed by a plain cross line at the board face
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


def entrance_casing(offsets, placements, cfg, lib=None):
    """額縁 at the entrance door's jambs on the room side (user's correction プラン3 1F, 10/27;
    provisional sizes, details to come): the wall stops 21 short of the frame and is closed
    there (structure line from the added insulation's face to the column face); the casing, 21
    thick, runs from the frame into the room 13 proud of the board face; the outer lines of the
    wall stop with it. Returns (offsets, extra lines)."""
    off = cfg["offsets_mm"]
    lay = cfg["layers"]["wall"]
    ec = cfg["placement"]["entrance_casing"]
    t_, proud = ec["thickness_mm"], ec["proud_mm"]
    col, board = off["column_face"], off["board_face_inside"]
    ins = off["exterior_side"]["added_insulation"]
    # 額縁の枠側の線は、付加断熱の外面まで（そこに玄関ドアの枠の耳の裏が付く。正解図、user 11/15。
    # 前は壁芯から95の決め打ちで、ドアの位置が変わると額縁が枠に重なった）
    reach = ins
    extras = []
    for pl in placements:
        if pl["word"] not in ENTRANCE_WORDS:
            continue
        h, L, si = pl["horizontal"], pl["line"], -pl["outward"]  # si: the room side
        lo, hi = pl["span"]

        def seg(a, b, h=h):  # (along, across) -> model
            return ((a[0], a[1]), (b[0], b[1])) if h else ((a[1], a[0]), (b[1], b[0]))
        part_segs = []
        if lib and pl["block"] in lib:
            part_segs = block_lines_mm([e_ for e_ in lib[pl["block"]][1] if e_["type"] == "LINE"], pl, cfg["layers"]["scale"])

        def first_hit(e, inward, c):
            """枠の端 e から内側（inward）へ、壁の長手の線（壁を横切る位置 c）を進めて最初に当たる部品の線"""
            best = None
            for (p1, p2) in part_segs:
                (a1, c1), (a2, c2) = (p1, p2) if h else ((p1[1], p1[0]), (p2[1], p2[0]))
                if abs(c2 - c1) < 1e-6 or not (min(c1, c2) - 0.01 <= c <= max(c1, c2) + 0.01):
                    continue
                a = a1 + (a2 - a1) * (c - c1) / (c2 - c1)
                dd = (a - e) * inward
                if -0.5 <= dd <= 150 and (best is None or dd < best):
                    best = dd
            return None if best is None else e + inward * best
        def frame_edge(e, inward):
            """額縁が突き当たる枠の位置：枠の端 e から内側へ、額縁の範囲（室内の額縁の面〜付加断熱の外面の手前）で
            最初に当たる部品の線（正解図：枠の端から72.5の枠の立ち上がり。user 11/15）"""
            best = None
            for (p1, p2) in part_segs:
                for k_ in range(41):
                    s_ = k_ / 40
                    a_ = (p1[0] + (p2[0] - p1[0]) * s_) if h else (p1[1] + (p2[1] - p1[1]) * s_)
                    c_ = (p1[1] + (p2[1] - p1[1]) * s_) if h else (p1[0] + (p2[0] - p1[0]) * s_)
                    if not (min(L - si * (ins - 1), L + si * (board + proud)) <= c_ <= max(L - si * (ins - 1), L + si * (board + proud))):
                        continue
                    dd = (a_ - e) * inward
                    if dd >= -0.5 and (best is None or dd < best):
                        best = dd
            return None if best is None else e + inward * best
        for e, d in ((lo, -1), (hi, 1)):
            ef = frame_edge(e, -d)  # 額縁の枠側の面
            if ef is None or abs(ef - e) > 200:
                ef = e - d * t_  # （部品の線がない時は今までどおり：枠の端から21内側）
            x = ef
            ew = ef + d * t_  # 額縁の壁側の面：壁の線はここで止めて閉じる
            # the lines that stop inside the casing stop at its wall side (ew)
            new = []
            for (p, q), color, label in offsets:
                (pa, pc_), (qa, qc) = ((p[0], p[1]), (q[0], q[1])) if h else ((p[1], p[0]), (q[1], q[0]))
                if abs(pc_ - qc) < EPS and abs(pc_ - L) <= 200:
                    # 付加断熱の外面より外の線（付加断熱の面・通気層・外壁の面）は、玄関ドアの枠の外側の出っ張りに
                    # 当たるまでのばす（枠の端で止めると、枠の手前で途切れた。正解図、user 11/15）。
                    # それより内の線（面材・柱の面・ボード）は、額縁の壁側の面で止める
                    if (pc_ - L) * si < 0 and abs(pc_ - L) >= ins - 0.5:
                        end_ = first_hit(e, -d, pc_)
                        end_ = e if end_ is None else end_
                    else:
                        end_ = ew
                    # 端が枠の端から額縁の少し先（枠の線のところ）までにある線（構造用面材の線は、前の工程で枠の線の
                    # 位置まで残っていて、額縁を貫通していた。user 11/15）
                    lo_w, hi_w = min(e, x) - 2, max(e, x) + 2
                    if (pc_ - L) * si >= 0 or abs(pc_ - L) < ins - 0.5:
                        lo_w, hi_w = min(e, x - d * 60) - 2, max(e, x - d * 60) + 2
                    if lo_w < pa < hi_w:
                        pa = end_
                    if lo_w < qa < hi_w:
                        qa = end_
                    (p, q) = seg((pa, pc_), (qa, qc))
                new.append(((p, q), color, label))
            offsets = new
            st = lay["style"]
            extras.append((seg((ew, L - si * ins), (ew, L + si * col)), lay["layer"], lay["structure_color"], st))
            extras.append((seg((ew, L + si * col), (ew, L + si * (board + proud))), lay["layer"], lay["finish_color"], st))
            extras.append((seg((ew, L + si * (board + proud)), (x, L + si * (board + proud))), lay["layer"], lay["finish_color"], st))
            extras.append((seg((x, L + si * (board + proud)), (x, L - si * reach)), lay["layer"], lay["finish_color"], st))
        print(f"玄関ドアの額縁（厚{t_}・暫定）: {pl['block']}")
    return offsets, extras


def window_extras(placements, cfg, spaces=(), lib=None):
    """Wall returns and 窓台 at each window: [((p, q), layer, color, style)] in model mm.
    Returns run from the room-side face to the window frame's interior face: the column-face /
    下地 line (52.5, color 1) at the frame edge, the board face (65, color 5) 12.5 inside it.
    窓台: 12.5 into the walls each side (= frame width), 10 in front of the board face.
    戻り線は、その位置で最初に当たる窓の部品の線まで伸ばす（部品の一番奥の線で止めると、縦すべり窓
    の端で枠まで15届かなかった。user 11/14 プラン3 2F）。当たらなければ一番奥の線で止める。"""
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

        segs_ = []
        if lib and pl["block"] in lib:
            segs_ = block_lines_mm([e for e in lib[pl["block"]][1] if e["type"] == "LINE"], pl, cfg["layers"]["scale"])

        def reach(t, start, inward):
            """壁の中心線に直角、along=t の線を start から外へ進めて、最初に当たる部品の線"""
            tt = t + inward * 0.5  # 枠の端ちょうどは避けて、少し窓の内側で見る
            best = None
            for (p1, p2) in segs_:
                (a1, c1), (a2, c2) = (p1, p2) if pl["horizontal"] else ((p1[1], p1[0]), (p2[1], p2[0]))
                if abs(a2 - a1) < 1e-6 or not (min(a1, a2) <= tt <= max(a1, a2)):
                    continue  # (along の向きの線と、届かない線は見ない)
                c = c1 + (c2 - c1) * (tt - a1) / (a2 - a1)
                d = (c - start) * s
                if d > 0.5 and (best is None or d < best):
                    best = d
            if best is None or abs(start + s * best - face) > 30:
                return face  # (当たらない・離れすぎ＝部品の別の所：今までどおり)
            return start + s * best

        st = lay["style"]
        for t, inw in ((lo, 1), (hi, -1)):
            out.append((across(t, L - s * col, reach(t, L - s * col, inw)), lay["layer"], lay["structure_color"], st))
        for t, inw in ((lo + embed, 1), (hi - embed, -1)):
            if not cfg.get("draw_board", True):
                break  # 石膏ボードなしの仕様：窓の戻りの石膏ボードの線も描かない
            out.append((across(t, L - s * board, reach(t, L - s * board, inw)), lay["layer"], lay["finish_color"], st))
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
    import spec as _spec
    cfg = _spec.load_cfg(cfg_path)
    grid = cfg["input"]["half_grid_mm"]  # walls may sit on 227.5 lines (closets)
    walls, det = load_walls(walls_path, grid)
    bare = bare_lines(det, img_path, walls_path.replace("_walls.json", "_openings.json"), cfg)
    if bare:
        walls, det = load_walls(walls_path, grid, bare)
    furniture_ops = drop_furniture_lines(walls, walls_path.replace("_walls.json", "_openings.json"), cfg)
    furniture_ops |= stair_foot_openings(walls, walls_path, cfg)
    extra_kos = furniture_runs(walls, det, img_path, walls_path.replace("_walls.json", ""), cfg)
    fill_wall_gaps(walls, walls_path.replace("_walls.json", "_openings.json"), cfg)
    SPUR_MAX[0] = cfg.get("exterior_close", {}).get("spur_max_mm", 0.0)
    walls.extend(close_exterior(walls, cfg))  # (the outline must close: the closing wall is part of it)
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
    ubs, ub_cuts = [], []
    ops_path = walls_path.replace("_walls.json", "_openings.json")
    root = Path(__file__).resolve().parent.parent
    libs_ = cfg["placement"]["library"]
    lib_paths = sorted({p for g in ([libs_] if isinstance(libs_, str) else libs_)
                        for p in root.glob(g.replace("\\", "/"))})  # 文字列でも、リストでもよい
    if Path(ops_path).exists() and lib_paths:
        import place_fittings as pf
        from opening_detect import Mapper
        lib = pf.load_library([str(p) for p in lib_paths])
        # a 引違い戸's frame must not be thinner than the wall (board face to board face, 130):
        # the frame faces move out to the board faces (user 11/12, プラン4 1F 洋室の入口 W1644, 115 deep)
        bf = cfg["placement"].get("door_frame_half_mm", 77.5)  # 建具枠は155（壁130＋12.5×2、user 11/12）
        S_ = cfg["layers"]["scale"]
        for key, v in lib.items():
            if v[3] != "引違い戸":
                continue
            ents = [e for e in v[1] if e["type"] == "LINE"]
            ys = [y * S_ for e in ents for y in (e["start_y"], e["end_y"])]
            half = max(abs(y) for y in ys)
            if half < bf - 0.5:
                d_ = (bf - half) / S_
                for e in ents:
                    for k in ("start_y", "end_y"):
                        if abs(abs(e[k] * S_) - half) < 0.5:
                            e[k] += d_ if e[k] > 0 else -d_
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
        shift_entrance_doors(placements, walls, cfg, find_entrance_runs(det, img_path, walls_path.replace("_walls.json", ""), cfg))
        # バルコニーの手すり壁の開口は窓ではない（窓の室内側がバルコニーなら置かない。開口は人が描く。
        # プラン2 2F x=-120 と y=0 の3つ、user 11/15）
        balc = [poly for poly, k in spaces if "バルコニー" in k]
        if balc:
            kept_ = []
            for pl in placements:
                if pl["word"] in pf_WINDOWS:
                    mid_ = (pl["span"][0] + pl["span"][1]) / 2
                    L_, s_ = pl["line"], pl["outward"]
                    probe_ = Point(mid_, L_ - s_ * 300) if pl["horizontal"] else Point(L_ - s_ * 300, mid_)
                    if any(b.contains(probe_) for b in balc):
                        print(f"バルコニーの手すり壁の開口なので窓を置かない: {pl['block']} 線{L_:g}")
                        continue
                kept_.append(pl)
            placements = kept_
        for pl in placements:
            if pl["word"] in ENTRANCE_WORDS:
                print(f"玄関ドア: {'横' if pl['horizontal'] else '縦'} 線{pl['line']:g} 枠{pl['span'][0]:.1f}〜{pl['span'][1]:.1f} 幅{pl['width']:g} 吊り元{'小' if pl['sx'] == 1 else '大'}側")
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

        def shift(q, mv):
            x, y = q["pos"]
            q["pos"] = (x + mv, y) if q["horizontal"] else (x, y + mv)
            q["span"] = (q["span"][0] + mv, q["span"][1] + mv)

        def trim_margins(q):
            """枠の端（span）から部品の一番外の線までの出（lo側, hi側）"""
            if not (lib and q["block"] in lib):
                return 0.0, 0.0
            ents_ = [e for e in lib[q["block"]][1] if e["type"] == "LINE"]
            ts_ = [t for (a_, b_) in block_lines_mm(ents_, q, cfg["layers"]["scale"])
                   for t in ((a_[0], b_[0]) if q["horizontal"] else (a_[1], b_[1]))]
            if not ts_:
                return 0.0, 0.0
            return max(0.0, q["span"][0] - min(ts_)), max(0.0, max(ts_) - q["span"][1])
        # 並んだ窓（連窓：枠どうしが100未満）はひとまとめにして動かす（プラン5 2F の W300×2 が重なった）
        groups = []
        for pl in sorted((p for p in placements if p["word"] in pf_WINDOWS),
                         key=lambda p: (p["horizontal"], round(p["line"], 1), p["span"][0])):
            g = groups[-1] if groups else None
            if g and g[0]["horizontal"] == pl["horizontal"] and abs(g[0]["line"] - pl["line"]) < 1 \
                    and pl["span"][0] - max(q["span"][1] for q in g) < 100:
                g.append(pl)
            else:
                groups.append([pl])
        # 連窓の窓と窓の間（枠と枠）は105（柱1本分）。全体の中心は動かさない（user 11/14）
        mull = cfg["column"]["size_mm"]
        for grp in groups:
            if len(grp) < 2:
                continue
            c0 = (grp[0]["span"][0] + grp[-1]["span"][1]) / 2
            total = sum(q["span"][1] - q["span"][0] for q in grp) + mull * (len(grp) - 1)
            t = c0 - total / 2
            for q in grp:
                mv = t - q["span"][0]
                if abs(mv) > 0.05:
                    shift(q, mv)
                    print(f"連窓の窓と窓の間を{mull:g}に（{mv:+.1f}）: {q['block']}")
                t = q["span"][1] + mull
        for grp in groups:
            pl = grp[0]
            h, L, s = pl["horizontal"], pl["line"], pl["outward"]
            lo, hi = grp[0]["span"][0], grp[-1]["span"][1]
            move = 0.0
            # the gap is kept from the part's outermost line (the frame's trim stands 25 past the
            # span), to the outermost line of the other exterior wall (user 11/12, プラン4 1F)
            m_lo, m_hi = trim_margins(grp[0])[0], trim_margins(grp[-1])[1]
            # (inside a garage the last finish is the 12.5 board, 152 from the centre, not the siding)
            mid = (lo + hi) / 2
            probe = Point(mid, L + s * 300) if h else Point(L + s * 300, mid)
            face_d = ext_side["exterior_face"]
            if any(k in cfg["outside_space"]["garage_names"] and p.contains(probe) for p, k in spaces):
                face_d = ext_side["ventilation"] + cfg["outside_space"]["garage_board_mm"]
            for w in walls:
                if not w.ext or w.h == h or not (w.a - 1 <= L + s * 200 <= w.b + 1):
                    continue  # (a perpendicular exterior wall running on outside the window's wall)
                if lo - 400 < w.c <= lo and lo < w.c + face_d + gap + m_lo:
                    move = max(move, w.c + face_d + gap + m_lo - lo)
                if hi <= w.c < hi + 400 and hi > w.c - face_d - gap - m_hi:
                    move = min(move, w.c - face_d - gap - m_hi - hi)
            if move:
                for q in grp:
                    shift(q, move)
                    print(f"入隅に近い窓を外壁面から{gap}離す（{move:+.1f}）: {q['block']}")
        # 入隅の外壁と取り合う窓で、反対の端に間仕切りが直交している（柱が要る）とき：入隅からは
        # なるべく離すが、優先は「入隅と反対の柱に寄せる」。枠の端＝柱の面から12.5（柱芯から65）
        # （user 11/13、プラン3 2F の縦すべり窓W640：間仕切りの柱 (2730,7280) に枠が食い込んでいた）
        # この納めができるのは窓幅640まで（特注を除く。user 11/14）
        hc_ = cfg["column"]["size_mm"] / 2
        edge_ = hc_ + 12.5
        for grp in groups:
            pl = grp[0]
            h, L, s = pl["horizontal"], pl["line"], pl["outward"]
            lo, hi = grp[0]["span"][0], grp[-1]["span"][1]
            if hi - lo > 640 + 1:
                continue
            corner_lo = any(w.ext and w.h != h and w.a - 1 <= L + s * 200 <= w.b + 1 and lo - 600 < w.c <= lo for w in walls)
            corner_hi = any(w.ext and w.h != h and w.a - 1 <= L + s * 200 <= w.b + 1 and hi <= w.c < hi + 600 for w in walls)
            if corner_lo == corner_hi:
                continue
            # 入隅と反対の端のそばで、室内側から直交してくる間仕切り（外壁には付く）
            inner = [w.c for w in walls if not w.ext and w.h != h and w.a - 1 <= L <= w.b + 1
                     and w.a - 1 <= L - s * 200 <= w.b + 1
                     and ((hi - 455 < w.c < hi + 455) if corner_lo else (lo - 455 < w.c < lo + 455))]
            if not inner:
                continue
            t = min(inner, key=lambda c: abs(c - (hi if corner_lo else lo)))
            move = (t - edge_ - hi) if corner_lo else (t + edge_ - lo)
            if abs(move) < 0.5:
                continue
            for q in grp:
                x, y = q["pos"]
                q["pos"] = (x + move, y) if h else (x, y + move)
                q["span"] = (q["span"][0] + move, q["span"][1] + move)
                print(f"入隅の窓を反対の間仕切りの柱に寄せる（柱面から12.5、{move:+.1f}）: {q['block']}")
        # 室内の開き戸は910グリッドの柱間の内側に置く：枠が、間仕切りの取り合う910の線（柱の位置）に
        # かかっていたら、柱面から12.5の所まで柱間の内側へずらす（user 11/13、プラン5 1F 片開きW650）
        g910 = 2 * cfg["input"]["grid_mm"]
        for pl in placements:
            if pl["word"] not in ("片開き", "両開き", "折戸"):
                continue
            h, L = pl["horizontal"], pl["line"]
            lo, hi = pl["span"]
            if hi - lo + 2 * edge_ > g910 + 4:
                continue
            k = math.floor(((lo + hi) / 2) / g910)
            g0, g1 = k * g910, (k + 1) * g910

            def meets(g):
                return any(w.h != h and abs(w.c - g) < 1 and w.a - 1 <= L <= w.b + 1 for w in walls)
            move = 0.0
            if lo < g0 + edge_ - 0.5 and meets(g0):
                move = g0 + edge_ - lo
            elif hi > g1 - edge_ + 0.5 and meets(g1):
                move = g1 - edge_ - hi
            if move:
                x, y = pl["pos"]
                pl["pos"] = (x + move, y) if h else (x, y + move)
                pl["span"] = (lo + move, hi + move)
                print(f"開き戸を910の柱間の内側へ（{move:+.1f}）: {pl['block']}")
        if os.environ.get("WIN_DEBUG"):
            for pl in placements:
                if pl["word"] not in pf_WINDOWS or pl["block"] not in lib:
                    continue
                h_ = pl["horizontal"]
                ents_ = [e for e in lib[pl["block"]][1] if e["type"] == "LINE"]
                ts = [t for (a_, b_) in block_lines_mm(ents_, pl, cfg["layers"]["scale"]) for t in ((a_[0], b_[0]) if h_ else (a_[1], b_[1]))]
                print("WINDBG", pl["block"], "span", [round(v, 1) for v in pl["span"]], "extent", round(min(ts), 1), round(max(ts), 1), "line", pl["line"])
        offsets = cut_openings(offsets, placements, cfg, lib=lib)
        # UB のドアの所は壁を切る（UB の部品のドアの枠の幅、user 11/16）
        ubs, ub_cuts = ub_layout(walls_path.replace("_walls.json", "_UB.json"), cfg)
        if ub_cuts:
            offsets = cut_openings(offsets, ub_cuts, cfg)
        offsets, casing = entrance_casing(offsets, placements, cfg, lib=lib)
        extras = window_extras(placements, cfg, spaces, lib=lib) + casing
        names_path = walls_path.replace("_walls.json", "_部屋名.json")
        if Path(names_path).exists():
            names = json.load(open(names_path, encoding="utf-8"))
            M = Mapper(det, cfg["input"]["grid_mm"])
            open_k = open_kamachi(img_path, walls, names, M, cfg)
            extras += kamachi_lines(list(ops) + open_k, names, M, cfg, walls)
    kos_path = walls_path.replace("_walls.json", "_腰壁.json")
    kos = []
    if Path(kos_path).exists() or extra_kos:
        voids = []
        npath = walls_path.replace("_walls.json", "_部屋名.json")
        # (1階の「上部吹抜」は上の階のこと。2階以上は自分の階の吹抜。3階建てでは2階にも上の階がある：11/15)
        if not Path(walls_path).name.startswith("1F") and Path(npath).exists():
            from opening_detect import Mapper
            Mv = Mapper(det, cfg["input"]["grid_mm"])
            voids = [(Mv.xmm(n["x"]), Mv.ymm(n["y"])) for n in json.load(open(npath, encoding="utf-8"))
                     if n.get("name") == "吹抜"]
        k_lines, kos = koshikabe_lines(join_kos(json.load(open(kos_path, encoding="utf-8")) if Path(kos_path).exists() else [], extra_kos), walls, fp, cfg, voids)
        extras += k_lines
    # 通り芯 added by hand-made data: 上がり框 (a stair's last step) has a beam, so a 芯 line is drawn
    # across the stair, which also divides it (user 11/12, プラン3 2F; file <階>_通り芯.json,
    # [{"horizontal": false, "line_mm": 6370, "a_mm": 6370, "b_mm": 7280}])
    tr_path = walls_path.replace("_walls.json", "_通り芯.json")
    if Path(tr_path).exists():
        cl = cfg["layers"]["center"]
        for t in json.load(open(tr_path, encoding="utf-8")):
            c, a, b = t["line_mm"], t["a_mm"], t["b_mm"]
            extras.append((((a, c), (b, c)) if t["horizontal"] else ((c, a), (c, b)),
                           cl["layer"], cl["color"], cl["style"]))
            print(f"通り芯（{t.get('label', '上がり框')}）: {'横' if t['horizontal'] else '縦'} {c} {a}〜{b}")
    fits =[(pl["horizontal"], pl["line"], *pl["span"], pl["word"]) for pl in placements] + shutters
    fits += [(c["horizontal"], c["line"], *c["span"], c["word"]) for c in ub_cuts]
    # 片引戸の開口側の半分（中方立まで＋150）：直交する壁の柱はここに置かない（中方立の付近に柱は置けない。
    # 戸袋側の半分には置ける：プラン1 1F の正解図 (3640,7735)。user 11/13）
    mullions = []
    for pl in placements:
        if pl["word"] != "片引戸" or not lib or pl["block"] not in lib:
            continue
        off_, W_ = lib[pl["block"]][5], lib[pl["block"]][4]
        base = pl["pos"][0] if pl["horizontal"] else pl["pos"][1]
        e1, e2 = base + pl["sx"] * off_, base + pl["sx"] * (off_ + W_ / 2 + 150)
        mullions.append((pl["horizontal"], pl["line"], min(e1, e2), max(e1, e2)))
    cols, defaults = place_columns(walls, nodes, cfg, upper, fits, fp, mullions)
    extras += stud_lines(walls, cols, fits, cfg, placements)
    if placements:
        offsets = jamb_closures(offsets, placements, cols, walls, cfg)
        offsets = furring_offsets(offsets, placements, cols, cfg)
    extras += furniture_symbols(det, img_path, walls_path.replace("_walls.json", ""), cfg, walls)
    extras += entrance_storage_symbols(det, img_path, walls_path.replace("_walls.json", ""), cfg, walls)
    extras = dot_under_boxes(extras, cfg)
    toilet_path = walls_path.replace("_walls.json", "_便器.json")
    toilets = json.load(open(toilet_path, encoding="utf-8")) if Path(toilet_path).exists() else []
    # キッチン：読み取った位置・向きに部品を置く（家具・設備のレイヤー。UB と同じ描き方、user 11/16）
    kitchens = kitchen_layout(walls_path.replace("_walls.json", "_キッチン.json"), cfg, walls)
    write_jww(prefix + ".jww", walls, offsets, cols, cfg, center, placements, lib, extras, toilets, list(ubs) + kitchens)
    # model mm = paper mm * scale + center; score_walls.py needs this to read corrected drawings back
    json.dump({"center_mm": center, "scale": cfg["layers"]["scale"],
               "placements": [{**{k: pl[k] for k in ("block", "word", "kind", "pos", "rot", "sx", "sy", "width", "horizontal", "line")}, "label": pl.get("label")}
                              for pl in placements],  # 建具の置き方（部品の差し替え用）
               "ub": [{k: u[k] for k in ("block", "kind", "pos", "rot", "sx", "sy")} for u in ubs],
               "ub_openings": [{k: c[k] for k in ("horizontal", "line", "span")} for c in ub_cuts],
               "kitchen": [{k: u[k] for k in ("block", "pos", "rot", "sx", "sy")} for u in kitchens]},
              open(prefix + "_meta.json", "w", encoding="utf-8"))
    # 建具の枠に入り込んだ紫線・壁の線を切る（検査 C・D と同じ箱で。入隅すぐ脇の窓の枠に外壁の面が当たる等、11/13）
    import fix_fittings
    fix_fittings.fix(prefix + ".jww", prefix + "_meta.json")
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
