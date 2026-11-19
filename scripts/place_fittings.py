"""Where and how to place a parts-library block for each detected opening.

Block base point = the CENTRE OF THE LEFT COLUMN (a grid point on the wall centre line), so a
person can copy a part from 建具部品集.jww by clicking a column centre (user, 2026-10-22). The
part itself sits where it is normally built, relative to that column:
  - doors: frame outer edge 65 from the grid (column face + 12.5). A door whose width is the
    clear width minus 2 mm (778 in a 910 bay: 780 hallway) is centred, 1 mm each side.
  - entrance door: frame 65 from the grid; its maker reference point (×印) on the outer face of
    the added insulation.
  - windows: centred in the bay (the smallest multiple of 455 that leaves room for the 105
    columns); maker reference point on the outer face of the added insulation.
In a drawing, windows are centred in the actual bay between the jamb grid lines, doors start from
the hinge-side grid line. The nearest library size is used for now (later: the instructed width).

Blocks are drawn horizontally: doors and the entrance door open toward local +y and are hinged at
the left; 縦すべり windows pivot at the right; windows and the entrance door have the exterior
toward +y; a sliding door's pocket is on the right.
"""
import math

import ezjww

GRID = 455.0
COLUMN = 105.0
DOOR_EDGE = 65.0  # frame outer edge from the grid line (column face 52.5 + 12.5)

# which library parts each detected kind may use (empty: no part yet -> marked, not drawn)
PART_FOR_KIND = {
    "swing": ("片開き",),
    # UB: no door; the UB wall is drawn closed and a person adds the UB with its door (the tub
    # faces two ways and the door moves with it, user 10/25)
    "ub_door": (),
    "sliding_pocket": ("片引戸",),
    "sliding": ("引違い戸",),  # 2F closet (user: 引違い, not 折戸)
    "storage": ("両開き",),
    "folding": ("折戸",),  # V-shaped leaves (interior_type.py)
    "window": ("引違い窓",),  # used when the window code could not be read
    "entrance_door": ("玄関片開き", "玄関親子"),
}
# window code letters -> parts (TF = 縦すべり + FIX side by side, the 縦すべり at its hinge side)
# TF: the maker's 連窓 part (LIXIL LSMKE023, user 10/24: do not build it from two parts); the
# two-part split is only the fallback when no 連窓 part fits
PART_FOR_LETTERS = {"H": ("引違い窓",), "T": ("縦すべり窓",), "F": ("FIX窓",), "TF": ("縦すべり窓", "FIX窓"),
                    "Y": ("横すべり窓",), "TY": ("横すべり窓", "FIX窓")}
WORDS = ("片開き", "片引戸", "引違い戸", "両開き", "折戸", "UBドア", "引違い窓", "縦すべり窓", "FIX窓", "横すべり窓",
         "縦すべり+FIX窓", "玄関片開き", "玄関親子", "アウトセット片引戸")
WINDOWS = {"引違い窓", "縦すべり窓", "FIX窓", "横すべり窓", "縦すべり+FIX窓"}
OUTWARD = WINDOWS | {"玄関片開き", "玄関親子"}  # exterior toward local +y, reference point outside
HINGE_RIGHT = {"縦すべり窓", "UBドア"}  # pivot drawn at the right end (LIXIL 縦すべり; the user's UB door)
# 縦すべり+FIX窓 の部品は、吊り元（弧の中心）が左端（x≈125–150）にある。右端扱いにしていて開く側が逆だった
# （プラン1 2F 洋室A、user 11/18 の 94）
NO_HINGE = {"引違い窓", "FIX窓", "横すべり窓", "片引戸", "引違い戸", "アウトセット片引戸"}
# アウトセット片引戸：壁を削らずに、壁の表面に後から付ける引戸。戸袋の側の壁が薄くならない（user 11/14）。
# 開口（910の柱間なら W1320、それより広ければ W1580）の両脇に柱が入る
OUTSET = "アウトセット片引戸"


def outset_jambs(entities, S):
    """アウトセット片引戸の開口の範囲（基準点から、局所 mm）：+y 側に立つ縦の枠（長さ100以上）の外から外"""
    xs = [e["start_x"] * S for e in entities if e["type"] == "LINE" and abs(e["start_x"] - e["end_x"]) < 1e-6
          and abs(e["end_y"] - e["start_y"]) * S > 100]
    return (min(xs), max(xs)) if xs else None


SLIDING_WINDOW_MIN_MM = 1500  # user (2026-10-22): 引違い窓 is used only at W1500 and wider


def std_bay(word, w):
    """The bay (column centre to column centre) a part normally sits in."""
    need = w + COLUMN if word in WINDOWS else w + 2 * DOOR_EDGE
    k = 1
    while k * GRID < need - 4:  # 778 + 130 = 908 still fits 910
        k += 1
    return k * GRID


def std_offset(word, w):
    """Distance from the left column centre to the frame's outer edge in the library."""
    bay = std_bay(word, w)
    if word in WINDOWS:
        return (bay - w) / 2
    if word not in ("玄関片開き", "玄関親子") and abs(bay - (w + 2 * DOOR_EDGE)) <= 4:
        return (bay - w) / 2  # clear width - 2: split the 2 mm, 1 mm each side
    return DOOR_EDGE


def load_library(paths):
    """{block name: (name, entities, W paper mm, kind word, W mm, offset mm, interior depth mm)}
    from every category file; the column-centre-to-frame offset comes from 建具部品集_一覧.csv
    beside them (parts cut from user drawings have their own), else from std_offset."""
    import csv
    from pathlib import Path
    lib = {}
    for path in paths:
        offsets = {}
        index = Path(path).parent / "建具部品集_一覧.csv"
        if index.exists():
            offsets = {r["ブロック名"]: float(r["柱芯から枠外まで"])
                       for r in csv.DictReader(open(index, encoding="utf-8-sig"))}
        lib.update(_load_one(path, offsets))
    return lib


def _load_one(path, offsets):
    doc = ezjww.read_document(path)
    S = doc["header"]["layer_groups"][0]["scale"]
    lib = {}
    for b in doc["block_defs"]:
        name = b["name"]
        w = int(name.rsplit("_W", 1)[1])
        # the frame's interior face: the most interior long horizontal line (local -y), in mm
        ys = [e["start_y"] for e in b["entities"] if e["type"] == "LINE"
              and abs(e["start_y"] - e["end_y"]) < 1e-6 and abs(e["end_x"] - e["start_x"]) * S > w / 2]
        inner = min(ys) * S if ys else 0.0
        if any(f"_{w_}_" in name for w_ in WINDOWS):
            # 窓：枠の室内側の面＝部品の線の一番室内側（YKKの枠は左右の脚が奥まで入るので、長い横線だけでは
            # 足りず、壁の戻り線が窓にはみ出した、user 11/13）
            ally = [y * S for e in b["entities"] if e["type"] == "LINE" for y in (e["start_y"], e["end_y"])]
            inner = min(ally) if ally else inner
        if "_2枚連動片引戸_" in name:
            # 2枚連動片引戸は、戸袋に2枚が引き込む片引戸。片引戸と同じに扱い、幅（開口＋戸袋）の近い方を選ぶ
            # （プラン6 1F ゲストルーム：開口1645＋戸袋で約2500。引違い戸ではない。user 11/17）
            lib[name] = (name, b["entities"], w / S, "片引戸", w, offsets.get(name, std_offset("片引戸", w)), inner)
            continue
        for word in WORDS:
            if f"_{word}_" in name:
                lib[name] = (name, b["entities"], w / S, word, w, offsets.get(name, std_offset(word, w)), inner)
    return lib


def choose(lib, words, width_mm, bay=None):
    """The part of those kinds whose catalogue width is closest to the wanted width, among the
    parts that fit the bay (column centre to column centre): a part never runs past the jambs
    (プラン2 2F: a 1644 sliding door in a 910 WIC opening stuck out through the wall)."""
    cands = [v for v in lib.values() if v[3] in words]
    if bay is not None:
        # the frame plus its clearances must fit the actual bay (which may be on 227.5)
        need = lambda v: v[4] + (COLUMN if v[3] in WINDOWS else 2 * DOOR_EDGE)  # noqa: E731
        cands = [v for v in cands if need(v) <= bay + 4]
    return min(cands, key=lambda v: abs(v[4] - width_mm)) if cands else None


# when no part of the detected kind fits the opening, similar kinds are tried in this order
# (never a sliding door replaced by a swing door: プラン5 1F 洋室A's 片引戸 became a 片開き, user 10/24)
# (no 引違い戸 → 折戸 / 両開き either: a swing door misread as sliding got a 折戸, プラン2 2F; no
# part is better than a wrong one, user 10/24)
FALLBACK = {"折戸": ("両開き", "片開き"),
            "両開き": ("折戸", "片開き"), "引違い窓": ("縦すべり窓", "FIX窓")}


def snap(v, step):
    return round(v / step) * step


def parts_for(o, lib, bay=None, small_door=None):
    """Parts for one opening: one part, or two for a TF window. Only parts that fit the bay."""
    lab = o.get("label")
    if o["kind"] == "window" and not lab and o.get("window_type") in ("縦すべり+FIX", "横すべり+FIX"):
        # no printed code, but the drawing shows a fan / dashed box over part of the window:
        # a TF / TY 連窓
        lab = {"letters": "TF" if o["window_type"] == "縦すべり+FIX" else "TY", "width_mm": o["width_mm"]}
    if o["kind"] == "window" and lab and lab["letters"] in PART_FOR_LETTERS:
        words = PART_FOR_LETTERS[lab["letters"]]
        if lab["letters"] == "TF":
            p = choose(lib, ("縦すべり+FIX窓",), lab["width_mm"], bay)
            if p:
                return [p]
        if len(words) == 2:  # 連窓: split the coded width in two
            first = choose(lib, words[:1], lab["width_mm"] / 2, bay / 2 if bay else None)
            rest = choose(lib, words[1:], lab["width_mm"] - first[4], bay / 2 if bay else None) if first else None
            return [p for p in (first, rest) if p]
        p = choose(lib, words, lab["width_mm"], bay)
        if lab.get("code"):
            # a printed size wins over the bay found in the drawing (例1 2F: 06011T is W640, but
            # only W500 fitted the 682.5 bay between the jamb grid lines; user 10/30)
            q = choose(lib, words, lab["width_mm"])
            if q and (p is None or abs(q[4] - lab["width_mm"]) < abs(p[4] - lab["width_mm"])):
                p = q
        return [p] if p else []
    words = PART_FOR_KIND.get(o["kind"], ())
    if o["kind"] == "sliding_pocket" and o.get("outset"):
        # 開口の幅で選ぶ（戸袋の長さは部品が持っている）
        # 枠の外から外を、開口の柱間の石膏ボード面から石膏ボード面（柱間−65×2）に入る一番大きいもの
        # （910の柱間なら W1580：枠の外 780 ≦ 780。user 11/14）
        gp = o.get("gap_pocket_px") or [1, 1]
        gap_mm = o["width_mm"] * gp[0] / max(1, gp[0] + gp[1])
        obay = math.ceil((gap_mm + COLUMN) / GRID - 0.05) * GRID
        fit = []
        for v in lib.values():
            if v[3] != OUTSET:
                continue
            j = outset_jambs(v[1], v[4] / v[2])
            if j and j[1] - j[0] <= obay - 2 * DOOR_EDGE + 1:
                fit.append(v)
        p = max(fit, key=lambda v: v[4]) if fit else choose(lib, (OUTSET,), 1320)
        return [p] if p else []
    if o["kind"] == "sliding_pocket":
        # a 片引戸's pocket half runs inside the wall beyond the visible opening: the bay does
        # not limit it; it is placed from the jamb the door starts at (プラン2 1F 脱衣所, 10/25)
        bay = None
        # 戸の枚数（2本線の組の数、interior_type.count_leaves）：3枚以上は部品がない（user 11/17）。
        # 2枚は 2枚連動片引戸だけから選ぶ
        if o.get("leaves", 1) >= 3:
            return []
        if o.get("leaves", 1) == 2:
            two = [v for v in lib.values() if "_2枚連動片引戸_" in v[0]]
            if two:
                return [min(two, key=lambda v: abs(v[4] - o["width_mm"]))]
    if o["kind"] == "entrance_door":
        # no columns at the entrance door (user 10/23), so the bay does not limit it: プラン2's
        # ジエスタ2 W924 stands in a 910 bay
        bay = None
    if o["kind"] == "window":
        words = (o.get("window_type"),) if o.get("window_type") else words
        if words == ("引違い窓",) and o["width_mm"] < SLIDING_WINDOW_MIN_MM:
            words = ("縦すべり窓",)  # 引違い is only used from W1500 up
    want = o["width_mm"]
    if words == ("片開き",) and small_door is not None:
        # two door sizes (user 10/23): 778 for rooms, the maker's small one (ラシッサ トイレドア
        # W648) where the opening is clearly narrower (1F toilet). The measured width is too
        # rough to pick by nearest size.
        want = 648 if o["width_mm"] < small_door else 778
        if want == 648:
            bay = None  # the W648 part goes in a ~616 opening as it is (プラン5 1F トイレ, user 10/28)
    p = choose(lib, words, want, bay)
    for w in words:
        if p:
            break
        for alt in FALLBACK.get(w, ()):
            p = choose(lib, (alt,), o["width_mm"], bay)
            if p:
                break
    return [p] if p else []


def plan(openings, lib, cfg, mapper, outside=None, larger_side=None, perp_at=None):
    """Return (placements, skipped). Coordinates in model mm.
    outside(horizontal, line, mid) -> +1/-1: the model direction (+Y or +X) of the exterior.
    larger_side(horizontal, line, mid) -> +1/-1: the side with the larger room. A sliding door
    runs on the smaller room's face (1F toilet: pocket inside the toilet, user 2026-10-22), so the
    block's local +y faces the larger room."""
    half = cfg["input"]["half_grid_mm"]
    grid = cfg["input"]["grid_mm"]
    S = cfg["layers"]["scale"]
    jamb = cfg["placement"]["jamb_search_mm"]
    out, skipped = [], []
    for o in openings:
        if o.get("rejected"):
            continue
        h = o["horizontal"]
        line = snap(o["p1_mm"][1] if h else o["p1_mm"][0], half)
        a, b = sorted((o["p1_mm"][0], o["p2_mm"][0]) if h else (o["p1_mm"][1], o["p2_mm"][1]))
        ga, gb = snap(a - jamb, half), snap(b + jamb, half)  # jamb grid lines (column centres)
        # hinge end (low / high along the wall): doors, the entrance door and 縦すべり windows
        p0, p1, _c = o["px"]
        if o["kind"] == "sliding_pocket":
            start_px = p1 if o["pocket"] == "before" else p0  # the end away from the pocket
        else:
            start_px = o.get("hinge_px")
        hinge_low = None
        if start_px is not None:
            t = mapper.xmm(start_px) if h else mapper.ymm(start_px)
            hinge_low = abs(t - a) <= abs(t - b)
        o_use = o
        # 2000 以上は 2枚連動片引戸（W2430）：1644／1454 に丸めない（プラン6 1F ゲストルーム、user 11/17）
        if o["kind"] == "sliding_pocket" and not o.get("outset") and perp_at is not None and hinge_low is not None \
                and o["width_mm"] < 2000 and o.get("leaves", 1) < 2:
            # the frame at the end away from the pocket is over 780 wide: a wall running across that
            # end leaves no room for a W1644, so the narrower part is used (user 10/28: プラン5 2F, プラン3 1F)
            end = ga if hinge_low else gb
            near_mullion = perp_at(h, line, end, 1 if hinge_low else -1)
            # a wall meeting the wall line at the W1644's 中方立: W1454; none: W1644 (user 10/29). Doors
            # narrower than 1430 in the drawing stay W1454 (プラン5 toilet, 1415)
            if near_mullion and o["width_mm"] >= 1400:
                o_use = dict(o, width_mm=1454)
            elif not near_mullion and o["width_mm"] >= 1430:
                o_use = dict(o, width_mm=1644)
        parts = parts_for(o_use, lib, gb - ga, cfg["placement"].get("door_small_max_mm"))
        # (no widening of the bay to the 455 lines when nothing fits: it moved a sliding door
        # 455 and is dropped, user 10/24. No part = suspect a misread)
        if not parts:
            skipped.append(o)
            continue
        side = o.get("swing_side") or 1  # image side the door swings to (+1/-1)
        s_out = outside(h, line, (a + b) / 2) if outside else 1
        windows = parts[0][3] in WINDOWS
        if windows:
            # centred in the actual bay; a 縦すべり goes to its hinge side within a 連窓
            total = sum(p[4] for p in parts)
            if len(parts) == 2 and hinge_low is False:
                parts = parts[::-1]
            lo = (ga + gb) / 2 - total / 2
            if len(parts) == 1 and total + COLUMN > gb - ga + 4:
                # wider than its bay (a printed size): the end at a 455 jamb keeps the clearance
                # the largest fitting part would have, the window grows toward the 227.5 jamb
                # (例1 2F 06011T ×2 beside the 5460 column, user's fix 10/30)
                fit = choose(lib, (parts[0][3],), total, gb - ga)
                margin = (gb - ga - fit[4]) / 2 if fit else COLUMN / 2
                on455 = lambda g: abs(g / grid - round(g / grid)) < 1e-6  # noqa: E731
                if on455(gb) and not on455(ga):
                    lo = gb - margin - total
                elif on455(ga) and not on455(gb):
                    lo = ga + margin
            frames = []
            for p in parts:
                frames.append((p, lo))
                lo += p[4]
        else:
            p = parts[0]
            if p[3] in ("玄関片開き", "玄関親子") and p[4] + 2 * DOOR_EDGE > gb - ga + 4:
                # wider than its bay (no columns there): where the drawing shows it, centred
                at_low = None
            elif p[3] in ("玄関片開き", "玄関親子") and hinge_low is not None:
                # the entrance frame stands 65 from the latch-side jamb (1F: the inner wall),
                # the hinge at the other end (user's correction 2026-10-22)
                at_low = not hinge_low
            elif p[3] in ("玄関片開き", "玄関親子", "両開き") or (p[3] == "折戸" and hinge_low is None):
                # (a 折戸 under 910 hangs its one V from one jamb: that jamb, like a door's hinge)
                # no hinge preference: 65 from whichever jamb the drawing fits better
                at_low = abs(a - (ga + p[5])) <= abs(b - (gb - p[5]))
            else:
                # the block's base column is its local left end: the hinge end, or for a part
                # pivoting at its right end (縦すべり, UB door) the other end
                at_low = hinge_low is not False
                if p[3] in HINGE_RIGHT and hinge_low is not None:
                    at_low = not hinge_low
                d_low, d_high = abs(a - (ga + p[5])), abs(b - (gb - p[5]))
                if p[3] == "片開き" and (gb - ga) - (p[4] + 2 * DOOR_EDGE) > 20 and abs(d_low - d_high) > 40:
                    # narrower than its bay: the rest of the bay is wall on the side the drawing
                    # shows it (プラン3 1F トイレ W648 against the 7280 jamb; user 10/27). An
                    # opening in the middle of the bay keeps the hinge rule (プラン2 1F トイレ)
                    at_low = d_low < d_high
            if p[3] in ("玄関片開き", "玄関親子") and gb - ga <= cfg["placement"].get("entrance_centre_bay_max_mm", 1369):
                # an entrance door in a 1365 bay between columns stands in the middle of it: with the
                # wall thickness there is almost no room left beside the frame (user 11/12, プラン4 1F)
                frames = [(p, (ga + gb) / 2 - p[4] / 2)]
            elif at_low is None:
                frames = [(p, (a + b) / 2 - p[4] / 2)]
            else:
                frames = [(p, ga + p[5])] if at_low else [(p, gb - p[5] - p[4])]
        for p, flo in frames:
            name, _ents, _wp, word, w, off, inner = p
            fhi = flo + w
            # direction of local +x along the wall: put the block's hinge end on the hinge side
            if word in NO_HINGE or hinge_low is None:
                d = 1 if hinge_low is not False else -1
            elif word in HINGE_RIGHT:
                d = -1 if hinge_low else 1
            else:
                d = 1 if hinge_low else -1
            along = flo - off if d == 1 else fhi + off  # the block base (left column centre)
            if word in OUTWARD:
                sy = s_out if h else -s_out  # rot 90: local +y -> model -X
            elif word in ("片引戸", OUTSET, "折戸", "両開き") and larger_side:
                # 折戸 / 両開き of a storage open out of it, toward the room: their handles are
                # only on the face they open to (user 10/24, プラン5 階段下物入 opened inward)
                s_big = larger_side(h, line, (a + b) / 2)
                if word in ("片引戸", OUTSET) and o.get("pocket_face") is not None:
                    # a 片引戸 slides on the face its leaf is drawn on in the input (user 10/29, プラン3):
                    # the block's local -y is that face, so +y faces away from it
                    face = -o["pocket_face"] if h else o["pocket_face"]  # model direction of the face
                    s_big = -face
                sy = s_big if h else -s_big
            else:
                sy = -1 if side == 1 else 1  # image +y is model -Y; vertical: local +y -> model -X
            across = line
            if word in ("折戸", "両開き"):
                # a storage door's frame stands proud of the board face on the side it opens to
                # (user 10/30: 折戸 frame 12 out of the wall face; 例1 1F 修正完了 3 places, 27.5 shift)
                depth = -inner  # frame half depth (its local -y face)
                shift = cfg["offsets_mm"]["board_face_inside"] + cfg["placement"]["storage_frame_proud_mm"] - depth
                if not name.startswith("LIXIL"):
                    # パナソニックの収納建具は、build_panasonic_parts.build_storage が、開く側の枠の面を
                    # ボード面から12.5出した位置にもう置いている。さらにずらすと反対側の端が壁から
                    # 枠厚分飛び出す（user 11/13）
                    shift = 0.0
                across = line + (sy * shift if h else -sy * shift)  # rot 90: local +y -> model -X
            pos = (along, across) if h else (across, along)
            # end check (user 10/24): the frame's start end must meet the drawn opening's end (its
            # jamb). A big gap means the part was placed from the wrong grid line
            if word in WINDOWS:
                end_gap = 0.0
            elif hinge_low is True:
                end_gap = abs(flo - a)
            elif hinge_low is False:
                end_gap = abs(fhi - b)
            else:
                end_gap = min(abs(flo - a), abs(fhi - b))
            span = (flo, fhi)
            if word == OUTSET:
                # 戸は壁の表面を走るので、壁を切るのも柱をよけるのも開口（枠の外から外）だけ
                j = outset_jambs(_ents, S)
                if j:
                    # 枠の外を石膏ボード面（柱芯から65）に合わせる。パナソニックの部品は基準点が65ずれていて、
                    # 枠の外が柱芯から130だった（user 11/14）
                    along += d * (DOOR_EDGE - j[0])
                    pos = (along, across) if h else (across, along)
                    e1, e2 = along + d * j[0], along + d * j[1]
                    span = (min(e1, e2), max(e1, e2))
            out.append({"block": name, "kind": o["kind"], "word": word, "pos": pos, "rot": 0 if h else 90,
                        "end_gap_mm": round(end_gap),
                        "sx": d, "sy": sy, "horizontal": h, "line": line, "span": span,
                        "width": w, "outward": s_out, "inner_mm": inner,
                        "label": (o.get("label") or {}).get("code"), "hinge_known": hinge_low is not None})
    return out, skipped
