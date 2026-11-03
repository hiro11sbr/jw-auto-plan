"""Build 建具部品集.jww: one block per plan part, laid out on a 455 grid, plus a CSV index.

usage: build_parts_library.py <parts_dir> <config.json> <out_dir>
writes <out_dir>/部品集_<category>.jww (室内ドア / 窓 / 玄関ドア) and <out_dir>/建具部品集_一覧.csv

Maker-independent: every part is described in <parts_dir>/parts_spec.json (source sheet + bbox
of its plan piece). Base point of each block = the CENTRE OF THE LEFT COLUMN on the wall centre
line, with the part where it is normally built (place_fittings.std_offset): a person copies a
part by clicking a column centre (user, 2026-10-22). On the sheet every part stands between two
105 columns, like the user's 部品.jww. Windows and the entrance door have their maker reference
point on the outer face of the added insulation.
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import ezjww

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from place_fittings import OUTWARD, std_bay, std_offset  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
parts_dir, cfg_path, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
spec = json.load(open(parts_dir / "parts_spec.json", encoding="utf-8"))
cfg = json.load(open(cfg_path, encoding="utf-8"))
S = spec["sheet_scale"]
L = cfg["layers"]
PART_LAYER = L["wall"]["layer"]  # マーク、物 (reference drawings: everything drawn on this layer)
GRID_LAYER = 0xE
SKIP_LAYERS = ("POINT", "MOJI")  # maker reference points, dimensions and text
FOLD_PAIR_MIN = cfg["interior_type"]["fold_pair_min_mm"]  # 折戸 this wide: a V at each jamb


def xy_of(e):
    if e["type"] == "LINE":
        return [(e["start_x"], e["start_y"]), (e["end_x"], e["end_y"])]
    if e["type"] in ("ARC", "CIRCLE"):
        return [(e["center_x"] - e["radius"], e["center_y"] - e["radius"]),
                (e["center_x"] + e["radius"], e["center_y"] + e["radius"])]
    return []


def extract(part):
    """Entities of one plan piece, moved to the base point and rescaled from the sheet's own
    scale (window sheets are 1/100 or 1/150) to the library scale."""
    x0, y0, x1, y1 = part["bbox"]
    doc = ezjww.read_document(str(parts_dir / part["file"]))
    grp = doc["header"]["layer_groups"][0]
    names = [l["name"] for l in grp["layers"]]
    k = grp["scale"] / S
    sel = []
    for e in doc["entities"]:
        if e["type"] not in ("LINE", "ARC", "CIRCLE"):
            continue
        if any(s in names[e["base"]["layer"]] for s in SKIP_LAYERS):
            continue
        # a user's drawing: only the fitting's own layers (its columns are on other layers, and
        # a column taken along shifted the base and was drawn twice, プラン3 1F 10/27)
        if "layers" in part and e["base"]["layer"] not in part["layers"]:
            continue
        if e["type"] == "ARC":  # a swing arc reaches outside the piece: test its centre
            inside = x0 <= e["center_x"] <= x1 and y0 <= e["center_y"] <= y1
        else:
            inside = all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in xy_of(e))
        if inside:
            sel.append(e)
    # base: left frame outer edge, frame-depth middle (from the horizontal lines only, so an
    # open sash or leaf line does not shift it)
    hor = [e for e in sel if e["type"] == "LINE" and abs(e["start_y"] - e["end_y"]) < 1e-6]
    xs = [x for e in hor for x in (e["start_x"], e["end_x"])]
    ys = [e["start_y"] for e in hor]
    bx, by = min(xs), (min(ys) + max(ys)) / 2
    x_hi = max(xs)
    # LIXIL sheets mark the frame's outer edges with reference points (×印の円). Their spacing
    # is exactly the catalogue W, while the drawn lines reach ~25 further (small stubs beside
    # the frame), so when the marks are there, the left mark is the base point.
    # The marks also give the depth position of the base point (windows: placed on the outer
    # face of the added insulation).
    marks = sorted((e["center_x"], e["center_y"]) for e in doc["entities"]
                   if e["type"] == "CIRCLE" and "POINT" in names[e["base"]["layer"]]
                   and x0 - 0.5 <= e["center_x"] <= x1 + 0.5 and y0 - 0.5 <= e["center_y"] <= y1 + 0.5)
    if len(marks) >= 2:
        bx, x_hi, by = marks[0][0], marks[-1][0], marks[0][1]
    out = []
    for e in sel:
        e = dict(e)
        # (flag 0: a user's drawing may carry curve-group flags the writer cannot save)
        e["base"] = dict(e["base"], layer=PART_LAYER, layer_group=L["group"], flag=0)
        if e["type"] == "LINE":
            e["start_x"] = (e["start_x"] - bx) * k; e["end_x"] = (e["end_x"] - bx) * k
            e["start_y"] = (e["start_y"] - by) * k; e["end_y"] = (e["end_y"] - by) * k
        else:
            e["center_x"] = (e["center_x"] - bx) * k; e["center_y"] = (e["center_y"] - by) * k
            e["radius"] *= k
            if "start_angle" in e:  # some maker sheets store negative angles
                e["start_angle"] %= 2 * math.pi
        out.append(e)
    width_paper = (x_hi - bx) * k
    return out, width_paper


def leaf_to_plus_y(entities, w_paper):
    """A swing door opens toward the face its leaf (扉) sits on. Maker sheets may draw the leaf
    on the -y face; mirror such a part so the leaf, and so the swing, is on +y (user: the swing
    line was on the side without the door, 2026-10-22)."""
    leaf_y = [e["start_y"] for e in entities if e["type"] == "LINE" and abs(e["start_y"] - e["end_y"]) < 1e-6
              and 0.3 * w_paper < abs(e["end_x"] - e["start_x"]) < 0.97 * w_paper]  # leaves (one or two), not the full-width head
    if not leaf_y or sum(leaf_y) / len(leaf_y) >= 0:
        return entities
    for e in entities:
        if e["type"] == "LINE":
            e["start_y"], e["end_y"] = -e["start_y"], -e["end_y"]
        else:
            e["center_y"] = -e["center_y"]
            if "start_angle" in e:
                e["start_angle"] = (-(e["start_angle"] + e.get("arc_angle", 0.0))) % (2 * math.pi)
    return entities


def chamfer_to_plus_y(entities):
    """A 折戸 opens toward its front face: the face where each panel is chamfered at both sides,
    so the panel joints show V notches (W734: 4 chamfers, W1644: 8); the back face is one flat
    line. Mirror the part so the chamfered face is on +y (user 10/27: the V had twice been drawn
    on the flat back face)."""
    horiz = [e for e in entities if e["type"] == "LINE" and abs(e["start_y"] - e["end_y"]) < 1e-6]
    longest = max(abs(e["end_x"] - e["start_x"]) for e in horiz)
    panel = [e for e in horiz if 0.15 * longest < abs(e["end_x"] - e["start_x"]) < 0.7 * longest]
    if not panel:
        return entities
    lo, hi = min(e["start_y"] for e in panel), max(e["start_y"] for e in panel)
    diag = [(e["start_y"] + e["end_y"]) / 2 for e in entities if e["type"] == "LINE"
            and abs(e["start_y"] - e["end_y"]) > 1e-6 and abs(e["start_x"] - e["end_x"]) > 1e-6
            and lo - 1e-6 <= min(e["start_y"], e["end_y"]) and max(e["start_y"], e["end_y"]) <= hi + 1e-6]
    if not diag or sum(diag) / len(diag) >= (lo + hi) / 2:
        return entities
    for e in entities:
        if e["type"] == "LINE":
            e["start_y"], e["end_y"] = -e["start_y"], -e["end_y"]
        else:
            e["center_y"] = -e["center_y"]
            if "start_angle" in e:
                e["start_angle"] = (-(e["start_angle"] + e.get("arc_angle", 0.0))) % (2 * math.pi)
    return entities


def swing(entities, tmp_msp, leaves=1):
    """Add 90-degree swings (arc + open leaf line) opening toward +y: one leaf hinged at the
    left, or two leaves hinged at both ends (double door)."""
    horiz = [e for e in entities if e["type"] == "LINE" and abs(e["start_y"] - e["end_y"]) < 1e-6]
    longest = max(abs(e["end_x"] - e["start_x"]) for e in horiz)
    leaf = [e for e in horiz if abs(e["end_x"] - e["start_x"]) > 0.8 * longest]
    xl = min(min(e["start_x"], e["end_x"]) for e in leaf)
    xr = max(max(e["start_x"], e["end_x"]) for e in leaf)
    yt = max(e["start_y"] for e in leaf)
    attrs = {"layer": PART_LAYER, "pen_color": L["wall"]["structure_color"], "pen_style": 1}
    r = (xr - xl) / leaves
    out = [tmp_msp.add_arc((xl, yt), r, start_angle=0, sweep_angle=90, jwwattribs=attrs),
           tmp_msp.add_line((xl, yt), (xl, yt + r), jwwattribs=attrs)]
    if leaves == 2:
        out += [tmp_msp.add_arc((xr, yt), r, start_angle=90, sweep_angle=90, jwwattribs=attrs),
                tmp_msp.add_line((xr, yt), (xr, yt + r), jwwattribs=attrs)]
    return entities + out


def folding(entities, tmp_msp, pairs=1, angle_deg=60):
    """Add the 折戸 開閉線 toward +y: each folding pair drawn half open as a V hanging from its
    jamb, its two panels (each half the pair) at angle_deg to the wall — the way the plans draw
    it. Under 910 one pair hinged at the left; wider, one pair at each jamb (user 2026-10-23).
    Maker sheets draw only the closed panels."""
    horiz = [e for e in entities if e["type"] == "LINE" and abs(e["start_y"] - e["end_y"]) < 1e-6]
    longest = max(abs(e["end_x"] - e["start_x"]) for e in horiz)
    leaf = [e for e in horiz if abs(e["end_x"] - e["start_x"]) > 0.8 * longest]
    xl = min(min(e["start_x"], e["end_x"]) for e in leaf)
    xr = max(max(e["start_x"], e["end_x"]) for e in leaf)
    yt = max(e["start_y"] for e in leaf)
    attrs = {"layer": PART_LAYER, "pen_color": L["wall"]["structure_color"], "pen_style": 1}
    pair = (xr - xl) / pairs
    panel = pair / 2
    a = math.radians(angle_deg)
    out = []
    for x0, sgn in ((xl, 1), (xr, -1))[:pairs]:
        apex = (x0 + sgn * panel * math.cos(a), yt + panel * math.sin(a))
        end = (x0 + sgn * 2 * panel * math.cos(a), yt)
        out += [tmp_msp.add_line((x0, yt), apex, jwwattribs=attrs),
                tmp_msp.add_line(apex, end, jwwattribs=attrs)]
    return entities + out


def yokosuberi(entities, tmp_msp, out_mm=250.0):
    """横すべり窓の開閉線 (user 10/24): a rectangle standing out_mm out of the window face (toward
    +y, the outside), and from its two outer corners lines to the centre of the glass face.
    Maker sheets draw only the closed sash."""
    horiz = [e for e in entities if e["type"] == "LINE" and abs(e["start_y"] - e["end_y"]) < 1e-6]
    longest = max(abs(e["end_x"] - e["start_x"]) for e in horiz)
    full = [e for e in horiz if abs(e["end_x"] - e["start_x"]) > 0.8 * longest]
    xl = min(min(e["start_x"], e["end_x"]) for e in full)
    xr = max(max(e["start_x"], e["end_x"]) for e in full)
    ys = sorted(e["start_y"] for e in full)
    y_face, y_glass = ys[-1], (ys[0] + ys[-1]) / 2
    y_out = y_face + out_mm / S
    attrs = {"layer": PART_LAYER, "pen_color": L["wall"]["structure_color"], "pen_style": 1}
    mid = (xl + xr) / 2
    out = [tmp_msp.add_line((xl, y_face), (xl, y_out), jwwattribs=attrs),
           tmp_msp.add_line((xl, y_out), (xr, y_out), jwwattribs=attrs),
           tmp_msp.add_line((xr, y_out), (xr, y_face), jwwattribs=attrs),
           tmp_msp.add_line((xl, y_out), (mid, y_glass), jwwattribs=attrs),
           tmp_msp.add_line((xr, y_out), (mid, y_glass), jwwattribs=attrs)]
    return entities + out


def ready_part(part):
    """A part already in library form (e.g. cut out of a user-corrected plan by
    extract_user_part.py): taken as it is."""
    doc = ezjww.read_document(str(parts_dir / part["file"]))
    out = []
    for e in doc["entities"]:
        if e["type"] in ("LINE", "ARC", "CIRCLE"):
            e = dict(e)
            e["base"] = dict(e["base"], layer=PART_LAYER, layer_group=L["group"])
            out.append(e)
    return out


# one file per category, so each stays a size a person can browse (user, 2026-10-22)
CATEGORY = {"片開き": "室内ドア", "両開き": "室内ドア", "片引戸": "室内ドア", "引違い戸": "室内ドア",
            "折戸": "室内ドア", "UBドア": "室内ドア", "引違い窓": "窓", "縦すべり窓": "窓", "FIX窓": "窓", "横すべり窓": "窓", "縦すべり+FIX窓": "窓",
            "玄関片開き": "玄関ドア", "玄関親子": "玄関ドア"}


def new_sheet(title):
    d = ezjww.new(paper_size=3, memo=f"建具部品集 {title}\r\n")
    g = d.options["layer_groups"][L["group"]]
    g["name"], g["scale"] = L["group_name"], S
    for i, n in enumerate(L["names"]):
        g["layers"][i]["name"] = n
    msp = d.modelspace()
    grid = cfg["input"]["grid_mm"] / S  # 455 mm on paper
    for i in range(SHEET_COLS + 1):
        x = GX0 + i * grid
        msp.add_line((x, GY0), (x, GY0 - SHEET_ROWS * grid), jwwattribs={"layer": GRID_LAYER, "pen_color": 9, "pen_style": 9})
    for j in range(SHEET_ROWS + 1):
        y = GY0 - j * grid
        msp.add_line((GX0, y), (GX0 + SHEET_COLS * grid, y), jwwattribs={"layer": GRID_LAYER, "pen_color": 9, "pen_style": 9})
    msp.add_text(f"建具部品集　{title}　（基準点＝左の柱の中心）", (GX0, GY0 + 4), (GX0 + 90, GY0 + 4),
                 jwwattribs={"layer": 0xA})
    return d, msp


SHEET_COLS, SHEET_ROWS = 30, 22
GX0, GY0 = -190.0, 140.0


def main():
    tmp = ezjww.new(paper_size=3).modelspace()
    grid = cfg["input"]["grid_mm"] / S  # 455 mm on paper
    sheets = {}

    # one row per kind, parts left to right between two columns (like 部品.jww);
    # a full row wraps to the next free row
    row_of = {"片開き": 2, "両開き": 2, "UBドア": 2, "片引戸": 5, "引違い戸": 5, "折戸": 8,
              "引違い窓": 2, "縦すべり窓": 8, "FIX窓": 11, "横すべり窓": 14, "縦すべり+FIX窓": 17, "玄関片開き": 2, "玄関親子": 2}
    parts = sorted(spec["parts"], key=lambda p: (CATEGORY.get(p["kind"], ""), row_of.get(p["kind"], 20), p["width"]))
    sources = {}
    src_csv = parts_dir / "元データ一覧.csv"
    if src_csv.exists():
        sources = {r["ファイル"]: r for r in csv.DictReader(open(src_csv, encoding="utf-8-sig"))}
    col_attr = {"layer": PART_LAYER, "pen_color": L["wall"]["structure_color"], "pen_style": 1}
    ref_out = cfg["offsets_mm"]["exterior_side"]["added_insulation"]
    h = cfg["column"]["size_mm"] / 2 / S
    next_col_of = defaultdict(dict)
    index = []
    for p in parts:
        word, w = p["kind"], p["width"]
        cat = CATEGORY.get(word, "その他")
        if cat not in sheets:
            sheets[cat] = new_sheet(cat)
        d, msp = sheets[cat]
        bay = std_bay(word, w)
        if p.get("ready"):
            ents, off = ready_part(p), p["offset_mm"]
        else:
            ents, wpaper = extract(p)
            if p["op"] in ("swing", "double_swing"):
                ents = leaf_to_plus_y(ents, wpaper)
                ents = swing(ents, tmp, leaves=2 if p["op"] == "double_swing" else 1)
            elif p["kind"] == "横すべり窓":
                ents = yokosuberi(ents, tmp)
            elif p["op"] == "folding":
                ents = chamfer_to_plus_y(ents)
                ents = folding(ents, tmp, pairs=1 if w < FOLD_PAIR_MIN else 2)
            off = std_offset(word, w)
            # move from (frame left edge, reference) to (left column centre, wall centre)
            dx, dy = off / S, (ref_out / S if word in OUTWARD else 0.0)
            # a measured correction: the sheet's outermost frame line is not the frame's true
            # outer face (LIXIL sliding doors sat 11.38 from the column face instead of 12.5,
            # cutting into the board, user 10/26)
            dx += p.get("base_shift_mm", 0.0) / S
            # a size the maker has no drawing for: the nearest drawing shortened. Each [t, d]
            # moves every point beyond t mm (from the column centre) by d mm, so the jambs and
            # the 中方立 keep their shapes (W1454 from W1644, measured on the user's 部品.jww)
            stretch = [(t / S, dd / S) for t, dd in p.get("stretch_mm", [])]

            def sx(x):
                return x + sum(dd for t, dd in stretch if x > t)
            for e in ents:
                if e["type"] == "LINE":
                    e["start_x"] = sx(e["start_x"] + dx); e["end_x"] = sx(e["end_x"] + dx)
                    e["start_y"] += dy; e["end_y"] += dy
                else:
                    e["center_x"] = sx(e["center_x"] + dx); e["center_y"] += dy
        name = f'{p["maker"]}_{p["series"]}_{p["kind"]}_W{w}'
        d.add_block(name, ents)
        next_col = next_col_of[cat]
        row = row_of.get(p["kind"], 20)
        while next_col.get(row, 1) + bay / S / grid + 1 > SHEET_COLS:
            row += 3
        col = next_col.get(row, 1)
        x = GX0 + col * grid
        y = GY0 - row * grid
        msp.add_block_ref(name, (x, y), jwwattribs={"layer": PART_LAYER})
        # the two columns the part stands between (105 square with both diagonals)
        for cx in (x, x + bay / S):
            c4 = [(cx - h, y - h), (cx + h, y - h), (cx + h, y + h), (cx - h, y + h)]
            for i in range(4):
                msp.add_line(c4[i], c4[(i + 1) % 4], jwwattribs=col_attr)
            msp.add_line(c4[0], c4[2], jwwattribs=col_attr)
            msp.add_line(c4[1], c4[3], jwwattribs=col_attr)
        label = f'{p["series"]} {p["kind"]} W{w}'
        msp.add_text(label, (x, y - 5.5), (x + len(label) * 1.6, y - 5.5), jwwattribs={"layer": 0xA})
        next_col[row] = col + int(round(bay / S / grid)) + 2
        src = sources.get(p["file"], {})
        index.append({"ファイル": f"部品集_{cat}.jww", "ブロック名": name, "メーカー": p["maker"],
                      "シリーズ": p["series"], "種類": p["kind"], "幅W": w, "柱芯から枠外まで": round(off, 1),
                      "柱芯々": round(bay), "備考": p["note"], "元データ": p["file"],
                      "元データ取得日": src.get("取得日", ""), "元データ公開日": src.get("公開日", ""),
                      "商品ページ": src.get("商品ページ", "")})
        print(f"{cat} {name}: {len(ents)} entities, column centre to frame {off:.1f}, bay {bay:.0f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat, (d, _msp) in sheets.items():
        d.saveas(str(out_dir / f"部品集_{cat}.jww"))
    with open(out_dir / "建具部品集_一覧.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    print("saved", out_dir, sorted(sheets))


if __name__ == "__main__":
    main()
