"""パナソニックの水廻り（キッチン・UB）の平面図 → 部品集 jww（ブロック）と一覧CSV。

usage: build_waterworks_parts.py <パナソニック_水廻りのフォルダ> <出力フォルダ>
  キッチン：リビングステーション V-style I型 間口2550（奥行き650）LVI25A（採用中の標準）
  UB：1616サイズ（ビバス標準タイプ スマイル浴槽）。ドアが下の壁（AR）＝正面にシャワー・鏡側、
      ドアが右の壁（BR）＝正面に浴槽側。※どの浴槽・シリーズにするかは仮（選定は任せるとの指示 11/8）
- レイヤー・線色は1つ（壁の線と同じ）。図面に混じる寸法・基準線・文字は入れず、部品の本体の線だけ（隣と離れた線は外す）
- データは公開版に入れない（パナソニックの利用条件）
"""
import csv
import json
import math
import sys
from pathlib import Path

import ezjww

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ub_guide import guide_box  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
S = 50.0
PARTS = [
    # (folder, file code, block name, label, note)
    ("キッチン_dxf", "LVI25A", "パナソニック_キッチン_LVI25A_W2550", "リビングステーション V-style I型 W2550（奥行き650）",
     "標準のキッチン（間口2550）。壁側の線・寸法は部品の外に出ることがある"),
    ("キッチン_dxf", "LVI25IA", "パナソニック_キッチン_LVI25IA_W2550_IH", "リビングステーション V-style I型 W2550（奥行き650）トリプルワイドIH",
     "IHコンロの型（LVI25IA）。間口2550。ガスの型は LVI25A"),
    ("../TOTO/キッチン_dxf", "G310982-B1", "TOTO_キッチン_ミッテI型_W2550", "TOTO ミッテ I型 W2550（KPS2550SAA + SSE3S + FSP1S、奥行き650）",
     "TOTO選択時の標準（ミッテ）。図はガスコンロ・シンクが右。出典：TOTO 一括ダウンロード 6_DXF.ZIP"),
    ("バスルーム_dxf", "BV6A3AA", "パナソニック_UB_1616_ドア下_シャワー鏡が正面", "ビバス標準 1616 スマイル浴槽 AR（ドアが下の壁・正面にシャワー側）",
     "UBは1616サイズ。浴槽が正面の型（BR）と対にして2種"),
    ("バスルーム_dxf", "BV6A3BA", "パナソニック_UB_1616_ドア右_浴槽が正面", "ビバス標準 1616 スマイル浴槽 BR（ドアが右の壁・正面に浴槽側）",
     "UBは1616サイズ。シャワー・鏡が正面の型（AR）と対にして2種"),
    # 2275×1820 の浴室用（パナソニックには 1620 がなく、いちばん近いのが 1621。user 11/16 の試し）
    ("バスルーム_dxf", "BV2A3AA", "パナソニック_UB_1621_ドア下_シャワー鏡が正面", "ビバス標準 1621 スマイル浴槽 AR（ドアが下の壁・正面にシャワー側）",
     "2275×1820 の浴室用（パナソニックに1620はない）。BR と対"),
    ("バスルーム_dxf", "BV2A3BA", "パナソニック_UB_1621_ドア右_浴槽が正面", "ビバス標準 1621 スマイル浴槽 BR（ドアが右の壁・正面に浴槽側）",
     "2275×1820 の浴室用（パナソニックに1620はない）。AR と対"),
    # ペニンシュラ（前面に腰壁も壁もなく、キッチンの短い辺の一方だけが壁に接する型。user 11/16）。
    # パナソニックの「フラット対面」。コンロ・シンク側が下（作業する側）、上がダイニング側のフラットカウンター
    ("キッチン_dxf", "LVFI24A", "パナソニック_キッチン_LVFI24A_W2435_ペニンシュラ", "リビングステーション V-style フラット対面 W2435（奥行き933）",
     "ペニンシュラの型。コンロ・シンクが下側、上がフラットカウンター。間口は元の図の2390に近いもの"),
    ("キッチン_dxf", "LVFI25A", "パナソニック_キッチン_LVFI25A_W2585_ペニンシュラ", "リビングステーション V-style フラット対面 W2585（奥行き933）",
     "ペニンシュラの型。W2435 より少し長いもの"),
    # 洗面化粧台（user 11/16）。壁が上（+y）、間口が x、奥行き 530〜550。元の図の洗面台の間口に近いものを置く
    ("洗面_dxf", "M60KSC1A", "パナソニック_洗面_MLine_W600", "M-Line 間口600（洗面ボールタイプ）", "壁が上。間口600・奥行き530"),
    ("洗面_dxf", "M75KSC1A", "パナソニック_洗面_MLine_W750", "M-Line 間口750・1面鏡（洗面ボールタイプ）", "壁が上。間口750・奥行き530"),
    ("洗面_dxf", "GQC905C_A", "パナソニック_洗面_CLine_W900", "C-Line 間口900（GQC905C）", "壁が上。間口900・奥行き550"),
    ("洗面_dxf", "GQC105A_A", "パナソニック_洗面_CLine_W1000", "C-Line 間口1000（GQC105A）", "壁が上。間口1000・奥行き550"),
    ("洗面_dxf", "GQC125A_A", "パナソニック_洗面_CLine_W1200", "C-Line 間口1200（GQC125A）", "壁が上。間口1200・奥行き550"),
    # スロップシンク（洗濯流し）：TOTO SK507（流し 510×500×深さ405）。間口 750 以下の「洗面台」はスロップシンク（user 11/17・11/18）
    ("../TOTO/洗面所_dxf/SK507", "G201651-B1", "TOTO_スロップシンク_SK507", "TOTO 洗濯用流し SK507（間口510・奥行き500、壁が上）",
     "壁が上。G201651-B1（意匠図・平面図）。出典：TOTO 一括ダウンロード 3_DXF.ZIP"),
]


KITCHEN_BOX = {"LVFI24A": (2435, 933), "LVFI25A": (2585, 933)}  # 外形（間口, 奥行き）。ほかは 2550×650


def ents_of(path):
    import ezdxf
    return [e for e in ezdxf.readfile(str(path)).modelspace() if e.dxftype() in ("LINE", "ARC", "CIRCLE")]


def bbox(e):
    if e.dxftype() == "LINE":
        a, b = e.dxf.start, e.dxf.end
        return min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y)
    c, r = e.dxf.center, e.dxf.radius
    if e.dxftype() == "ARC":
        a0, a1 = e.dxf.start_angle, e.dxf.end_angle
        if a1 < a0:
            a1 += 360
        ps = [(c.x + r * math.cos(math.radians(a)), c.y + r * math.sin(math.radians(a))) for a in (a0, (a0 + a1) / 2, a1)]
        return min(p[0] for p in ps), min(p[1] for p in ps), max(p[0] for p in ps), max(p[1] for p in ps)
    return c.x - r, c.y - r, c.x + r, c.y + r


def biggest_cluster(ents, gap=10.0):
    groups = []
    for b, e in sorted(((bbox(e), e) for e in ents), key=lambda t: t[0][0]):
        hit = [g for g in groups if b[0] <= g["b"][2] + gap and b[2] >= g["b"][0] - gap
               and b[1] <= g["b"][3] + gap and b[3] >= g["b"][1] - gap]
        if not hit:
            groups.append({"b": list(b), "e": [e]})
            continue
        g = hit[0]
        for h in hit[1:]:
            g["e"] += h["e"]
            g["b"] = [min(g["b"][0], h["b"][0]), min(g["b"][1], h["b"][1]), max(g["b"][2], h["b"][2]), max(g["b"][3], h["b"][3])]
            groups.remove(h)
        g["e"].append(e)
        g["b"] = [min(g["b"][0], b[0]), min(g["b"][1], b[1]), max(g["b"][2], b[2]), max(g["b"][3], b[3])]
    return max(groups, key=lambda g: len(g["e"]))


def pts_of(e):
    if e.dxftype() == "LINE":
        return [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
    b = bbox(e)
    return [(b[0], b[1]), (b[2], b[3])]


def drop_guides(ents, box):
    """Lines that run out of the part's body box are CAD guides (the cross at a corner): left out (user 11/8)."""
    x0, y0, x1, y1 = box
    out = []
    for e in ents:
        if e.dxftype() == "LINE" and any(not (x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1) for x, y in pts_of(e)):
            continue
        out.append(e)
    return out


def drop_shower_circle(ents):
    """UB: the big circle and its cross in front of the door are guides (user 11/8). The cross: a
    horizontal and a vertical line of the same length crossing at their middles; the longest such cross
    is the one on the circle. Everything inside a square around its centre goes."""
    hs = [e for e in ents if e.dxftype() == "LINE" and abs(e.dxf.start.y - e.dxf.end.y) < 0.01]
    vs = [e for e in ents if e.dxftype() == "LINE" and abs(e.dxf.start.x - e.dxf.end.x) < 0.01]
    best = None
    for h_ in hs:
        for v_ in vs:
            hl, vl = abs(h_.dxf.end.x - h_.dxf.start.x), abs(v_.dxf.end.y - v_.dxf.start.y)
            cx, cy = (h_.dxf.start.x + h_.dxf.end.x) / 2, (v_.dxf.start.y + v_.dxf.end.y) / 2
            if hl >= 300 and vl >= 300 and abs(v_.dxf.start.x - cx) < 2 and abs(h_.dxf.start.y - cy) < 2:
                if best is None or hl + vl > best[0]:
                    best = (hl + vl, cx, cy, max(hl, vl) / 2)
    if not best:
        return ents
    _t, cx, cy, half = best
    half += 40
    keep = []
    for e in ents:
        b = bbox(e)
        inside = cx - half <= b[0] and b[2] <= cx + half and cy - half <= b[1] and b[3] <= cy + half
        is_cross = e.dxftype() == "LINE" and ((abs(e.dxf.start.y - e.dxf.end.y) < 0.01 and abs(e.dxf.start.y - cy) < 2)
                                              or (abs(e.dxf.start.x - e.dxf.end.x) < 0.01 and abs(e.dxf.start.x - cx) < 2))
        short = e.dxftype() == "LINE" and math.dist((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)) < 60
        if inside and (e.dxftype() in ("ARC", "CIRCLE") or is_cross or short):
            continue
        keep.append(e)
    return keep


def ub_door_to_fold(ents, box):
    """The UB's door is a folding door (折戸): the maker's swing arc and slanted lines are replaced by the
    折戸's V (half open, 60 degrees, hinged where the arc's centre is). Returns (kept entities, V lines)."""
    bx0, by0, bx1, by1 = box
    arcs = [e for e in ents if e.dxftype() == "ARC" and 350 <= e.dxf.radius <= 900
            and (min(abs(e.dxf.center.y - by0), abs(e.dxf.center.y - by1), abs(e.dxf.center.x - bx0), abs(e.dxf.center.x - bx1)) < 90)]
    if not arcs:
        return ents, []
    arc = max(arcs, key=lambda e: e.dxf.radius)
    c, r = arc.dxf.center, arc.dxf.radius
    a0, a1 = arc.dxf.start_angle, arc.dxf.end_angle
    if a1 < a0:
        a1 += 360
    mid = math.radians((a0 + a1) / 2)
    free = (c.x + r * math.cos(math.radians(a1 if False else a0)), c.y + r * math.sin(math.radians(a0)))
    ends = [(c.x + r * math.cos(math.radians(a)), c.y + r * math.sin(math.radians(a))) for a in (a0, a1)]
    # which wall the door is in: the unit's edge nearer to the hinge (arc centre)
    near_v = min(abs(c.x - bx0), abs(c.x - bx1)) < min(abs(c.y - by0), abs(c.y - by1))
    horizontal = not near_v
    face_ends = [p for p in ends if (abs(p[1] - c.y) < 3 if horizontal else abs(p[0] - c.x) < 3)]
    fx, fy = face_ends[0] if face_ends else ends[0]
    box = bbox(arc)
    keep = []
    slanted_near = lambda e: (e.dxftype() == "LINE" and abs(e.dxf.start.x - e.dxf.end.x) > 0.5 and abs(e.dxf.start.y - e.dxf.end.y) > 0.5
                              and 20 < math.dist((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)) < 600
                              and math.dist(((e.dxf.start.x + e.dxf.end.x) / 2, (e.dxf.start.y + e.dxf.end.y) / 2), (c.x, c.y)) < 800)
    for e in ents:
        if e is arc or slanted_near(e):
            continue
        if e.dxftype() == "LINE":
            mx, my = (e.dxf.start.x + e.dxf.end.x) / 2, (e.dxf.start.y + e.dxf.end.y) / 2
            across = abs(my - c.y) if horizontal else abs(mx - c.x)
            if box[0] - 30 <= mx <= box[2] + 30 and box[1] - 30 <= my <= box[3] + 30 and across > 45:
                continue  # a line of the door symbol (the open-door line, the slanted lines), not the frame on the wall
        keep.append(e)
    panel = r / 2
    a = math.radians(60)
    # the V runs from the hinge along the wall toward the middle of the unit and into the room
    # (not out of the unit: user 11/8)
    mx_, my_ = (bx0 + bx1) / 2, (by0 + by1) / 2
    if horizontal:
        sx = 1 if mx_ > c.x else -1
        sy = 1 if my_ > c.y else -1
        apex = (c.x + sx * panel * math.cos(a), c.y + sy * panel * math.sin(a))
        end = (c.x + sx * 2 * panel * math.cos(a), c.y)
    else:
        sy = 1 if my_ > c.y else -1
        sx = 1 if mx_ > c.x else -1
        apex = (c.x + sx * panel * math.sin(a), c.y + sy * panel * math.cos(a))
        end = (c.x, c.y + sy * 2 * panel * math.cos(a))
    lines_ = [((c.x, c.y), apex), (apex, end)]
    return keep, lines_


def kitchen_layout(ents):
    """The stove and the sink of a kitchen drawing (x ranges from the unit's left end, in the drawing's own
    direction): the body's outline lines are left out, the rest is grouped (gaps under 40); the group
    with the burners (circles) at the left or the right is the stove, the widest other one the sink."""
    inner = [e for e in ents if not (e.dxftype() == "LINE" and math.dist((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)) > 1500)]
    groups = []
    for b, e in sorted(((bbox(e), e) for e in inner), key=lambda t: t[0][0]):
        hit = [g for g in groups if b[0] <= g["b"][2] + 40 and b[2] >= g["b"][0] - 40 and b[1] <= g["b"][3] + 40 and b[3] >= g["b"][1] - 40]
        if not hit:
            groups.append({"b": list(b), "e": [e]})
            continue
        g = hit[0]
        for h in hit[1:]:
            g["e"] += h["e"]
            g["b"] = [min(g["b"][0], h["b"][0]), min(g["b"][1], h["b"][1]), max(g["b"][2], h["b"][2]), max(g["b"][3], h["b"][3])]
            groups.remove(h)
        g["e"].append(e)
        g["b"] = [min(g["b"][0], b[0]), min(g["b"][1], b[1]), max(g["b"][2], b[2]), max(g["b"][3], b[3])]
    big = [g for g in groups if len(g["e"]) >= 10]
    if len(big) < 2:
        return None
    circles = lambda g: sum(1 for e in g["e"] if e.dxftype() == "CIRCLE")
    stove = max(big, key=circles)
    others = [g for g in big if g is not stove]
    sink = max(others, key=lambda g: g["b"][2] - g["b"][0])
    return {"stove": [round(stove["b"][0]), round(stove["b"][2])], "sink": [round(sink["b"][0]), round(sink["b"][2])],
            "stove_side": "左" if stove["b"][0] < sink["b"][0] else "右"}


def main():
    cfg = json.load(open(Path(__file__).resolve().parent.parent / "05_設定" / "default.json", encoding="utf-8"))
    wl = cfg["layers"]["wall"]
    L = cfg["layers"]
    attrs = {"layer": wl["layer"], "pen_color": wl["structure_color"], "pen_style": 1}
    d = ezjww.new(paper_size=3, memo="水廻り部品集（パナソニック）\r\n")
    g = d.options["layer_groups"][L["group"]]
    g["name"], g["scale"] = L["group_name"], S
    for i, n in enumerate(L["names"]):
        g["layers"][i]["name"] = n
    msp = d.modelspace()
    index = []
    x_cursor = -150.0
    for folder, code, name, label, note in PARTS:
        base_ = (root / folder).resolve()
        path = next(base_.rglob(f"{code}.DXF"), None) or next(base_.rglob(f"{code}.dxf"), None)
        if path is None:
            print("no file", code)
            continue
        cl = biggest_cluster(ents_of(path))
        extra_lines = []
        lay_info = None
        if "キッチン" in folder:
            kw, kd = KITCHEN_BOX.get(code, (2550, 650))
            cl["e"] = drop_guides(cl["e"], (0, -kd, kw, 0))
            cl["b"] = [0, -kd, kw, 0]
            lay_info = kitchen_layout(cl["e"])
        elif "洗面" in folder:
            pass  # 洗面化粧台：壁が上（+y）、間口が x。部品の線はそのまま（11/16）
        else:
            cl["e"] = drop_guides(cl["e"], (-5, -35, cl["b"][2] + 5, cl["b"][3] + 5))
            cl["e"] = drop_shower_circle(cl["e"])
            cl["e"], extra_lines = ub_door_to_fold(cl["e"], tuple(cl["b"]))
        bx, by = cl["b"][0], cl["b"][1]
        guide = None
        if "バスルーム" in folder:
            # 部品の原点を、4つの角の L字ガイドの長方形の中心にする（置く時はこの点を UB の壁芯の中心に合わせる）
            segs = [((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)) for e in cl["e"] if e.dxftype() == "LINE"]
            segs += [tuple(pq) for pq in extra_lines]
            guide = guide_box(segs)
            if guide is not None:
                bx, by = (guide[0] + guide[2]) / 2, (guide[1] + guide[3]) / 2
            else:
                # L字ガイドのない部品もある（user 11/18）。その時は外形（1000mm 以上の線＝壁パネルの外側）の範囲の中心
                # （user 11/18「外形から中心求めて」。パナソニックの UB はパネルの外形がガイドより左右に 7.5mm 偏っている）
                lg = [pq for pq in segs if math.dist(*pq) >= 1000]
                xs_ = [v[0] for pq in lg for v in pq]
                ys_ = [v[1] for pq in lg for v in pq]
                bx, by = (min(xs_) + max(xs_)) / 2, (min(ys_) + max(ys_)) / 2
                print(f"{name}: L字ガイドなし。外形の中心を原点にした")
        ents = []
        for e in cl["e"]:
            if e.dxftype() == "LINE":
                p, q = ((e.dxf.start.x - bx) / S, (e.dxf.start.y - by) / S), ((e.dxf.end.x - bx) / S, (e.dxf.end.y - by) / S)
                if math.dist(p, q) > 0.001:
                    ents.append(("L", p, q))
            elif e.dxftype() == "CIRCLE":
                ents.append(("C", ((e.dxf.center.x - bx) / S, (e.dxf.center.y - by) / S), e.dxf.radius / S))
            else:
                a0, a1 = e.dxf.start_angle, e.dxf.end_angle
                ents.append(("A", ((e.dxf.center.x - bx) / S, (e.dxf.center.y - by) / S), e.dxf.radius / S, a0 % 360.0, (a1 - a0) % 360.0))
        tmp = ezjww.new(paper_size=3).modelspace()
        block = []
        for t in ents:
            if t[0] == "L":
                block.append(tmp.add_line(t[1], t[2], jwwattribs=attrs))
            elif t[0] == "C":
                block.append(tmp.add_circle(t[1], t[2], jwwattribs=attrs))
            else:
                block.append(tmp.add_arc(t[1], t[2], start_angle=t[3], sweep_angle=t[4], jwwattribs=attrs))
        for (p, q) in extra_lines:
            block.append(tmp.add_line(((p[0] - bx) / S, (p[1] - by) / S), ((q[0] - bx) / S, (q[1] - by) / S), jwwattribs=attrs))
        d.add_block(name, block)
        w_mm, h_mm = cl["b"][2] - cl["b"][0], cl["b"][3] - cl["b"][1]
        msp.add_block_ref(name, (x_cursor + (bx - cl["b"][0]) / S, (by - cl["b"][1]) / S), jwwattribs={"layer": wl["layer"]})
        msp.add_text(label, (x_cursor, -6), (x_cursor + len(label) * 1.6, -6), jwwattribs={"layer": 0xA})
        x_cursor += w_mm / S + 25
        index.append({"ブロック名": name, "メーカー": ("TOTO" if name.startswith("TOTO") else "パナソニック"), "名称": label, "外形の幅(mm)": round(w_mm), "外形の奥行き(mm)": round(h_mm),
                      "元データ": path.name, "備考": note,
                      "コンロの位置(左端から)": (f"{lay_info['stove'][0]}〜{lay_info['stove'][1]}" if lay_info else ""),
                      "シンクの位置(左端から)": (f"{lay_info['sink'][0]}〜{lay_info['sink'][1]}" if lay_info else ""),
                      "コンロの側（図の向き）": (lay_info["stove_side"] if lay_info else ""),
                      "L字ガイドの長方形(mm)": (f"{guide[2] - guide[0]:.1f} x {guide[3] - guide[1]:.1f}（部品の原点＝その中心）" if guide
                                                  else "ガイドなし（原点＝外形の中心）" if "バスルーム" in folder else "")})
        print(f"{name}: {len(ents)}本 {w_mm:.0f} x {h_mm:.0f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    d.saveas(str(out_dir / "水廻り部品集.jww"))
    with open(out_dir / "水廻り部品集_一覧.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)


if __name__ == "__main__":
    main()
