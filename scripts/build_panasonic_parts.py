"""パナソニック ベリティス（DXF）→ 部品集用の「できあがり部品」jww と parts_spec.json を作る。

usage: build_panasonic_parts.py <ベリティスのフォルダ>
  <フォルダ>\\スタンダード2026_dxf\\indoor60.csv と DXF を読み、<フォルダ>\\ready\\*.jww と
  <フォルダ>\\parts_spec.json を書く。続けて
  build_parts_library.py <フォルダ> 05_設定\\default.json <フォルダ>\\部品集
  で部品集（建具部品集_室内ドア.jww と一覧CSV）ができる。

決まり（ユーザー 11/7）
- 片開きは全幅種（UW）、枠はケーシングなしの固定枠（3方枠）155。壁130に対し枠155＝壁面から12.5ずつ飛び出す
- 建具の高さ（UH）は部品集の備考に残す（標準設定の有無・選べるサイズの根拠）。平面図は高さで変わらない
- 開き戸の円弧は、水平断面で枠の面とドア面が揃っている面に描く（段の大きい側には描かない）。
  蝶番の丸を軸にする。円弧は扉の戸先の端から始める（蝶番の丸の位置から扉面まで円弧を上げる）。
  開いた扉の線は蝶番の丸の中心から引き、円弧の先端にそろえる
- レイヤー・線色は1つ（円弧もドアと同じ色）
- データは公開版に入れない（パナソニックの利用条件）
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

import ezjww

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from place_fittings import std_bay, std_offset  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
root = Path(sys.argv[1])
dxf_root = root / "スタンダード2026_dxf"
S = 50.0
cfg_board = 65.0  # the board face from the wall centre
FRAME = "固定枠（３方枠）１５５ｍｍタイプ"
FRAME_OF = {"３枚連動片引き": "固定枠（３方枠）１９２ｍｍタイプ"}  # (this one has no 155 frame: its own 192)
PLAN_TOP_MIN_GAP = 150  # the plan section stands above the elevation by at least this


def zen2han(s):
    return s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def read_rows():
    rows = list(csv.reader(open(dxf_root / "indoor60.csv", encoding="cp932")))[1:]
    out = []
    for r in rows:
        if r[5] != "納まり図":
            continue
        m = re.match(r"(.+?)（[ＵＷ]+(\d+)／[ＵＨ]+(\d+)）", zen2han(r[7]).replace("ＵＷ", "UW"))
        m = re.match(r"(.+?)（(?:ＵＷ|Ｗ)([０-９]+)／(?:ＵＨ|Ｈ)([０-９]+)）", r[7])
        if not m:
            continue
        out.append({"type": m.group(1), "uw": int(zen2han(m.group(2))), "uh": int(zen2han(m.group(3))),
                    "frame": r[8].split("・")[0], "full_frame": r[8], "code": r[11], "group": r[6]})
    return out


def find_dxf(code):
    hits = list(dxf_root.rglob(f"{code}.DXF")) + list(dxf_root.rglob(f"{code}.dxf"))
    return hits[0] if hits else None


def plan_section(ents):
    """The horizontal section drawn above the elevation: everything above the largest gap
    in the top part of the sheet."""
    import ezdxf  # noqa: F401  (entities come from ezdxf)

    def yrange(e):
        if e.dxftype() == "LINE":
            return sorted((e.dxf.start.y, e.dxf.end.y))
        c = e.dxf.center
        return [c.y - e.dxf.radius, c.y + e.dxf.radius]
    ys = sorted({round(v) for e in ents for v in yrange(e)})
    # the biggest gap between the elevation's head and the section
    top = ys[-1]
    gaps = [(ys[i + 1] - ys[i], ys[i]) for i in range(len(ys) - 1) if ys[i] > 1500 and ys[i + 1] - ys[i] >= PLAN_TOP_MIN_GAP]
    cut = max(gaps)[1] if gaps else ys[0]
    return [e for e in ents if yrange(e)[0] > cut]


def rotate180(x, y, cx, cy):
    return 2 * cx - x, 2 * cy - y


def clear_opening(dxf_path, uw, uh):
    """有効開口 (clear opening) from the drawing: the width between the jambs' inner faces (plan) and the
    height from the floor line to the head's inner face (elevation), when the drawing shows them."""
    import ezdxf
    ents = [e for e in ezdxf.readfile(str(dxf_path)).modelspace() if e.dxftype() == "LINE"]
    plan_ents = [e for e in plan_section([e for e in ezdxf.readfile(str(dxf_path)).modelspace() if e.dxftype() in ("LINE", "ARC", "CIRCLE")]) if e.dxftype() == "LINE"]
    tallv = [e for e in plan_ents if abs(e.dxf.start.x - e.dxf.end.x) < 0.01 and abs(e.dxf.start.y - e.dxf.end.y) > 100]
    cw = None
    if tallv:
        fy0 = min(min(e.dxf.start.y, e.dxf.end.y) for e in tallv)
        xl = min(min(e.dxf.start.x, e.dxf.end.x) for e in tallv)
        # the jamb's thickness: the short horizontal line on the frame face at the left end
        th = [abs(e.dxf.end.x - e.dxf.start.x) for e in plan_ents if abs(e.dxf.start.y - e.dxf.end.y) < 0.01
              and abs(e.dxf.start.y - fy0) < 0.6 and 5 < abs(e.dxf.end.x - e.dxf.start.x) < 60 and min(e.dxf.start.x, e.dxf.end.x) < xl + 1]
        if th:
            cw = round(uw - 2 * max(th), 1)
    hy = sorted({round(e.dxf.start.y, 1) for e in ents if abs(e.dxf.start.y - e.dxf.end.y) < 0.01
                 and abs(e.dxf.end.x - e.dxf.start.x) > 0.8 * uw and e.dxf.start.y < uh + 50})
    ch = None
    if len(hy) >= 3:
        top = hy[-1]
        below = [y for y in hy if y < top - 5]
        if below:
            ch = round(max(below) - hy[0], 1)
    return cw, ch


surround_colors = (5, 1)
mid_post = [None]  # the 中方立's distance from the frame's left end (set by the last sliding part built)


def build_swing(dxf_path, uw, out_path, offset_mm, attrs, swing=True, outset=False, irizumi=False, surround=None, both_ways=False):
    import ezdxf
    ents = [e for e in ezdxf.readfile(str(dxf_path)).modelspace() if e.dxftype() in ("LINE", "ARC", "CIRCLE")]
    plan = plan_section(ents)
    lines = [e for e in plan if e.dxftype() == "LINE"]
    xs = [v for e in lines for v in (e.dxf.start.x, e.dxf.end.x)]
    ys = [v for e in lines for v in (e.dxf.start.y, e.dxf.end.y)]
    x_lo, x_hi = min(xs), max(xs)
    # the frame: the tall vertical lines (the jambs' outer faces) give its depth
    tall = [e for e in lines if abs(e.dxf.start.x - e.dxf.end.x) < 0.01 and abs(e.dxf.start.y - e.dxf.end.y) > 100]
    if tall:
        fy0 = min(min(e.dxf.start.y, e.dxf.end.y) for e in tall)
        fy1 = max(max(e.dxf.start.y, e.dxf.end.y) for e in tall)
    else:  # (an outset door has no frame: the drawing's own extent)
        fy0, fy1 = min(ys), max(ys)
    cx, cy = (x_lo + x_hi) / 2, (fy0 + fy1) / 2  # frame centre: the wall's centre line
    hinges = [(e.dxf.center.x, e.dxf.center.y) for e in plan if e.dxftype() == "CIRCLE"]
    if not swing:  # a sliding part: only the 180 degree turn (leaf to the -y face, like the LIXIL part)
        d = ezjww.new(paper_size=3)
        msp = d.modelspace()
        cx_, cy_ = (x_lo + x_hi) / 2, (fy0 + fy1) / 2
        # an outset door hangs on the wall face: the open face of its 3-sided black box (the 中方立
        # side) sits on the room-side board face (-65); the leaf then slides on the -y side, like
        # every other sliding part (user 11/8). No turn.
        depth_f = fy1 - fy0
        # a frame deeper than 155 (the 192 of the 3枚連動片引き): the slide side (-y) face stands 12.5 proud
        # of the board face, the other side runs out beyond (the wall is built out there); not centred
        ysh = (-(cfg_board + 12.5) + depth_f / 2) if (not outset and 156 < depth_f < 300) else 0.0
        guide_ids = set()
        y_open = None
        if outset:
            # the horizontal lines that run out past both ends of the part are CAD guides: the board face
            # the part is set on (not part of the part, so they are left out; user 11/8)
            groups = {}
            for e in lines:
                if abs(e.dxf.start.y - e.dxf.end.y) < 0.01 and abs(e.dxf.end.x - e.dxf.start.x) > 20:
                    groups.setdefault(round(e.dxf.start.y, 1), []).append(e)
            cand = [(sum(abs(e.dxf.end.x - e.dxf.start.x) for e in g), y, g) for y, g in groups.items() if len(g) >= 2]
            if cand:
                _t, y_open, g_lines = max(cand, key=lambda c: c[0])
                guide_ids = {id(e) for e in g_lines}
            else:  # (an 入隅 drawing has no guide line: its own 130 wall, two long lines 130 apart, gives the room-side board face)
                longs = sorted({round(e.dxf.start.y, 1) for e in lines if abs(e.dxf.start.y - e.dxf.end.y) < 0.01 and abs(e.dxf.end.x - e.dxf.start.x) > 300})
                pair = [(lo_, hi_) for lo_ in longs for hi_ in longs if abs((hi_ - lo_) - 130.0) < 1.5]
                if pair:
                    y_open = pair[0][0]
            if y_open is None:  # (the top of the end posts)
                vtops = [max(e.dxf.start.y, e.dxf.end.y) for e in lines if abs(e.dxf.start.x - e.dxf.end.x) < 0.01
                         and (abs(e.dxf.start.x - x_lo) < 3 or abs(e.dxf.start.x - x_hi) < 3) and abs(e.dxf.start.y - e.dxf.end.y) > 20]
                y_open = max(vtops) if vtops else (fy0 + fy1) / 2
        board = cfg_board
        slide_h, slide_v = [], []
        for e in plan:
            if id(e) in guide_ids:
                continue
            t = e.dxftype()
            def T2(x, y):
                if outset:
                    return ((x_hi - x) + offset_mm) / S, (y - y_open - board) / S  # mirrored in x only
                xr, yr = rotate180(x, y, cx_, cy_)
                return ((xr - x_lo) + offset_mm) / S, (yr - cy_ + ysh) / S
            if t == "LINE":
                p, q = (e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)
                if math.dist(p, q) >= 0.05:
                    a_, b_ = T2(*p), T2(*q)
                    msp.add_line(a_, b_, jwwattribs=attrs)
                    if abs(a_[1] - b_[1]) < 1e-9:
                        slide_h.append((min(a_[0], b_[0]), max(a_[0], b_[0]), a_[1]))
                    elif abs(a_[0] - b_[0]) < 1e-9:
                        slide_v.append((a_[0], min(a_[1], b_[1]), max(a_[1], b_[1])))
            elif t == "CIRCLE":
                msp.add_circle(T2(e.dxf.center.x, e.dxf.center.y), e.dxf.radius / S, jwwattribs=attrs)
            elif t == "ARC":
                a0, a1 = ((180.0 - e.dxf.end_angle, 180.0 - e.dxf.start_angle) if outset else (e.dxf.start_angle + 180.0, e.dxf.end_angle + 180.0))
                msp.add_arc(T2(e.dxf.center.x, e.dxf.center.y), e.dxf.radius / S, start_angle=a0 % 360.0,
                            sweep_angle=(a1 - a0) % 360.0, jwwattribs=attrs)
        # the sliding lines (user 11/8): from each leaf's end (the right end: the opening is on the left),
        # at the centre of the leaf's thickness, a dotted line out to the frame
        right_in = ((x_hi - x_lo) + offset_mm - (0 if outset else 21.0)) / S
        dot = dict(attrs, pen_style=2)
        leaves_ = []
        for i1, a1 in enumerate(slide_h):
            for a2 in slide_h[i1 + 1:]:
                if (abs(a1[0] - a2[0]) < 3 / S and abs(a1[1] - a2[1]) < 3 / S and abs(a1[2] - a2[2]) < 3 / S
                        and False):
                    pass
                if abs(a1[0] - a2[0]) < 3 / S and abs(a1[1] - a2[1]) < 3 / S and 15 / S <= abs(a1[2] - a2[2]) <= 45 / S                         and 300 / S < (a1[1] - a1[0]) < 0.9 * (x_hi - x_lo) / S:
                    leaves_.append((a1[1], (a1[2] + a2[2]) / 2, a1[0]))
        left_in = (offset_mm + (0 if outset else 21.0)) / S
        for xr_, yc_, xl_leaf in leaves_:
            if right_in - xr_ > 5 / S:
                msp.add_line((xr_, yc_), (right_in, yc_), jwwattribs=dot)
            if both_ways and xl_leaf - left_in > 5 / S:  # a 引違い slides both ways: dotted to the left frame too (user 11/8)
                msp.add_line((xl_leaf, yc_), (left_in, yc_), jwwattribs=dot)
        # the 中方立: a short vertical inside the frame, away from both ends (used by the furred wall)
        xa_, xb_ = offset_mm / S, ((x_hi - x_lo) + offset_mm) / S
        posts = [v for v in slide_v if 50 / S < (v[2] - v[1]) < 120 / S and xa_ + 500 / S < v[0] < xb_ - 300 / S]
        mid_post[0] = round((posts[0][0] - xa_) * S, 1) if (posts and not outset) else None
        if outset and irizumi:
            # 入隅: the 戸先 side frame (the left one) meets a wall at right angles to the door. The wall (130
            # thick) lies against the frame's outer edge and runs through the door's wall (a T, to 450 beyond each face): its two faces and its
            # centre line, in blue (pen colour 3) (user 11/8)
            xl_ = offset_mm / S
            blue = dict(attrs, pen_color=3)
            for dxm in (-130.0, -65.0, 0.0):
                msp.add_line((xl_ + dxm / S, (cfg_board + 450.0) / S), (xl_ + dxm / S, (-cfg_board - 450.0) / S), jwwattribs=blue)
            # the door's own wall (blue): it starts at the perpendicular wall's face and does not cross it
            xr_end = (offset_mm + (x_hi - x_lo)) / S
            for yy in (-cfg_board, 0.0, cfg_board):
                msp.add_line((xl_, yy / S), (xr_end + 130.0 / S, yy / S), jwwattribs=blue)
        if surround and not outset:
            # the wall around the door, with the furring on the 中方立 side (+y): board faces (finish colour),
            # column-face lines 52.5 (structure colour); over the frame the +y board face moves out so the
            # frame stands 12.5 proud of it, with returns; the 52.5 line stays, so the two are drawn apart
            bay = std_bay(surround, uw) / S
            lay_ = dict(attrs)
            fin = dict(attrs, pen_color=surround_colors[0])
            st = dict(attrs, pen_color=surround_colors[1])
            x_a, x_b = offset_mm / S, (offset_mm + (x_hi - x_lo)) / S
            new_face = (-(cfg_board + 12.5) + depth_f - 12.5) / S
            b = cfg_board / S
            far = new_face
            xp_ = x_a + ((mid_post[0] + 24.0) / S if mid_post[0] else (x_b - x_a))  # the post's right face (24 wide): stop before it
            # every wall line ends against the door (the frame end or the 中方立) or at a column line
            for (xa, xb) in ((0.0, x_a), (x_b, bay)):
                for yy in (-52.5 / S, 52.5 / S):
                    msp.add_line((xa, yy), (xb, yy), jwwattribs=st)
                msp.add_line((xa, -b), (xb, -b), jwwattribs=fin)                 # slide side: the board face
            for (xa, xb) in ((0.0, x_a), (xp_, bay)):                            # 中方立 side: the built-out board face
                if xb - xa > 0.01:
                    msp.add_line((xa, far), (xb, far), jwwattribs=fin)
                    msp.add_line((xa, far - 12.5 / S), (xb, far - 12.5 / S), jwwattribs=st)  # the board's thickness (12.5), behind the surface: toward the centre
            # the board surface between the 中方立 and the right frame: a plain surface line (no thickness line)
            if x_b - xp_ > 0.01:
                msp.add_line((xp_, b), (x_b, b), jwwattribs=fin)
        d.saveas(str(out_path))
        return {"frame_depth": fy1 - fy0, "outer_w": x_hi - x_lo, "swing_r": [], "flush_step": 0.0}
    # the door leaves: long horizontal lines inside the frame (a 親子ドア has two leaves)
    hor = [e for e in lines if abs(e.dxf.start.y - e.dxf.end.y) < 0.01 and abs(e.dxf.end.x - e.dxf.start.x) > 150]
    leaf_y = sorted({round(e.dxf.start.y, 1) for e in hor})
    # the swing side = the leaf face nearer to a frame face (a small step): the frame and the door
    # are nearly flush there (user 11/7: do not draw the arc on the side with big steps)
    step_lo, step_hi = leaf_y[0] - fy0, fy1 - leaf_y[-1]
    cy_flip = step_lo > step_hi  # the flush face is at the top: only the x turn is wanted
    face = leaf_y[-1] if cy_flip else leaf_y[0]
    d = ezjww.new(paper_size=3)
    msp = d.modelspace()
    ox = offset_mm

    def T(x, y):
        # 180 degree turn about the frame centre: hinge to the left, flush face to +y
        xr, yr = rotate180(x, y, cx, cy)
        if cy_flip:
            yr = 2 * cy - yr
        return ((xr - x_lo) + ox) / S, (yr - cy) / S

    for e in plan:
        t = e.dxftype()
        if t == "LINE":
            p, q = (e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)
            if math.dist(p, q) < 0.05:
                continue
            msp.add_line(T(*p), T(*q), jwwattribs=attrs)
        elif t == "CIRCLE":
            msp.add_circle(T(e.dxf.center.x, e.dxf.center.y), e.dxf.radius / S, jwwattribs=attrs)
        elif t == "ARC":
            c = T(e.dxf.center.x, e.dxf.center.y)
            a0, a1 = e.dxf.start_angle + 180.0, e.dxf.end_angle + 180.0
            if cy_flip:
                a0, a1 = -a1, -a0
            msp.add_arc(c, e.dxf.radius / S, start_angle=a0 % 360.0, sweep_angle=(a1 - a0) % 360.0, jwwattribs=attrs)
    # the swings: each leaf (its face on the flush side) turns about the hinge circle at one end;
    # the arc starts at the leaf's free end, on the leaf face; the open leaf line runs from the
    # circle's centre to the arc's tip
    segs = sorted((min(e.dxf.start.x, e.dxf.end.x), max(e.dxf.start.x, e.dxf.end.x))
                  for e in hor if abs(e.dxf.start.y - face) < 0.2)
    radii = []
    for sx0, sx1 in segs:
        hx, hy = min(hinges, key=lambda h: min(abs(h[0] - sx0), abs(h[0] - sx1)))
        free_x = sx1 if abs(hx - sx0) < abs(hx - sx1) else sx0
        r = abs(free_x - hx)
        axis = T(hx, hy)
        start_pt = T(free_x, face)
        centre = (axis[0], start_pt[1])  # level with the leaf face, above the hinge circle
        # the arc turns from the free end up to the line over the hinge: towards the hinge side
        toward_left = axis[0] < start_pt[0]
        msp.add_arc(centre, r / S, start_angle=0 if toward_left else 90, sweep_angle=90, jwwattribs=attrs)
        msp.add_line(axis, (axis[0], centre[1] + r / S), jwwattribs=attrs)
        radii.append(r)
    d.saveas(str(out_path))
    return {"frame_depth": fy1 - fy0, "outer_w": x_hi - x_lo, "swing_r": radii, "flush_step": min(step_lo, step_hi)}


# (library kind, CSV type name, op, needs the 155 fixed frame)
TYPES = [
    ("片開き", "片開きドア", "swing", True),
    ("親子ドア", "親子ドア", "swing", True),
    ("片引戸", "片引き", "sliding", True),
    ("片引戸", "表示錠付き片引き", "sliding", True),
    ("引違い戸", "引違い", "sliding", True),
    ("引違い戸", "３枚連動引違い", "sliding", True),
    ("引違い戸", "４枚引違い", "sliding", True),
    ("2枚連動片引戸", "２枚連動片引き", "sliding", True),
    ("3枚連動片引戸", "３枚連動片引き", "sliding", True),
    ("2枚両引き戸", "２枚両引き", "sliding", True),
    ("アウトセット片引戸", "壁付けアウトセット片引き", "sliding", False),
    ("アウトセット片引戸入隅", "壁付けアウトセット片引き・入隅", "sliding", False),
    ("アウトセット表示錠付き片引戸", "壁付けアウトセット表示錠付き片引き", "sliding", False),
    ("アウトセット表示錠付き片引戸入隅", "壁付けアウトセット表示錠付き片引き・入隅", "sliding", False),
    ("アウトセット両引き戸", "壁付けアウトセット両引き", "sliding", False),
    ("アウトセット天井埋込片引戸", "天井埋込アウトセット片引き", "sliding", False),
    ("アウトセット天井埋込片引戸入隅", "天井埋込アウトセット片引き・入隅", "sliding", False),
    ("アウトセット天井埋込表示錠付き片引戸", "天井埋込アウトセット表示錠付き片引き", "sliding", False),
    ("アウトセット天井埋込表示錠付き片引戸入隅", "天井埋込アウトセット表示錠付き片引き・入隅", "sliding", False),
    ("アウトセット天井埋込両引き戸", "天井埋込アウトセット両引き", "sliding", False),
]


def _clean_cluster(ents, w):
    """The plan file holds other things apart from the plan (notes, axis lines, a small cross guide):
    keep the cluster of entities that best fits the frame's width, and drop the cross guide
    (two 75 mm lines, horizontal and vertical, crossing at one point; user 11/8)."""
    def xr(e):
        if e.dxftype() == "LINE":
            return sorted((e.dxf.start.x, e.dxf.end.x))
        return [e.dxf.center.x - e.dxf.radius, e.dxf.center.x + e.dxf.radius]
    ents = [e for e in ents if not (e.dxftype() == "LINE" and math.dist((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)) > 1.25 * w)]
    iv = sorted(xr(e) for e in ents)
    groups = []
    for lo_, hi_ in iv:
        if groups and lo_ - groups[-1][1] < 80:
            groups[-1][1] = max(groups[-1][1], hi_)
        else:
            groups.append([lo_, hi_])
    gx0, gx1 = min(groups, key=lambda g: abs((g[1] - g[0]) - w))
    ents = [e for e in ents if gx0 - 1 <= xr(e)[0] and xr(e)[1] <= gx1 + 1]
    hs = [e for e in ents if e.dxftype() == "LINE" and abs(e.dxf.start.y - e.dxf.end.y) < 0.01 and abs(abs(e.dxf.end.x - e.dxf.start.x) - 75) < 1]
    vs = [e for e in ents if e.dxftype() == "LINE" and abs(e.dxf.start.x - e.dxf.end.x) < 0.01 and abs(abs(e.dxf.end.y - e.dxf.start.y) - 75) < 1]
    drop = set()
    for h_ in hs:
        for v_ in vs:
            if abs((h_.dxf.start.x + h_.dxf.end.x) / 2 - v_.dxf.start.x) < 1 and abs((v_.dxf.start.y + v_.dxf.end.y) / 2 - h_.dxf.start.y) < 1:
                drop |= {id(h_), id(v_)}
    return [e for e in ents if id(e) not in drop]


def build_storage(ents, w, op, offset_mm, attrs, out_path, proud=12.5, board=65.0):
    """収納用建具 (折戸 / 両開き) as a finished part. The frame's face on the opening side stands
    proud of the board face by 12.5 (not centred on the wall centre; user 11/8): on the opening side
    (+y) the frame face is at +board+proud. 折戸: the 開閉線 V starts on the chamfered face of the
    panels (the face where each panel joint shows the V notch: the chamfers keep fingers from being
    caught, the room side), half open at 60 degrees, one pair hinged at each jamb for four panels.
    両開き: arcs about the hinge ends on the leaf face on the opening side."""
    lines = [e for e in ents if e.dxftype() == "LINE"]
    tall = [e for e in lines if abs(e.dxf.start.x - e.dxf.end.x) < 0.01 and abs(e.dxf.start.y - e.dxf.end.y) > 40]
    x_lo = min(min(e.dxf.start.x, e.dxf.end.x) for e in tall)
    x_hi = max(max(e.dxf.start.x, e.dxf.end.x) for e in tall)
    y_min = min(min(e.dxf.start.y, e.dxf.end.y) for e in tall)
    y_max = max(max(e.dxf.start.y, e.dxf.end.y) for e in tall)
    hor = [e for e in lines if abs(e.dxf.start.y - e.dxf.end.y) < 0.01 and abs(e.dxf.end.x - e.dxf.start.x) > 100
           and y_min + 0.5 < e.dxf.start.y < y_max - 0.5]
    ys = sorted({round(e.dxf.start.y, 1) for e in hor})
    if op == "folding":
        diag = [(e.dxf.start.y + e.dxf.end.y) / 2 for e in lines if abs(e.dxf.start.x - e.dxf.end.x) > 0.01 and abs(e.dxf.start.y - e.dxf.end.y) > 0.01
                and 5 < math.dist((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)) < 30]
        panel_ys = [y for y in ys if y_min + 10 < y < y_max - 10]
        face_y = min(panel_ys, key=lambda y: abs(y - (sum(diag) / len(diag)))) if diag else panel_ys[0]
        open_low = face_y < (y_min + y_max) / 2
    else:
        face_y = ys[0] if ys else (y_min + y_max) / 2
        open_low = (face_y - y_min) < (y_max - face_y)
        face_y = ys[0] if open_low else ys[-1]
    d = ezjww.new(paper_size=3)
    msp = d.modelspace()

    def T(x, y):
        yy = (board + proud + (y_min - y)) if open_low else (board + proud - (y_max - y))
        return ((x - x_lo) + offset_mm) / S, yy / S

    for e in ents:
        t = e.dxftype()
        if t == "LINE":
            p, q = (e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)
            if math.dist(p, q) >= 0.05:
                msp.add_line(T(*p), T(*q), jwwattribs=attrs)
        elif t == "CIRCLE":
            msp.add_circle(T(e.dxf.center.x, e.dxf.center.y), e.dxf.radius / S, jwwattribs=attrs)
        else:
            a0, a1 = e.dxf.start_angle, e.dxf.end_angle
            if open_low:
                a0, a1 = -a1, -a0
            msp.add_arc(T(e.dxf.center.x, e.dxf.center.y), e.dxf.radius / S, start_angle=a0 % 360.0, sweep_angle=(a1 - a0) % 360.0, jwwattribs=attrs)
    fy = T(0, face_y)[1]
    segs = sorted((min(e.dxf.start.x, e.dxf.end.x), max(e.dxf.start.x, e.dxf.end.x))
                  for e in hor if abs(e.dxf.start.y - face_y) < 0.2)
    if op == "folding":
        back = [e for e in hor if 5 < abs(e.dxf.start.y - face_y) < 40]
        n = max(2, len({round(min(e.dxf.start.x, e.dxf.end.x)) for e in back}))
        x0 = ((segs[0][0] - x_lo) + offset_mm) / S
        x1 = ((segs[-1][1] - x_lo) + offset_mm) / S
        pw = (x1 - x0) / n                      # one panel's width
        a = math.radians(60.0)

        def zigzag(xh, sg, k):
            """k V's hung from a jamb (each V = 2 panels, half open at 60 degrees)."""
            px, pts = xh, [(xh, fy)]
            for _ in range(k):
                pts.append((px + sg * pw * math.cos(a), fy + pw * math.sin(a)))
                px += sg * 2 * pw * math.cos(a)
                pts.append((px, fy))
            for u, v in zip(pts, pts[1:]):
                msp.add_line(u, v, jwwattribs=attrs)
        # 2 panels: one V at the left; 4: one at each jamb; 6: two V's at the left, one at the right;
        # 8: two V's at each jamb (user 11/8)
        if n <= 2:
            zigzag(x0, 1, 1)
        elif n <= 4:
            zigzag(x0, 1, 1)
            zigzag(x1, -1, 1)
        elif n <= 6:
            zigzag(x0, 1, 2)
            zigzag(x1, -1, 1)
        else:
            zigzag(x0, 1, 2)
            zigzag(x1, -1, 2)
    else:
        for sx0, sx1 in segs:
            hx_left = sx0 - x_lo < (x_hi - x_lo) / 2
            hx = ((sx0 if hx_left else sx1) - x_lo + offset_mm) / S
            r = (sx1 - sx0) / S
            msp.add_arc((hx, fy), r, start_angle=0 if hx_left else 90, sweep_angle=90, jwwattribs=attrs)
            msp.add_line((hx, fy), (hx, fy + r), jwwattribs=attrs)
    d.saveas(str(out_path))


def storage_parts(spec, wl):
    """収納用建具 (折れ戸 / 開き扉): the plan drawing is a file of its own (the 平面図 column).
    折れ戸 -> 折戸 (all widths, 4方枠の固定枠), 開き扉 -> 両開き (3方枠の固定枠). Heights are kept."""
    sdir = root / "収納用建具_dxf"
    if not sdir.exists():
        return
    import ezdxf
    csvs = list(sdir.rglob("*.csv"))
    rows = [r for r in csv.reader(open(csvs[0], encoding="cp932"))][1:]
    files = {f.stem.upper(): f for f in sdir.rglob("*.DXF")}
    files.update({f.stem.upper(): f for f in sdir.rglob("*.dxf")})
    ready = root / "ready"
    ready.mkdir(exist_ok=True)
    attrs = {"layer": wl["layer"], "pen_color": wl["structure_color"], "pen_style": 1}
    plans = [("折れ戸", "折戸", "folding", "４方枠（固定枠）"), ("開き扉", "両開き", "double_swing", "３方枠（固定枠）")]
    for csv_type, word, op, frame in plans:
        sel = [r for r in rows if len(r) > 12 and csv_type in r[3] and r[4].startswith("ＶＥＲＩＴＩＳシリーズ")
               and r[5] == "フラットタイプ" and r[6] == frame]
        by_w = {}
        for r in sel:
            m = re.match(r"枠外寸法.(\d+)×(\d+)", zen2han(r[9]))
            if m and (r[12].upper() in files):
                by_w.setdefault(int(m.group(1)), []).append((int(m.group(2)), r))
        for w, items in sorted(by_w.items()):
            heights = sorted({h for h, _ in items})
            r = next(r for h, r in items if h == heights[0])
            doc = ezdxf.readfile(str(files[r[12].upper()]))
            ents = [e for e in doc.modelspace() if e.dxftype() in ("LINE", "ARC", "CIRCLE")]
            if not ents:
                continue
            ents = _clean_cluster(ents, w)
            off = std_offset(word, w)
            out = ready / f"{word}_W{w}.jww"
            try:
                build_storage(ents, w, op, off, attrs, out)
            except Exception as ex:
                print("失敗", word, w, ex)
                continue
            spec["parts"].append({"maker": "パナソニック", "series": "ベリティス", "kind": word, "op": "swing", "width": w,
                                  "note": f"スタンダード2026 収納用建具 {csv_type} {frame} 高さ{'/'.join(map(str, heights))}",
                                  "file": f"ready/{out.name}", "ready": True, "offset_mm": off,
                                  "heights_mm": heights, "model": r[12], "frame": frame, "csv_type": csv_type})
            print(f"{word}（{csv_type}） W{w}: 高さ{heights}")


def main():
    cfg = json.load(open(Path(__file__).resolve().parent.parent / "05_設定" / "default.json", encoding="utf-8"))
    wl = cfg["layers"]["wall"]
    attrs = {"layer": wl["layer"], "pen_color": wl["structure_color"], "pen_style": 1}  # one layer, one colour
    rows = read_rows()
    ready = root / "ready"
    ready.mkdir(exist_ok=True)
    spec = {"_comment": "パナソニック ベリティス（スタンダード2026）の部品。build_panasonic_parts.py が作る。"
                        "ready=できあがり部品（build_parts_library はそのまま使う）。heights_mm＝選べる建具の高さ", "sheet_scale": 50, "parts": []}
    for word, tname, op, need_frame in TYPES:
        fr = FRAME_OF.get(tname, FRAME)
        sel = [r for r in rows if r["type"] == tname and (not need_frame or fr in r["full_frame"])]
        for uw in sorted({r["uw"] for r in sel}):
            cand = [r for r in sel if r["uw"] == uw]
            heights = sorted({r["uh"] for r in cand})
            base = next(r for r in cand if r["uh"] == min(heights))
            dxf = find_dxf(base["code"])
            if not dxf:
                print("no dxf", word, uw, base["code"])
                continue
            off = std_offset(word, uw)
            out = ready / f"{word}_{tname}_W{uw}.jww"
            try:
                info = build_swing(dxf, uw, out, off, attrs, swing=(op == "swing"), outset=word.startswith("アウトセット"), irizumi=("入隅" in word), both_ways=(word == "引違い戸"))
            except Exception as ex:  # one odd drawing must not stop the others
                print("失敗", word, tname, uw, ex)
                continue
            clears = []
            for hh in heights:
                # 有効開口: the maker's table is not found yet (a measured 738 for UW780 was wrong; user 11/8).
                # Only the leaf size a shop listing gives is kept (片開き, UH2035: width UW-56, height 1995)
                leaf = {"戸の幅": uw - 56, "戸の高さ": 1995, "出典": "販売店ページの記載（メーカー表ではない）"} if (tname == "片開きドア" and hh == 2035) else None
                clears.append({"高さUH": hh, "有効開口幅": None, "有効開口高さ": None, "戸の寸法": leaf, "備考": "有効開口はメーカーの表で確認（未確認）"})
            frame = ("固定枠(3方枠)" + ("192" if tname in FRAME_OF else "155")) if need_frame else "枠なし（壁付け）"
            if tname in FRAME_OF:  # the 192 frame: also a part with the wall around it (furred), to look at
                out2 = ready / f"{word}_ふかし壁付き_W{uw}.jww"
                wl2 = cfg["layers"]["wall"]
                surround_colors_ = (wl2["finish_color"], wl2["structure_color"])
                globals()["surround_colors"] = surround_colors_
                build_swing(dxf, uw, out2, off, attrs, swing=False, surround=word)
                spec["parts"].append({"maker": "パナソニック", "series": "ベリティス", "kind": word + "ふかし壁付き", "op": op, "width": uw,
                                      "note": f"3枚連動片引戸（192の枠）に、ふかした壁（中方立側）を付けた版。石膏ボード面と52.5の線を含む",
                                      "file": f"ready/{out2.name}", "ready": True, "offset_mm": off, "heights_mm": heights,
                                      "model": base["code"], "frame": "固定枠(3方枠)192", "csv_type": tname})
            spec["parts"].append({"maker": "パナソニック", "series": "ベリティス", "kind": word, "op": op, "width": uw,
                                  "note": f"スタンダード2026 {tname} {frame} 高さ{'/'.join(map(str, heights))}",
                                  "file": f"ready/{out.name}", "ready": True, "offset_mm": off,
                                  "heights_mm": heights, "clear_openings": clears, "mid_post_from_frame_left_mm": (mid_post[0] if tname in FRAME_OF else None), "model": base["code"], "frame": frame, "csv_type": tname})
            print(f"{word}（{tname}） UW{uw}: 枠の奥行き{info['frame_depth']:.0f} 外幅{info['outer_w']:.0f} 高さ{heights}")
    storage_parts(spec, wl)
    json.dump(spec, open(root / "parts_spec.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
