"""基礎伏図（布基礎）を作る。布の位置は、ユーザーが指示するまでの仮として、1F の壁の判定（<階>_walls.json）と同じ線を使う（入力は「布の線」のリスト）。

usage: make_foundation.py <1F_walls.json> <出力.jww> [基礎.json] [--plan 平面詳細図.jww]（--plan があれば、その柱を避けてアンカーを置く）

- 立上り：布の芯の両側に、立上りの幅の半分ずつ（外形線2本）。布の端は、角・T・自由な端のどこでも、幅の半分だけ飛び出す（立上りも底盤も）。布は全部の線に通す（提案の位置。
  ユーザーが位置を指示するまでの仮、11/15）
- 底盤（布のフーチング）：布の芯の両側に、底盤の幅の半分ずつ（実線。user 11/15）
- 通り芯（布芯）と、芯々の寸法、通り芯の記号
- 値は 05_設定/基礎.json（立上り幅150・底盤幅600・底盤厚150・立上り高450・根入れ600・布基礎、全部仮）
- アンカーボルト：基礎伏図の中の別レイヤ（7番「アンカー」）。置き方は布の線だけから決める（図面にない開口・柱位置などは予測しない。ユーザーの指示が出てから）。--no-anchor で描かない
作図の環境（色・線幅・縮尺）は平面詳細図と同じ（jww_env）。
"""
import json
import os
import shutil
import sys
from pathlib import Path

from shapely.geometry import LineString
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ezjww  # noqa: E402
import jww_env  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = HERE.parent
S = 50.0  # 縮尺 1/50


def rings(geom):
    polys = getattr(geom, "geoms", [geom])
    for p in polys:
        yield list(p.exterior.coords)
        for h in p.interiors:
            yield list(h.coords)


def read_columns(plan_jww, meta_json, size=105):
    """平面詳細図の柱（105角・対角線2本）の中心を読む。model mm = paper mm × 縮尺 + center"""
    meta = json.load(open(meta_json, encoding="utf-8"))
    cx, cy = meta["center_mm"]
    sc = meta["scale"]
    d = ezjww.read_document(plan_jww)
    diag = size * 2 ** 0.5 / sc
    seen = {}
    for e in d["entities"]:
        if e["type"] != "LINE":
            continue
        x1, y1, x2, y2 = e["start_x"], e["start_y"], e["end_x"], e["end_y"]
        if abs(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 - diag) < 0.02:
            m = (round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2))
            seen[m] = seen.get(m, 0) + 1
    return [(round(x * sc + cx, 1), round(y * sc + cy, 1)) for (x, y), n in seen.items() if n >= 2]


def grid_labels(vals, unit=910):
    """通り芯の番号：先頭を1とし、unit（910）ごとに1ずつ。unit に満たない（半分の）通り芯は .5。
    次の910に布がなければ番号は飛ばす。全体が unit の整数倍でなく端が455で終わる時は、左から数えた最後の通り芯 a と、
    右から数えた最初の通り芯 b が455はなれている所（継ぎ目、真ん中に近い所）を探し、その455を番号1つ分にする。
    見つからなければ警告を出し、左から数える。"""
    first, last = vals[0], vals[-1]
    half = unit / 2
    seam = None
    if (last - first) % unit > 1:
        best = None
        for a, b in zip(vals, vals[1:]):
            if abs((a - first) % unit) < 1 and abs((last - b) % unit) < 1 and abs((b - a) % unit - half) < 1:
                d = abs((a + b) / 2 - (first + last) / 2)
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best:
            seam = best[1:]
        else:
            print("警告：全体の寸法が910の倍数でなく、455を詰められる継ぎ目の通り芯が見つかりません（左から数えます）")
    out = []
    for v in vals:
        if seam and v >= seam[1] - 1:
            n0, ref, na = (seam[0] - first) // unit + 2, seam[1], v - seam[1]
        else:
            n0, ref, na = 1, first, v - first
        k = int(na // unit + 1e-6)
        rem = na - k * unit
        n = n0 + k
        out.append(f"{n}" if rem < 1 else f"{n}.5")
    return out


def anchor_points(walls, ac, cols=()):
    """アンカーの位置（案）：各布の芯の上に、両端から end_mm の所、間が max_spacing_mm を超えたら等間隔に足す。
    短い布（長さ short_mm 以下）は真ん中に1本。柱（平面詳細図）の中心から col_clear_mm 以内は避け、布に沿って
    近い空きへずらす（柱は図面にあるので避けてよい。user 11/15）。開口などの予測はしない。"""
    pts = []
    clear = ac.get("col_clear_mm", 0)
    for p, q in walls:
        L = abs(q[0] - p[0]) + abs(q[1] - p[1])
        if L < 1:
            continue
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
        e = ac["end_mm"]
        ts = [L / 2] if L <= ac["short_mm"] else [e, L - e]
        i = 0
        while i < len(ts) - 1:
            gap = ts[i + 1] - ts[i]
            if gap > ac["max_spacing_mm"]:
                n = int(-(-gap // ac["max_spacing_mm"])) - 1  # 間に足す本数
                ts[i + 1:i + 1] = [ts[i] + gap * k / (n + 1) for k in range(1, n + 1)]
            i += 1
        def at(t):
            return (p[0] + ux * t, p[1] + uy * t)
        def free(t):
            x, y = at(t)
            return all(max(abs(x - c[0]), abs(y - c[1])) >= clear for c in cols)
        for t in ts:
            if cols and not free(t):
                for k in range(1, 40):  # 25ずつ、近い順に両側を探す
                    ok = [t + sg * 25 * k for sg in (1, -1) if 0 < t + sg * 25 * k < L and free(t + sg * 25 * k)]
                    if ok:
                        t = ok[0]
                        break
            x, y = at(t)
            pts.append((round(x, 1), round(y, 1)))
    # 同じ場所（角・T で重なったもの）は1本に
    out = []
    for a in pts:
        if all(abs(a[0] - b[0]) + abs(a[1] - b[1]) > ac["merge_mm"] for b in out):
            out.append(a)
    return out


def draw_anchors(walls, ac, msp, P, cols=()):
    r = ac["radius_mm"]
    attr = {"layer": ac["layer"], "pen_color": ac["color"], "pen_style": ac["style"]}
    pts = anchor_points(walls, ac, cols)
    for x, y in pts:
        msp.add_circle(P(x, y), r / S, jwwattribs=attr)
        msp.add_line(P(x - r * 1.6, y), P(x + r * 1.6, y), jwwattribs=attr)
        msp.add_line(P(x, y - r * 1.6), P(x, y + r * 1.6), jwwattribs=attr)
    return len(pts)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_anchor = "--no-anchor" not in sys.argv  # アンカー（別レイヤ7）。既定で描く
    plan = None
    if "--plan" in sys.argv:
        plan = sys.argv[sys.argv.index("--plan") + 1]
        args = [a for a in args if a != plan]
    walls_path, out = args[0], args[1]
    cfg_path = args[2] if len(args) > 2 else str(ROOT / "05_設定" / "基礎.json")
    fc = json.load(open(cfg_path, encoding="utf-8"))
    det = json.load(open(walls_path, encoding="utf-8"))
    walls = [(tuple(w["p1"]), tuple(w["p2"])) for w in det["walls"]]
    xs = [p[0] for w in walls for p in w]
    ys = [p[1] for w in walls for p in w]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    A, B = fc["tachiagari_width"], fc["teiban_width"]
    lay = fc["layers"]

    d = ezjww.new(paper_size=3)
    jww_env.apply_env(d)
    g = d.options["layer_groups"][0]
    g["scale"] = S
    if "anchor" in fc:
        g["layers"][fc["anchor"]["layer"]]["name"] = fc["anchor"].get("name", "アンカー")
    msp = d.modelspace()

    def P(x, y):
        return ((x - cx) / S, (y - cy) / S)

    def line(p, q, key):
        a = lay[key]
        msp.add_line(P(*p), P(*q), jwwattribs={"layer": a["layer"], "pen_color": a["color"], "pen_style": a["style"]})

    def dot(pt, key="dim"):
        """寸法線と補助線の交点は、斜線ではなく点（Jw_cad の点）"""
        a = lay[key]
        msp.add_point(P(*pt), jwwattribs={"layer": a["layer"], "pen_color": a["color"], "pen_style": 1})

    def text(s, x, y, key, size=None, angle=0.0, length=None):
        a = lay[key]
        size = size or fc["dim"]["text_size_mm"]
        n = max(1, len(s)) * size * 0.9 * S if length is None else length
        sx, sy = P(x, y)
        import math
        ex, ey = sx + math.cos(math.radians(angle)) * n / S, sy + math.sin(math.radians(angle)) * n / S
        msp.add_text(s, (sx, sy), (ex, ey), size_x=size * 0.9, size_y=size, angle=angle,
                     jwwattribs={"layer": a["layer"], "pen_color": a["color"], "pen_style": 1})

    # 底盤と立上り（布の芯から両側に半分ずつ。端（角・T・自由な端）は幅の半分だけ飛び出す（square cap））
    for width, key in ((B, "teiban"), (A, "tachiagari")):
        shape = unary_union([LineString(w).buffer(width / 2, cap_style=3, join_style=2) for w in walls])
        for r in rings(shape):
            for p, q in zip(r, r[1:]):
                if abs(p[0] - q[0]) + abs(p[1] - q[1]) > 0.5:
                    line((round(p[0], 1), round(p[1], 1)), (round(q[0], 1), round(q[1], 1)), key)
    # 通り芯（布芯）
    for p, q in walls:
        line(p, q, "center")
    if want_anchor:
        cols = read_columns(plan, walls_path.replace("_walls.json", "_meta.json")) if plan else []
        n_anchor = draw_anchors(walls, fc["anchor"], msp, P, cols)
        print(f"柱 {len(cols)} 本を避ける")
        print(f"アンカー: {n_anchor} 本（別レイヤ {fc['anchor']['layer']}）")
    # 寸法：4面に書く。同じ布の間を2回拾わない（user 11/15）。X方向の区間は、真ん中から左半分を下・右半分を上、
    # Y方向の区間は、真ん中から下半分を左・上半分を右に振り分ける。全体の寸法は下と左に1本ずつ。
    # 通り芯＝布が通っている線（縦の布の x、横の布の y）。布の端だけの位置は通り芯にしない
    gx = sorted({round(w[0][0], 1) for w in walls if abs(w[0][0] - w[1][0]) < 1})
    gy = sorted({round(w[0][1], 1) for w in walls if abs(w[0][1] - w[1][1]) < 1})
    off = fc["dim"]["offset_mm"]
    minl = fc["dim"]["min_text_mm"]
    r = fc["dim"]["bubble_radius_mm"]
    ext = B / 2 + 100  # 補助線の始まり（底盤の外から100あけ）
    base = {"bottom": min(ys) - B / 2 - off, "left": min(xs) - B / 2 - off,
            "top": max(ys) + B / 2 + off, "right": max(xs) + B / 2 + off}

    def dim_chain(vals, side, whole=False, lvl=0):
        """vals の隣どうしの区間を、side（bottom/top/left/right）の寸法線に書く。補助線・交点の点・数字"""
        if len(vals) < 2:
            return
        horizontal = side in ("bottom", "top")
        sign = 1 if side in ("top", "right") else -1
        b = base[side] + sign * 600 * lvl if whole else base[side]
        edge = {"bottom": min(ys), "left": min(xs), "top": max(ys), "right": max(xs)}[side] + sign * ext
        lo, hi = vals[0], vals[-1]
        if horizontal:
            line((lo, b), (hi, b), "dim")
        else:
            line((b, lo), (b, hi), "dim")
        for t in vals:
            if horizontal:
                line((t, edge), (t, b), "dim")
                dot((t, b))
            else:
                line((edge, t), (b, t), "dim")
                dot((b, t))
        for a_, b_ in zip(vals, vals[1:]):
            n = b_ - a_
            if n < minl:
                continue
            s = f"{n:,.0f}"
            if horizontal:
                text(s, (a_ + b_) / 2 - len(s) * 55, b + (50 if sign > 0 else 50), "dim")
            else:
                text(s, b - 50, (a_ + b_) / 2 - len(s) * 55, "dim", angle=90.0)

    # 寸法線：4面に書く（user 11/15）。図の真ん中に縦の軸と横の軸をとり、布がどちら側にあるかで、近い側の寸法線に書く。
    #   Y方向（縦の寸法）：横の布が、真ん中の縦の軸より左にあれば左の寸法線、右にあれば右の寸法線。
    #   X方向（横の寸法）：縦の布が、真ん中の横の軸より下にあれば下の寸法線、上にあれば上の寸法線。
    # どの寸法線にも両端の通り芯（最大寸法の端）を入れるので、区間を足すと最大寸法と合う（途中でやめない）。
    # 最大寸法は X を上、Y を左に1本ずつだけ（固定）。
    midx, midy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    def pick(vals, runs):
        return sorted({vals[0], vals[-1], *runs})

    h_left = [round(p[1], 1) for p, q in walls if abs(p[1] - q[1]) < 1 and (p[0] + q[0]) / 2 <= midx]
    h_right = [round(p[1], 1) for p, q in walls if abs(p[1] - q[1]) < 1 and (p[0] + q[0]) / 2 > midx]
    v_low = [round(p[0], 1) for p, q in walls if abs(p[0] - q[0]) < 1 and (p[1] + q[1]) / 2 <= midy]
    v_up = [round(p[0], 1) for p, q in walls if abs(p[0] - q[0]) < 1 and (p[1] + q[1]) / 2 > midy]
    dim_chain(pick(gx, v_low), "bottom")
    dim_chain(pick(gx, v_up), "top")
    dim_chain(pick(gy, h_left), "left")
    dim_chain(pick(gy, h_right), "right")
    dim_chain([gx[0], gx[-1]], "top", whole=True, lvl=1)
    dim_chain([gy[0], gy[-1]], "left", whole=True, lvl=1)

    # 通り芯の記号：910ごとに番号を振る（910に満たない所は .5）。次の910に布がなければ番号を飛ばす。
    labels_x = grid_labels(gx, fc["dim"].get("unit_mm", 910))
    labels_y = grid_labels(gy, fc["dim"].get("unit_mm", 910))
    ts = fc["dim"].get("bubble_text_mm", 1.5)

    def bubble(label, cx_, cy_):
        msp.add_circle(P(cx_, cy_), r / S, jwwattribs={"layer": lay["dim"]["layer"], "pen_color": lay["dim"]["color"], "pen_style": 1})
        w = len(label) * ts * 0.9 * S
        text(label, cx_ - w / 2, cy_ - 70, "dim", size=ts, length=w)

    yt = base["top"] + 600 + 700 + r
    xr = base["right"] + 700 + r
    for v, lb in zip(gx, labels_x):
        line((v, base["top"] + 600 + 250), (v, yt - r), "center")
        bubble("X" + lb, v, yt)
    for v, lb in zip(gy, labels_y):
        line((base["right"] + 250, v), (xr - r, v), "center")
        bubble("Y" + lb, xr, v)
    # 表題と、基礎の寸法（仮）
    ty = min(ys) - B / 2 - 900 - 560
    text("基礎伏図（布基礎）　縮尺 1/50　※寸法はすべて仮", min(xs), ty, "dim", size=3.5, length=14000)
    note = (f"立上り幅{A}／底盤幅{B}／底盤厚{fc['teiban_thickness']}／"
            f"立上り高{fc['tachiagari_height']}（GL上）／根入れ{fc['neire']}（GL下）")
    text(note, min(xs), ty - 400, "dim", size=2.4, length=14000)
    tmp = Path(os.environ.get("TEMP", ".")) / "jw_foundation_tmp.jww"
    d.saveas(str(tmp))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp, out)
    print(f"基礎伏図: 布 {len(walls)} 本、立上り{A}・底盤{B} → {out}")


if __name__ == "__main__":
    main()
