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
    # 寸法：下と左に、芯々の寸法。上と右に通り芯の記号
    # 通り芯＝布が通っている線（縦の布の x、横の布の y）。布の端だけの位置は通り芯にしない
    gx = sorted({round(w[0][0], 1) for w in walls if abs(w[0][0] - w[1][0]) < 1})
    gy = sorted({round(w[0][1], 1) for w in walls if abs(w[0][1] - w[1][1]) < 1})
    off = fc["dim"]["offset_mm"]
    minl = fc["dim"]["min_text_mm"]
    y0 = min(ys) - B / 2 - off
    x0 = min(xs) - B / 2 - off

    def chain(vals, horizontal, base):
        """寸法線：補助線（布の芯の位置から、寸法線を少し越えるまで）・45°の斜線・寸法の数字（線の上／左）"""
        edge = (min(ys) if horizontal else min(xs)) - B / 2 - 100  # 補助線の始まり（底盤の外から100あけ）
        for lvl, vv in ((0, vals), (1, [vals[0], vals[-1]])):
            b = base - lvl * 600
            if horizontal:
                line((vv[0], b), (vv[-1], b), "dim")
            else:
                line((b, vv[0]), (b, vv[-1]), "dim")
            for t in vv:
                if horizontal:
                    line((t, edge), (t, b - 150), "dim")
                    line((t - 85, b - 85), (t + 85, b + 85), "dim")
                else:
                    line((edge, t), (b - 150, t), "dim")
                    line((b - 85, t - 85), (b + 85, t + 85), "dim")
            for a_, b_ in zip(vv, vv[1:]):
                L = b_ - a_
                if L < minl:
                    continue
                s = f"{L:,.0f}"
                if horizontal:
                    text(s, (a_ + b_) / 2 - len(s) * 55, b + 50, "dim")
                else:
                    text(s, b - 50, (a_ + b_) / 2 - len(s) * 55, "dim", angle=90.0)

    chain(gx, True, y0)
    chain(gy, False, x0)
    # 通り芯の記号（上と右）
    r = fc["dim"]["bubble_radius_mm"]
    yt = max(ys) + B / 2 + off
    xr = max(xs) + B / 2 + off
    def spaced(vals):
        """記号が重ならないように、前の記号から記号の直径以上はなれた通り芯だけに付ける"""
        out, last = [], None
        for v in vals:
            if last is None or v - last >= 2 * r + 60:
                out.append(v)
                last = v
        return out

    for i, t in enumerate(spaced(gx), 1):
        line((t, max(ys) + B / 2 + 100), (t, yt - r), "center")
        msp.add_circle(P(t, yt), r / S, jwwattribs={"layer": lay["dim"]["layer"], "pen_color": lay["dim"]["color"], "pen_style": 1})
        text(f"X{i}", t - r * 0.62, yt - 80, "dim", size=1.8)
    for i, t in enumerate(spaced(gy), 1):
        line((max(xs) + B / 2 + 100, t), (xr - r, t), "center")
        msp.add_circle(P(xr, t), r / S, jwwattribs={"layer": lay["dim"]["layer"], "pen_color": lay["dim"]["color"], "pen_style": 1})
        text(f"Y{i}", xr - r * 0.62, t - 80, "dim", size=1.8)
    # 表題と、基礎の寸法（仮）
    ty = y0 - 1150
    text("基礎伏図（布基礎）　縮尺 1/50　※寸法はすべて仮", min(xs), ty, "dim", size=3.5, length=14000)
    note = (f"立上り幅{A}／底盤幅{B}／底盤厚{fc['teiban_thickness']}／"
            f"立上り高{fc['tachiagari_height']}（GL上）／根入れ{fc['neire']}（GL下）")
    text(note, min(xs), ty - 520, "dim", size=2.4, length=14000)
    tmp = Path(os.environ.get("TEMP", ".")) / "jw_foundation_tmp.jww"
    d.saveas(str(tmp))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp, out)
    print(f"基礎伏図: 布 {len(walls)} 本、立上り{A}・底盤{B} → {out}")


if __name__ == "__main__":
    main()
