"""基礎伏図（布基礎）を、壁の判定（<階>_walls.json）から作る。

usage: make_foundation.py <1F_walls.json> <出力.jww> [基礎.json]

- 立上り：壁芯の両側に、立上りの幅の半分ずつ（外形線2本）。壁の端は、角・T・自由な端のどこでも、幅の半分だけ飛び出す（立上りも底盤も）。壁の下は全部に通す（提案の位置。
  ユーザーが位置を指示するまでの仮、11/15）
- 底盤（布のフーチング）：壁芯の両側に、底盤の幅の半分ずつ（実線。user 11/15）
- 通り芯（壁芯）と、芯々の寸法、通り芯の記号
- 値は 05_設定/基礎.json（立上り幅150・底盤幅600・底盤厚150・立上り高450・根入れ600・布基礎、全部仮）
- アンカーボルトは別の図で描く（ユーザーの指示待ち）
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


def main():
    walls_path, out = sys.argv[1], sys.argv[2]
    cfg_path = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "05_設定" / "基礎.json")
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

    # 底盤と立上り（壁芯から両側に半分ずつ。端（角・T・自由な端）は幅の半分だけ飛び出す（square cap））
    for width, key in ((B, "teiban"), (A, "tachiagari")):
        shape = unary_union([LineString(w).buffer(width / 2, cap_style=3, join_style=2) for w in walls])
        for r in rings(shape):
            for p, q in zip(r, r[1:]):
                if abs(p[0] - q[0]) + abs(p[1] - q[1]) > 0.5:
                    line((round(p[0], 1), round(p[1], 1)), (round(q[0], 1), round(q[1], 1)), key)
    # 通り芯（壁芯）
    for p, q in walls:
        line(p, q, "center")
    # 寸法：下と左に、芯々の寸法。上と右に通り芯の記号
    # 通り芯＝壁が通っている線（縦の壁の x、横の壁の y）。壁の端だけの位置は通り芯にしない
    gx = sorted({round(w[0][0], 1) for w in walls if abs(w[0][0] - w[1][0]) < 1})
    gy = sorted({round(w[0][1], 1) for w in walls if abs(w[0][1] - w[1][1]) < 1})
    off = fc["dim"]["offset_mm"]
    minl = fc["dim"]["min_text_mm"]
    y0 = min(ys) - B / 2 - off
    x0 = min(xs) - B / 2 - off

    def chain(vals, horizontal, base):
        for lvl, vv in ((0, vals), (1, [vals[0], vals[-1]])):
            b = base - lvl * 600
            if horizontal:
                line((vv[0], b), (vv[-1], b), "dim")
            else:
                line((b, vv[0]), (b, vv[-1]), "dim")
            for t in vv:
                p, q = ((t, b - 120), (t, b + 120)) if horizontal else ((b - 120, t), (b + 120, t))
                line(p, q, "dim")
            for a_, b_ in zip(vv, vv[1:]):
                L = b_ - a_
                if L < minl:
                    continue
                s = f"{L:,.0f}"
                if horizontal:
                    text(s, (a_ + b_) / 2 - len(s) * 55, b + 40, "dim")
                else:
                    text(s, b - 40, (a_ + b_) / 2 - len(s) * 55, "dim", angle=90.0)

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
    print(f"基礎伏図: 壁 {len(walls)} 本、立上り{A}・底盤{B} → {out}")


if __name__ == "__main__":
    main()
