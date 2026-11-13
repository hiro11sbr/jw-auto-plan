"""トイレの便器の位置と向きを、入力の画像から見つけて <階>_便器.json に書く（make_plan が部品を置く）。

usage: toilet_place.py <plan.png> <walls.json> <部屋名.json> <config.json> <out_便器.json>

- トイレの区画（2畳以下で、トイレの文字があるか、便器の形が見つかる区画）で、便器の半楕円
  （toilet_detect.find_pan）を探し、前の端の位置（モデルmm）と前を向く向き（モデルの単位ベクトル）を残す。
- 区画の中に便器が見つからない時は書かない（トイレの文字だけでは向きがわからない）。
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from opening_detect import Mapper  # noqa: E402
from toilet_detect import find_pan  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
LEN_MM = 736  # 便器の長さ（標準）


def main():
    img_path, walls_path, names_path, cfg_path, out_path = sys.argv[1:6]
    import spec
    cfg = spec.load_cfg(cfg_path)
    tc = cfg["toilet"]
    det = json.load(open(walls_path, encoding="utf-8"))
    M = Mapper(det, cfg["input"]["grid_mm"])
    mmpp = det["mm_per_px"]
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR).astype(int)
    b_, g_, r_ = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    neutral = (np.abs(r_ - g_) <= 30) & (np.abs(g_ - b_) <= 30)
    lines = neutral & (bgr.mean(axis=2) < tc["dark_max"])
    names = json.load(open(names_path, encoding="utf-8")) if Path(names_path).exists() else []
    faces = list(polygonize(unary_union([LineString([w["p1"], w["p2"]]) for w in det["walls"]])))
    out = []
    for f in faces:
        if f.area > tc["max_area_m2"] * 1e6 or f.area < tc["min_area_m2"] * 1e6:
            continue
        has_name = any(n.get("name") == "トイレ" and f.contains(Point(M.xmm(n["x"]), M.ymm(n["y"]))) for n in names)
        xs, ys = zip(*[(M.x(x), M.y(y)) for x, y in f.exterior.coords])
        pad = 7
        x0, y0, x1, y1 = int(min(xs)) + pad, int(min(ys)) + pad, int(max(xs)) - pad, int(max(ys)) - pad
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue
        # 向きごとに探し、タンク側（後ろ）が壁に近い向きを選ぶ（後ろから壁まで max_back_to_wall_mm 以内。
        # ユーザーの正解図：タンクは壁に付く）。近い向きがなければ一番一致する向き
        cands = []
        for fr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r = find_pan(lines[y0:y1, x0:x1], mmpp, tc, (fr,))
            if r[3] is None or r[0] < tc["min_score"] * 0.5:
                continue
            tip_ = (r[1] + x0 + r[3][0] * r[4], r[2] + y0 + r[3][1] * r[4])
            mx0_, my0_ = M.xmm(r[1] + x0), M.ymm(r[2] + y0)
            mx1_, my1_ = M.xmm(r[1] + x0 + r[3][0] * 100), M.ymm(r[2] + y0 + r[3][1] * 100)
            L_ = math.hypot(mx1_ - mx0_, my1_ - my0_)
            fd_ = ((mx1_ - mx0_) / L_, (my1_ - my0_) / L_)
            tx_, ty_ = M.xmm(tip_[0]), M.ymm(tip_[1])
            hit_ = LineString([(tx_, ty_), (tx_ - fd_[0] * 2500, ty_ - fd_[1] * 2500)]).intersection(f.exterior)
            hp_ = [hit_] if hit_.geom_type == "Point" else [q for q in getattr(hit_, "geoms", []) if q.geom_type == "Point"]
            D_ = min((math.hypot(q.x - tx_, q.y - ty_) for q in hp_), default=None)
            back_gap = None if D_ is None else abs(LEN_MM - (D_ - cfg["offsets_mm"]["board_face_inside"]))
            cands.append((r, back_gap))
        near = [c for c in cands if c[1] is not None and c[1] <= tc["max_back_to_wall_mm"]]
        pool = near or [c for c in cands if c[0][0] >= tc["min_score"]]
        if pool:
            sc, cx, cy, front, a, b = max(pool, key=lambda c: c[0][0])[0]
        else:
            sc, cx, cy, front, a, b = 0.0, None, None, None, None, None
        if sc < tc["min_score"] or front is None:
            if has_name:
                print(f"トイレの文字はあるが便器の形が見つからない: 区画 {f.area / 1e6:.2f}㎡（最良 {sc:.2f}）")
            continue
        px, py = cx + x0, cy + y0
        tip = (px + front[0] * a, py + front[1] * a)  # 前の端（画像px）
        mx0, my0 = M.xmm(px), M.ymm(py)
        mx1, my1 = M.xmm(px + front[0] * 100), M.ymm(py + front[1] * 100)
        L = math.hypot(mx1 - mx0, my1 - my0)
        fdir = ((mx1 - mx0) / L, (my1 - my0) / L)
        tx, ty = M.xmm(tip[0]), M.ymm(tip[1])
        # 向きの決まり（user 11/13）：便器の長手＝トイレの長手方向、とがった側（前）＝トイレの中心を向く
        # （タンク側が短辺の壁に付く）。どちらの短辺にタンクを付けるかは、検出した向きと位置から決める
        bx0, by0, bx1, by1 = f.bounds
        long_x = (bx1 - bx0) >= (by1 - by0)
        u = (1.0, 0.0) if long_x else (0.0, 1.0)  # 長手方向
        lo, hi = (bx0, bx1) if long_x else (by0, by1)
        board = cfg["offsets_mm"]["board_face_inside"]
        pu = tx if long_x else ty  # 検出した前の端の長手方向の位置
        if abs(fdir[0] * u[0] + fdir[1] * u[1]) > 0.7:
            sign = 1.0 if (fdir[0] * u[0] + fdir[1] * u[1]) > 0 else -1.0  # 検出が長手向き：その向き
        else:
            sign = 1.0 if (pu - lo) < (hi - pu) else -1.0  # 検出が短手向き：前の端が近い側の反対にタンク
            sign = -sign
        fdir = (u[0] * sign, u[1] * sign)
        end = lo + board if sign > 0 else hi - board  # タンクの付く壁のボード面
        new_u = end + sign * LEN_MM
        # 便器の軸（長手の中心線）は、トイレの軸＝2つの長い壁の芯の真ん中（910幅なら、芯から455の位置）
        mid_w = (by0 + by1) / 2 if long_x else (bx0 + bx1) / 2
        if long_x:
            tx, ty = new_u, mid_w
        else:
            tx, ty = mid_w, new_u
        out.append({"tip_mm": [round(tx, 1), round(ty, 1)], "front": [round(fdir[0], 4), round(fdir[1], 4)],
                    "score": round(sc, 2), "named": has_name, "area_m2": round(f.area / 1e6, 2)})
        print(f"便器: 前の端 mm({out[-1]['tip_mm'][0]:.0f},{out[-1]['tip_mm'][1]:.0f}) 向き{fdir} 一致{sc:.2f} 区画{f.area / 1e6:.2f}㎡ 文字{'あり' if has_name else 'なし'}")
    fix = Path(out_path.replace("_便器.json", "_便器_手直し.json"))
    if fix.exists():  # ユーザーの正解図の位置・向きで置き換える（手直し）
        out = json.load(open(fix, encoding="utf-8"))
        print(f"手直しの便器を使う: {fix.name}")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"便器: {len(out)}")


if __name__ == "__main__":
    main()
