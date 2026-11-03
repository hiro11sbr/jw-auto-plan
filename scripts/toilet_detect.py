"""便器の形でトイレを見つける（ユーザー 2026-10-24）。トイレの文字が読めなくても、便器（楕円、後ろに
四角いタンクがある場合と無い場合あり）がある2畳以下の閉じた区画はトイレ。

便器の前側（丸い側）の半楕円が細い線で描かれているかを、大きさと向き（壁に平行・直角の4方向）を変えて
探す。文字が便器に重なっていても、半楕円の線の大部分が見えていれば拾える。便器の後ろ（タンク側）は
壁に近い。

usage: toilet_detect.py <plan.png> <walls.json> <部屋名.json> <config.json>
見つけたトイレを <部屋名.json> に足す（source="便器の形"）。すでにトイレの文字がある区画には足さない。
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opening_detect import Mapper  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def find_pan(line_mask, mmpp, tc):
    """Best half-oval in the mask: (score, cx, cy, front direction, half-length px, half-width px).
    The front half of a bowl: an ellipse half, its open side toward the tank (back)."""
    H, W = line_mask.shape
    best = (0.0, None, None, None, None, None)
    near = cv2.dilate(line_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    for a_mm in range(tc["half_long_mm"][0], tc["half_long_mm"][1] + 1, 25):
        for b_mm in range(tc["half_short_mm"][0], tc["half_short_mm"][1] + 1, 20):
            a, b = a_mm / mmpp, b_mm / mmpp
            if b > a:
                continue
            for front in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                # points on the front half: angle -80..80 deg around the front direction; also the
                # same curve shrunk and grown a little, which must be mostly blank (a drawn bowl is
                # one line with clear space on both sides; hatching or clutter is ink everywhere)
                def curve(s):
                    out = []
                    for deg in range(-80, 81, 8):
                        t = math.radians(deg)
                        u, v = s * a * math.cos(t), s * b * math.sin(t)  # u along front, v across
                        out.append((int(round(u * front[0] - v * front[1])), int(round(u * front[1] + v * front[0]))))
                    return out
                pts, pin, pout = curve(1.0), curve(tc["blank_in"]), curve(tc["blank_out"])
                # the bowl's two sides run on straight back toward the tank (a letter's arc or a
                # box corner does not)
                legs = []
                for k in range(2, 9):
                    u = -a * tc["leg_len"] * k / 8
                    for v in (b, -b):
                        legs.append((int(round(u * front[0] - v * front[1])), int(round(u * front[1] + v * front[0]))))
                allp = pts + pin + pout + legs
                xs = [p[0] for p in allp]
                ys = [p[1] for p in allp]
                x0, x1 = -min(xs), W - max(xs)
                y0, y1 = -min(ys), H - max(ys)
                if x1 <= x0 or y1 <= y0:
                    continue

                def mean_at(ps, m):
                    acc = np.zeros((y1 - y0, x1 - x0), np.float32)
                    for dx, dy in ps:
                        acc += m[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
                    return acc / len(ps)
                raw = line_mask.astype(np.float32)
                acc = np.minimum(mean_at(pts, near), mean_at(legs, near)) \
                    - np.maximum(mean_at(pin, raw), mean_at(pout, raw))
                k = int(np.argmax(acc))
                sc = float(acc.flat[k])
                if sc > best[0]:
                    cy, cx = divmod(k, acc.shape[1])
                    best = (sc, cx + x0, cy + y0, front, a, b)
    return best


def main():
    img_path, walls_path, names_path, cfg_path = sys.argv[1:5]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
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
    found = 0
    for f in faces:
        if f.area > tc["max_area_m2"] * 1e6 or f.area < tc["min_area_m2"] * 1e6:
            continue
        if any(n.get("name") == "トイレ" and f.contains(Point(M.xmm(n["x"]), M.ymm(n["y"]))) for n in names):
            continue
        xs, ys = zip(*[(M.x(x), M.y(y)) for x, y in f.exterior.coords])
        pad = 7  # inside the wall band
        x0, y0, x1, y1 = int(min(xs)) + pad, int(min(ys)) + pad, int(max(xs)) - pad, int(max(ys)) - pad
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue
        sc, cx, cy, front, a, b = find_pan(lines[y0:y1, x0:x1], mmpp, tc)
        if sc < tc["min_score"]:
            continue
        px, py = cx + x0, cy + y0
        names.append({"raw": "（便器の形）", "x": round(px, 1), "y": round(py, 1),
                      "box": [round(px - 10, 1), round(py - 10, 1), round(px + 10, 1), round(py + 10, 1)],
                      "name": "トイレ", "kind": "水回り", "score": round(sc, 2), "matched": "便器",
                      "source": "便器の形"})
        found += 1
        print(f"トイレ（便器の形）: 区画 {f.area / 1e6:.2f}㎡ 一致{sc:.2f} 位置px({px:.0f},{py:.0f})")
    json.dump(names, open(names_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"便器の形で見つけたトイレ: {found}")


if __name__ == "__main__":
    main()
