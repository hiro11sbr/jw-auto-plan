"""洗面台（洗面化粧台）の位置・向き・間口を、元の画像から読む（user 11/16「洗面台も置けるように」）。

usage: vanity_detect.py <plan.png> <中間データの階のprefix> <config.json>
  → <prefix>_洗面.json : [{"box_mm": [x0, y0, x1, y1], "horizontal", "back": [dx, dy], "back_wall_mm",
                           "width_mm", "depth_mm", "centre_mm"}]
  back：洗面台の背中（壁側）の向き（図の向き、+x＝右、+y＝上）

見つけ方（プラン1〜5、11/16）：
  「洗面・脱衣・ランドリー」の名の区画の中にある、黒い線で囲まれた明るい長方形（中が灰色の洗濯機・冷蔵庫は
  除く）で、長い辺 550〜1700mm・短い辺 430〜640mm、縦横比 1.15 以上（正方形の洗濯機の防水パンは除く）。
  背中＝長い辺のうち、近くに壁（300mm 以内、長さの半分以上が壁に沿う）がある側。壁のない形は置かない。
  ※元の図が洗面台をほとんど絵で描いている（輪郭が薄い）プランでは見つからないことがある
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402
from opening_detect import Mapper  # noqa: E402

NAMES = ("洗面", "脱衣", "ランドリー")


def detect(img, mmpp, regions):
    """画像の上の洗面台（px）。[{box:(x0,y0,x1,y1), L, S, horiz}]"""
    v = img.astype(int).mean(2)
    dark = (v < 140).astype(np.uint8)
    cs, hier = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for r in regions:
        if not any(any(k in n for k in NAMES) for n in r["names"]) or r["area_m2"] > 12:
            continue
        x0, y0, x1, y1 = r["bbox_px"]
        parts = []
        for k, c in enumerate(cs):
            if hier[0][k][3] < 0:
                continue  # 線に囲まれた内側（穴）だけ
            x, y, w, h = cv2.boundingRect(c)
            if not (x0 - 5 <= x and y0 - 5 <= y and x + w <= x1 + 5 and y + h <= y1 + 5):
                continue
            if max(w, h) * mmpp > 1700 or min(w, h) * mmpp < 100 or cv2.contourArea(c) / (w * h) < 0.85:
                continue
            if v[y + 2:y + h - 2, x + 2:x + w - 2].mean() < 190:
                continue  # 灰色の塗り（洗濯機・冷蔵庫など）
            parts.append([x, y, x + w, y + h])
        # 引き戸や扉の線で何枚かに分かれた明るい四角は、隣り合う（そろった辺・すき間 4 画素以内）ものをつなぐ
        merged = True
        while merged:
            merged = False
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    a, b = parts[i], parts[j]
                    same_x = abs(a[0] - b[0]) <= 3 and abs(a[2] - b[2]) <= 3 and b[1] - a[3] <= 4 and a[1] - b[3] <= 4
                    same_y = abs(a[1] - b[1]) <= 3 and abs(a[3] - b[3]) <= 3 and b[0] - a[2] <= 4 and a[0] - b[2] <= 4
                    if same_x or same_y:
                        parts[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        del parts[j]
                        merged = True
                        break
                if merged:
                    break
        for x0_, y0_, x1_, y1_ in parts:
            w, h = x1_ - x0_, y1_ - y0_
            L, S = max(w, h) * mmpp, min(w, h) * mmpp
            if not (550 <= L <= 1700 and 430 <= S <= 640 and L / S >= 1.15):
                continue
            if any(abs(o["box"][0] - x0_) < 4 and abs(o["box"][1] - y0_) < 4 for o in out):
                continue
            out.append(dict(box=(x0_, y0_, x1_, y1_), L=L, S=S, horiz=w >= h))
    return out


def main():
    img_p, pre, cfgp = sys.argv[1:4]
    cfg = spec.load_cfg(cfgp)
    det = json.load(open(f"{pre}_walls.json", encoding="utf-8"))
    mmpp = det["mm_per_px"]
    M = Mapper(det, cfg["input"]["grid_mm"])
    regp = Path(f"{pre}_区画.json")
    regions = json.load(open(regp, encoding="utf-8")) if regp.exists() else []
    img = cv2.imdecode(np.fromfile(img_p, np.uint8), cv2.IMREAD_COLOR)
    walls = []
    for w in det["walls"]:
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        if abs(y1 - y2) < 1:
            walls.append((True, y1, min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) < 1:
            walls.append((False, x1, min(y1, y2), max(y1, y2)))
    out = []
    for c in detect(img, mmpp, regions):
        X0, Y0, X1, Y1 = c["box"]
        h = c["horiz"]
        xa, xb = sorted((M.xmm(X0), M.xmm(X1)))
        ya, yb = sorted((M.ymm(Y0), M.ymm(Y1)))
        ta, tb, ua, ub = (xa, xb, ya, yb) if h else (ya, yb, xa, xb)

        def side(u_edge, sg):
            """長い辺の外（sg の向き）の、いちばん近い壁芯（300mm 以内、洗面台の長さの半分以上が沿う）"""
            c_ = [ln for hh, ln, a, b in walls if hh == h and 0 <= (ln - u_edge) * sg <= 300
                  and min(b, tb) - max(a, ta) >= 0.5 * (tb - ta)]
            return min(c_, key=lambda ln: abs(ln - u_edge)) if c_ else None
        wl, wh = side(ua, -1), side(ub, 1)
        if wl is None and wh is None:
            print(f"洗面台の候補 @({(xa + xb) / 2:.0f},{(ya + yb) / 2:.0f}) {c['L']:.0f}×{c['S']:.0f}：近くに壁がない。置かない")
            continue
        bs = -1 if (wh is None or (wl is not None and abs(wl - ua) <= abs(wh - ub))) else 1
        back_wall = wl if bs < 0 else wh
        item = {"box_mm": [round(xa, 1), round(ya, 1), round(xb, 1), round(yb, 1)], "horizontal": h,
                "back": [0, bs] if h else [bs, 0], "back_wall_mm": back_wall,
                "width_mm": round(tb - ta), "depth_mm": round(ub - ua),
                "centre_mm": [round((xa + xb) / 2, 1), round((ya + yb) / 2, 1)]}
        out.append(item)
        print(f"洗面台 {'横' if h else '縦'} ({xa:.0f},{ya:.0f})-({xb:.0f},{yb:.0f}) 間口{tb - ta:.0f} 奥行き{ub - ua:.0f} 背中{tuple(item['back'])}")
    json.dump(out, open(f"{pre}_洗面.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"洗面台 {len(out)}")


if __name__ == "__main__":
    main()
