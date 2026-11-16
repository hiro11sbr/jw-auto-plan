"""ユニットバス（UB）の位置・浴槽の向き・ドアを、元の画像から読む（user 11/16）。

usage: ub_detect.py <plan.png> <中間データの階のprefix> <config.json>
  → <prefix>_UB.json : [{"center_mm", "tub", "door", "door_end", "door_line_mm", "door_span_mm",
                         "hinge_mm", "round", "inner_mm", "score"}]
  向きはどれも図の向きの単位ベクトル（+x＝右、+y＝上）。
  - tub：UB の中心から見て、浴槽が付いている壁の向き
  - door：ドアのある壁の向き。door_end：その壁の上で、ドアが UB の中心からずれている向き
  - round：浴槽の丸い側の向き
  - center_mm：UB を囲む壁芯の真ん中（UB の部品の外形の中心をここに置く）

見つけ方（5プランで確かめた、11/16）：
  浴槽の区画＝洗い場と仕切り線で分かれた 1500〜1750 × 620〜820mm の白い長方形で、中に浴槽
  （1000〜1550 × 400〜780mm の線のかたまり）があるもの。外側が壁（灰色）でない辺が仕切り線。
  仕切り線から洗い場の向こうの壁までで UB の内側の四角。その4辺の外で壁の帯が 550mm 以上
  途切れる所がドアで、吊元は円弧が描いてある側の端（円弧の当たり 100% 対 48% 以下）。
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402
from opening_detect import Mapper  # noqa: E402

DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 画像の向き（上・下・左・右）


def grayish(v):
    return (v >= 165) & (v <= 215)


def detect(img, mmpp):
    """画像の上の UB（px）。[{ub:(X0,Y0,X1,Y1), sep, tub_box, round, doors:[(dir, s, e, hinge, score)]}]"""
    v = img.astype(int).mean(2)
    dark = v < 140
    H, W = v.shape
    cs, hier = cv2.findContours(dark.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    boxes = [cv2.boundingRect(c) for c in cs]
    res = []
    for k, c in enumerate(cs):
        if hier[0][k][3] < 0:
            continue
        x, y, w, h = boxes[k]
        L, S = max(w, h) * mmpp, min(w, h) * mmpp
        if not (1500 <= L <= 1750 and 620 <= S <= 820 and cv2.contourArea(c) / (w * h) > 0.95):
            continue
        tubs = [j for j, (a, b, ww, hh) in enumerate(boxes) if hier[0][j][3] < 0 and j != k
                and a >= x and b >= y and a + ww <= x + w and b + hh <= y + h
                and 1000 <= max(ww, hh) * mmpp <= 1550 and 400 <= min(ww, hh) * mmpp <= 780]
        if not tubs:
            continue

        def out_gray(dx, dy):
            if dy:
                yy = (y - 6) if dy < 0 else (y + h + 5)
                strip = v[max(0, yy - 2):yy + 3, x + 5:x + w - 5]
            else:
                xx = (x - 6) if dx < 0 else (x + w + 5)
                strip = v[y + 5:y + h - 5, max(0, xx - 2):xx + 3]
            return grayish(strip).mean() if strip.size else 1.0
        sep = min(DIRS, key=lambda d: out_gray(*d))
        dx, dy = sep
        depths = []
        for f in np.linspace(0.2, 0.8, 7):
            px, py = (x + f * w, y + h if dy > 0 else y) if dy else (x + w if dx > 0 else x, y + f * h)
            for t in range(4, int(2200 / mmpp)):
                qx, qy = int(px + dx * t), int(py + dy * t)
                if not (0 <= qx < W and 0 <= qy < H):
                    break
                if grayish(v[max(0, qy - 1):qy + 2, max(0, qx - 1):qx + 2]).mean() > 0.8:
                    depths.append(t)
                    break
        if not depths:
            continue
        t_wall = int(np.median(depths))
        if dy > 0:
            ub = (x, y, x + w, y + h + t_wall)
        elif dy < 0:
            ub = (x, y - t_wall, x + w, y + h)
        elif dx > 0:
            ub = (x, y, x + w + t_wall, y + h)
        else:
            ub = (x - t_wall, y, x + w, y + h)
        # 内側の大きさ：1616 は 1400〜1900 角、1621（壁芯 2275×1820）は長い辺が 2400 まで（11/16）
        ew_, eh_ = (ub[2] - ub[0]) * mmpp, (ub[3] - ub[1]) * mmpp
        if not (1400 <= min(ew_, eh_) <= 1900 and max(ew_, eh_) <= 2400):
            continue
        # 浴槽の丸い側：外の線が真ん中でふくらむ端
        j = max(tubs, key=lambda j: boxes[j][2] * boxes[j][3])
        a, b, ww, hh = boxes[j]
        sub_ = dark[b:b + hh, a:a + ww]
        if ww <= hh:
            sub_ = sub_.T
        n_ = sub_.shape[0]
        lo_ = np.array([np.flatnonzero(sub_[i])[0] if sub_[i].any() else 999 for i in range(n_)])
        hi_ = np.array([np.flatnonzero(sub_[i])[-1] if sub_[i].any() else -1 for i in range(n_)])
        cen = slice(int(n_ * 0.4), int(n_ * 0.6) + 1)
        edg = list(range(int(n_ * 0.12), int(n_ * 0.25) + 1)) + list(range(int(n_ * 0.75), int(n_ * 0.88) + 1))
        b0 = np.median(lo_[edg]) - np.median(lo_[cen])
        b1 = np.median(hi_[cen]) - np.median(hi_[edg])
        if ww > hh:
            rnd = (-1, 0) if b0 > b1 else (1, 0)
        else:
            rnd = (0, -1) if b0 > b1 else (0, 1)
        # ドア：UB の4辺の外で、壁の帯が 550mm 以上途切れる所。吊元＝円弧のある端
        doors = []
        X0, Y0, X1, Y1 = ub
        for d in DIRS:
            ddx, ddy = d
            if ddy:
                line = Y0 - 6 if ddy < 0 else Y1 + 5
                prof = grayish(v[line - 2:line + 3, X0:X1]).mean(0) > 0.5
                base = X0
            else:
                line = X0 - 6 if ddx < 0 else X1 + 5
                prof = grayish(v[Y0:Y1, line - 2:line + 3]).mean(1) > 0.5
                base = Y0
            m_ = np.concatenate(([0], (~prof).astype(int), [0]))
            st, en = np.flatnonzero(np.diff(m_) == 1), np.flatnonzero(np.diff(m_) == -1)
            face = (Y0 if ddy < 0 else Y1) if ddy else (X0 if ddx < 0 else X1)
            for s_, e_ in zip(st, en):
                r = e_ - s_
                if r * mmpp < 550:
                    continue
                per = {}
                for hinge in (s_, e_):
                    sgn = 1 if hinge == s_ else -1
                    best = 0.0
                    for o in range(0, 9):
                        for rr in range(int(0.6 * r), r + 1):
                            hits = 0
                            angs = np.linspace(math.radians(20), math.radians(70), 25)
                            for ang in angs:
                                along = hinge + sgn * (o + rr * math.cos(ang))
                                into = rr * math.sin(ang)
                                qx, qy = (base + along, face - ddy * into) if ddy else (face - ddx * into, base + along)
                                qx, qy = int(round(qx)), int(round(qy))
                                if 0 <= qx < W and 0 <= qy < H and dark[max(0, qy - 1):qy + 2, max(0, qx - 1):qx + 2].any():
                                    hits += 1
                            best = max(best, hits / len(angs))
                    per[int(hinge)] = best
                hg = max(per, key=per.get)
                doors.append((d, base + s_, base + e_, base + hg, per[hg], min(per.values())))
        res.append(dict(ub=ub, sep=sep, tub_box=(a, b, a + ww, b + hh), round=rnd, doors=doors))
    return res


def to_mm_dir(d):
    """画像の向き（+y が下）→ 図の向き（+y が上）"""
    return (d[0], -d[1])


def main():
    img_p, pre, cfgp = sys.argv[1:4]
    cfg = spec.load_cfg(cfgp)
    det = json.load(open(f"{pre}_walls.json", encoding="utf-8"))
    M = Mapper(det, cfg["input"]["grid_mm"])
    img = cv2.imdecode(np.fromfile(img_p, np.uint8), cv2.IMREAD_COLOR)
    walls = []
    for w in det["walls"]:
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        if abs(y1 - y2) < 1:
            walls.append((True, y1, min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) < 1:
            walls.append((False, x1, min(y1, y2), max(y1, y2)))
    out = []
    for r in detect(img, det["mm_per_px"]):
        X0, Y0, X1, Y1 = r["ub"]
        xa, xb = sorted((M.xmm(X0), M.xmm(X1)))
        ya, yb = sorted((M.ymm(Y0), M.ymm(Y1)))
        cxm, cym = (xa + xb) / 2, (ya + yb) / 2

        def wall_line(h, edge, sgn, mid):
            """UB の辺の外（sgn の向き）で、いちばん近い壁芯（300mm 以内、UB の真ん中を通る壁）"""
            c_ = [ln for hh, ln, a, b in walls if hh == h and a - 1 <= mid <= b + 1 and 0 <= (ln - edge) * sgn <= 300]
            return min(c_, key=lambda ln: abs(ln - edge)) if c_ else None
        L_ = wall_line(False, xa, -1, cym)
        R_ = wall_line(False, xb, 1, cym)
        B_ = wall_line(True, ya, -1, cxm)
        T_ = wall_line(True, yb, 1, cxm)
        cx_ = (L_ + R_) / 2 if L_ is not None and R_ is not None else cxm
        cy_ = (B_ + T_) / 2 if B_ is not None and T_ is not None else cym
        tub = to_mm_dir((-r["sep"][0], -r["sep"][1]))
        doors = [d for d in r["doors"] if d[4] >= 0.6]
        if not doors:
            print(f"UB @({cx_:.0f},{cy_:.0f})：ドアの円弧が見つからない（UB は置き、ドアの開口は作らない）")
        dr = max(doors, key=lambda d: d[4]) if doors else None
        item = {"center_mm": [round(cx_, 1), round(cy_, 1)], "inner_mm": [round(xb - xa), round(yb - ya)],
                "tub": list(tub), "round": list(to_mm_dir(r["round"]))}
        if dr:
            d, s_, e_, hg, sc, other = dr
            dd = to_mm_dir(d)
            h = dd[1] != 0  # ドアの壁が横（上か下）
            if h:
                a_, b_ = sorted((M.xmm(s_), M.xmm(e_)))
                hinge = M.xmm(hg)
                line = T_ if dd[1] > 0 else B_
                mid_, c0 = (a_ + b_) / 2, cx_
            else:
                a_, b_ = sorted((M.ymm(s_), M.ymm(e_)))
                hinge = M.ymm(hg)
                line = R_ if dd[0] > 0 else L_
                mid_, c0 = (a_ + b_) / 2, cy_
            sgn = 1 if mid_ >= c0 else -1
            item.update({"door": list(dd), "door_end": [sgn, 0] if h else [0, sgn], "door_horizontal": h,
                         "door_line_mm": line, "door_span_mm": [round(a_, 1), round(b_, 1)],
                         "hinge_mm": round(hinge, 1), "score": [round(sc, 2), round(other, 2)]})
        out.append(item)
        print(f"UB @({cx_:.0f},{cy_:.0f}) 浴槽{tuple(tub)} 丸{tuple(item['round'])} ドア{tuple(item.get('door', ()))}"
              f" {item.get('door_span_mm')} 吊元{item.get('hinge_mm')}")
    json.dump(out, open(f"{pre}_UB.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"UB {len(out)}")


if __name__ == "__main__":
    main()
