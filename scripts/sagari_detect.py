"""下がり壁（天井から下がった壁。床までは無い）をプラン図から読む（user 11/18 の 97）。

usage: sagari_detect.py <plan.png> <walls.json> <config.json> <out_prefix>
writes <out_prefix>_下がり壁.json  [{horizontal, line_mm, a_mm, b_mm}]

プラン図では、壁の両側の線を点線で描き、間は塗らない（床の色のまま）。必ずグリッド（455・227.5）の上にあり、
芯振り分け（65 と 65）。読み方：グリッドの線ごとに、芯の両側（壁の厚みほど離れた所）の2本が、どちらも
点線（黒い所と切れ目がくり返す）で、間に黒も壁の灰色も無い所を探す。壁と読んだ所の上は見ない。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from opening_detect import Mapper  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def runs(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def main():
    img_path, walls_path, cfg_path, prefix = sys.argv[1:5]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    sc = cfg["sagari_kabe"]
    out = []
    if not sc.get("use", False):
        json.dump(out, open(prefix + "_下がり壁.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    det = json.load(open(walls_path, encoding="utf-8"))
    M = Mapper(det, cfg["input"]["grid_mm"])
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(int)
    b, g_, r = (bgr[..., i].astype(int) for i in range(3))
    neutral = (np.abs(r - g_) <= 10) & (np.abs(g_ - b) <= 10)
    H, W = gray.shape
    dark = gray < sc["dark_max"]
    wallgray = neutral & (gray >= 150) & (gray <= 215)
    half = cfg["input"]["half_grid_mm"]
    mmpp = abs(M.xmm(100) - M.xmm(0)) / 100
    xs = [w["p1"][0] for w in det["walls"]] + [w["p2"][0] for w in det["walls"]]
    ys = [w["p1"][1] for w in det["walls"]] + [w["p2"][1] for w in det["walls"]]
    walls = [(abs(w["p1"][1] - w["p2"][1]) < 1, w) for w in det["walls"]]
    win = max(8, int(round(sc["window_mm"] / mmpp)))
    for horiz in (True, False):
        lo, hi = (min(ys), max(ys)) if horiz else (min(xs), max(xs))
        for n in range(int(np.floor(lo / half)), int(np.ceil(hi / half)) + 1):
            line = n * half
            c0 = int(round(M.y(line) if horiz else M.x(line)))
            if not (20 <= c0 < (H if horiz else W) - 20):
                continue
            L = W if horiz else H
            best = None
            # 両側の線の位置（芯から k px）。点線は線の上で切れているので、少し幅を持たせて見る。
            # 帯の中心は通り芯から少しずれる（プラン2 1F 玄関と土間収納の間：2.8px）
            sm = sc.get("center_shift_px", 3)
            for k, ci in [(k, c0 + d) for d in sorted(range(-sm, sm + 1), key=abs)
                          for k in range(sc["half_min_px"], sc["half_max_px"] + 1)]:
                def strip(o):
                    s = slice(ci + o - 1, ci + o + 2)
                    return (dark[s, :] if horiz else dark[:, s].T).any(axis=0)
                ea, eb = strip(-k), strip(k)
                inner = slice(ci - k + 2, ci + k - 1)
                inside_dark = (dark[inner, :] if horiz else dark[:, inner].T).any(axis=0)
                # 芯の上（帯の真ん中）に点線が乗っているものは下がり壁ではない（トイレ横の点線など。下がり壁の
                # 点線は芯から壁厚の半分ずれた2本。user 11/18）
                centre = strip(0) if False else (dark[ci - 1:ci + 2, :] if horiz else dark[:, ci - 1:ci + 2].T).any(axis=0)
                inside_gray = (wallgray[inner, :] if horiz else wallgray[:, inner].T).mean(axis=0) > 0.5
                ok = np.zeros(L, bool)
                for s0 in range(0, L - win, max(2, win // 4)):
                    s1 = s0 + win
                    fa, fb = ea[s0:s1].mean(), eb[s0:s1].mean()
                    na = len(runs(ea[s0:s1]))
                    nb = len(runs(eb[s0:s1]))
                    if (sc["dash_min"] <= fa <= sc["dash_max"] and sc["dash_min"] <= fb <= sc["dash_max"]
                            and na >= sc["dash_count_min"] and nb >= sc["dash_count_min"]
                            and inside_dark[s0:s1].mean() <= sc["inside_dark_max"]
                            and centre[s0:s1].mean() < sc["dash_min"]):
                        ok[s0:s1] = True
                score = ok.sum()
                if best is None or score > best[0]:
                    best = (score, ok, k, ea & eb | ea ^ eb, ea, eb)
            if not best or best[0] == 0:
                continue
            _s, ok, k, _any, ea, eb = best
            for s, e in runs(ok):
                # 窓（window_mm）で見た範囲は、点線の外へはみ出す：両側の点線がどちらもある所の両端まで縮める
                # 実線（長く続く黒）は壁の輪郭：点線の所だけ（短い黒）で両端を決める
                def dashes(x):
                    d = x.copy()
                    for r0, r1 in runs(x):
                        if r1 - r0 > sc.get("solid_px", 15):
                            d[r0:r1] = False
                    return d
                da, db = dashes(ea[s:e]), dashes(eb[s:e])
                both = np.where(da | db)[0]
                if len(both) == 0:
                    continue
                s, e = s + both[0], s + both[-1] + 1
                if (e - s) * mmpp < sc["min_len_mm"]:
                    continue
                a = M.xmm(s) if horiz else M.ymm(e - 1)
                b2 = M.xmm(e - 1) if horiz else M.ymm(s)
                a, b2 = sorted((a, b2))
                # 壁と読んだ所の上でも、点線の組なら下がり壁（実線の壁は点線の条件に当てはまらない）。
                # そこの壁は作図で外す（プラン2 1F 玄関と土間収納の間が壁と読まれて閉じていた。user 11/18）
                on_wall = False
                for wh, w in walls:
                    if wh != horiz:
                        continue
                    wc = w["p1"][1] if horiz else w["p1"][0]
                    wa, wb = sorted((w["p1"][0], w["p2"][0]) if horiz else (w["p1"][1], w["p2"][1]))
                    if abs(wc - line) < 1 and min(b2, wb) - max(a, wa) > 0.5 * (b2 - a):
                        on_wall = True
                # 建物の中だけ。両端の近く（end_wall_mm 以内）に壁があること（下がり壁は壁から壁へ渡る）
                if horiz:
                    a, b2 = max(a, min(xs)), min(b2, max(xs))
                else:
                    a, b2 = max(a, min(ys)), min(b2, max(ys))
                if b2 - a < sc["min_len_mm"]:
                    continue

                def near_wall(t):
                    for wh, w in walls:
                        wa, wb = sorted((w["p1"][0], w["p2"][0]) if wh else (w["p1"][1], w["p2"][1]))
                        wc = w["p1"][1] if wh else w["p1"][0]
                        if wh != horiz and abs(wc - t) <= sc["end_wall_mm"] and wa - 1 <= line <= wb + 1:
                            return True  # 直交する壁
                        if wh == horiz and abs(wc - line) < 1 and min(abs(wa - t), abs(wb - t)) <= sc["end_wall_mm"]:
                            return True  # 同じ通りの壁の端
                    return False
                if not (near_wall(a) and near_wall(b2)):
                    continue
                # 芯から芯（両端の直交する壁の芯）までが 910 未満のものは除外（user 11/18）
                # 両端は、近くの壁の芯（直交する壁、または同じ通りの壁の端）に合わせる。無ければ 227.5 の倍数
                ends = [(w["p1"][1] if wh else w["p1"][0]) for wh, w in walls if wh != horiz
                        and min((w["p1"][0], w["p2"][0]) if wh else (w["p1"][1], w["p2"][1])) - 1 <= line
                        <= max((w["p1"][0], w["p2"][0]) if wh else (w["p1"][1], w["p2"][1])) + 1]
                ends += [v for wh, w in walls if wh == horiz and abs((w["p1"][1] if wh else w["p1"][0]) - line) < 1
                         for v in ((w["p1"][0], w["p2"][0]) if wh else (w["p1"][1], w["p2"][1]))]

                def snap_end(t):
                    c = min(ends, key=lambda v: abs(v - t), default=None)
                    return c if c is not None and abs(c - t) <= sc["end_wall_mm"] else round(t / half) * half
                ca, cb = snap_end(a), snap_end(b2)
                a, b2 = min(ca, cb), max(ca, cb)
                if abs(cb - ca) < sc.get("min_span_mm", 910) - 1:
                    print(f"除外（芯芯 {abs(cb - ca):.0f} < 910）: {'横' if horiz else '縦'} {line} {a:.0f}〜{b2:.0f}")
                    continue
                out.append({"horizontal": horiz, "line_mm": line, "a_mm": round(a, 1), "b_mm": round(b2, 1),
                            "half_px": k, "over_wall": on_wall})
    json.dump(out, open(prefix + "_下がり壁.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for s in out:
        print("下がり壁", "横" if s["horizontal"] else "縦", s["line_mm"], s["a_mm"], s["b_mm"])


if __name__ == "__main__":
    main()
