"""腰壁 (low walls: beside stairs, in front of the kitchen, ...) on the plan image.

usage: koshikabe_detect.py <plan.png> <walls.json> <config.json> <out_prefix>
writes <out_prefix>_腰壁.json  [{horizontal, line_mm, a_mm, b_mm}]

On the input plans a 腰壁 is drawn like a wall band but NOT filled: two outlines a wall's
thickness apart with white between (a real wall is filled grey). Its centre sits on a grid line
(455 / 227.5); its free end need not (1F kitchen: 2580 from the wall centre, user 2026-10-22).
Runs on the band of a detected wall are skipped. 框 (between 玄関 and ホール) is not looked for:
it is not on a wall line with this band shape.
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
    kc = cfg["koshikabe"]
    det = json.load(open(walls_path, encoding="utf-8"))
    M = Mapper(det, cfg["input"]["grid_mm"])
    gray = cv2.cvtColor(cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    dark = gray < kc["dark_max"]
    light = gray > kc["light_min"]
    half = cfg["input"]["half_grid_mm"]
    xs = [w["p1"][0] for w in det["walls"]] + [w["p2"][0] for w in det["walls"]]
    ys = [w["p1"][1] for w in det["walls"]] + [w["p2"][1] for w in det["walls"]]
    walls = [(abs(w["p1"][1] - w["p2"][1]) < 1, w) for w in det["walls"]]
    def scan(shift_max):
        out = []
        for horiz in (True, False):
            lo, hi = (min(ys), max(ys)) if horiz else (min(xs), max(xs))
            n0, n1 = int(np.floor(lo / half)), int(np.ceil(hi / half))
            for n in range(n0, n1 + 1):
                line = n * half
                c = M.y(line) if horiz else M.x(line)
                ci = int(round(c))
                if not (10 <= ci < (H if horiz else W) - 10):
                    continue
                L = W if horiz else H
                ok = np.zeros(L, bool)
                R = 8 + 3 * kc["half_max_px"]
                # 帯の中心は通り芯から少しずれることがある（プラン1 2F 階段の上 y=2730：1.5px。user 11/18 の 90）
                shifts = sorted(range(-shift_max, shift_max + 1), key=abs)
                for t in range(L):
                    wide = dark[max(0, ci - R):ci + R + 1, t] if horiz else dark[t, max(0, ci - R):ci + R + 1]
                    wl_ = light[max(0, ci - R):ci + R + 1, t] if horiz else light[t, max(0, ci - R):ci + R + 1]
                    o = min(R, ci)
                    hit = False
                    for sft in shifts:
                        if hit:
                            break
                        prof_d = wide[o + sft - 8:o + sft + 9]
                        prof_l = wl_[o + sft - 8:o + sft + 9]
                        if len(prof_d) != 17:
                            continue
                        for k in range(kc["half_min_px"], kc["half_max_px"] + 1):
                            if not (prof_d[8 - k - 1:8 - k + 2].any() and prof_d[8 + k - 1:8 + k + 2].any()
                                    and prof_l[8 - k + 2:8 + k - 1].all()):
                                continue
                            # 両外側にも同じ間隔で線が続く縞（棚・ハッチング）は腰壁ではない（user 11/18：プラン1 2F
                            # 主寝室の H1500 の棚の縞を腰壁と読んだ）。階段の踏み板の縞は片側だけなので残る
                            if kc.get("reject_stripes", True):
                                # 帯の2本の線の位置（a, b）と間隔 p_。外側の同じ間隔の所に線があるか
                                a_ = o + sft - 8 + max(i for i in range(8 - k - 1, 8 - k + 2) if prof_d[i])
                                b_ = o + sft - 8 + min(i for i in range(8 + k - 1, 8 + k + 2) if prof_d[i])
                                p_ = b_ - a_

                                def at(i):
                                    return 0 <= i - 1 and i + 2 <= len(wide) and wide[i - 1:i + 2].any()
                                if at(a_ - p_) and at(b_ + p_) and (at(a_ - 2 * p_) or at(b_ + 2 * p_)):
                                    continue
                            ok[t] = True
                            hit = True
                            break
                # close small gaps (text, JPEG)
                for s, e in runs(~ok):
                    if e - s <= kc["gap_px"]:
                        ok[s:e] = True
                for s, e in runs(ok):
                    if e - s < kc["min_len_px"]:
                        continue
                    a = M.xmm(s) if horiz else M.ymm(e - 1)
                    b = M.xmm(e - 1) if horiz else M.ymm(s)
                    a, b = sorted((a, b))
                    # not on the band of a detected wall
                    on_wall = False
                    for wh, w in walls:
                        if wh != horiz:
                            continue
                        wc = w["p1"][1] if horiz else w["p1"][0]
                        wa, wb = sorted((w["p1"][0], w["p2"][0]) if horiz else (w["p1"][1], w["p2"][1]))
                        if abs(wc - line) < 1 and min(b, wb) - max(a, wa) > 0.5 * (b - a):
                            on_wall = True
                    if not on_wall:
                        out.append({"horizontal": horiz, "line_mm": line, "a_mm": round(a, 1), "b_mm": round(b, 1),
                                    "len_px": int(e - s)})
        return out

    out = scan(0)
    sm = kc.get("center_shift_px", 0)
    if sm:
        # 帯の中心が通り芯から少しずれた腰壁（プラン1 2F 階段の上 y=2730：1.5px。user 11/18 の 90）は、
        # 見つかった腰壁の端から L字に続くものだけ足す（どこでもずれを許すと、家具や文字の白い隙間まで拾う）
        J = kc.get("corner_join_mm", 200)
        # 続く先は、作図に使う長さ（min_len_mm 以上）の腰壁だけ（短いものは浴槽の縁などのことがある）
        longs = [r for r in out if r["b_mm"] - r["a_mm"] >= kc["min_len_mm"]]
        ends = [((r["a_mm"], r["line_mm"]) if r["horizontal"] else (r["line_mm"], r["a_mm"]), r["horizontal"]) for r in longs]
        ends += [((r["b_mm"], r["line_mm"]) if r["horizontal"] else (r["line_mm"], r["b_mm"]), r["horizontal"]) for r in longs]
        for r in scan(sm):
            if any(q["horizontal"] == r["horizontal"] and abs(q["line_mm"] - r["line_mm"]) < 1
                   and min(q["b_mm"], r["b_mm"]) - max(q["a_mm"], r["a_mm"]) > 0 for q in out):
                continue
            mine = [(r["a_mm"], r["line_mm"]), (r["b_mm"], r["line_mm"])] if r["horizontal"] else [(r["line_mm"], r["a_mm"]), (r["line_mm"], r["b_mm"])]
            if any(h != r["horizontal"] and abs(p[0] - m[0]) <= J and abs(p[1] - m[1]) <= J for p, h in ends for m in mine):
                out.append(dict(r, shifted=True))
    json.dump(out, open(prefix + "_腰壁.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in out:
        print("腰壁", "横" if k["horizontal"] else "縦", k["line_mm"], k["a_mm"], k["b_mm"], k["len_px"])


if __name__ == "__main__":
    main()
