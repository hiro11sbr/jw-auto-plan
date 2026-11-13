"""玄関収納（造り付け）を、インプット図から見つける（作図は make_plan.furniture_symbols 側）。

見本（ユーザー 11/7〜8）：奥行き約400・幅は400の倍数のユニットが壁ぎわに並ぶ。
低いもの＝天板のハッチング（長手方向の平行線）、背の高いもの＝四角の中に対角線が1本。
玄関・土間（収納）・シューズクローク・玄関に隣接するホールにだけ置く。建具（扉・窓）は先に見つけてあるので、
その範囲は除く。トイレ・洗面・浴室・キッチンの近くのものは除く。

見つけ方（ピクセルをmmに換算して判定）：
 1) ハッチング：長さ200mm以上の平行線が、300mm角の窓の中に数本（間隔50mmほど）ある所
 2) 対角線：45度の線で、長さが 400×√2 ほど（400mm角の対角線）
 3) 400mm角の四角（外枠が3辺以上ある）の中の対角線だけを「トール」とする
 4) 隣り合うものを1本の並び（run）にまとめる。壁（灰色の帯）に接していること

detect() は [{"horizontal": bool, "c_mm": 並びの中心線の座標（横なら y、縦なら x）, "a_mm", "b_mm": 並びの両端,
              "units": [("低"|"トール", t0_mm, t1_mm), ...]}] を返す。

usage: entrance_storage.py <plan.png> <walls.json> <部屋名.json> <openings.json> <config.json> <out.png>
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opening_detect import Mapper  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ALLOW = ("玄関", "土間", "シューズ", "SIC", "SCL")
NEAR_BAD = ("トイレ", "洗面", "UB", "浴室", "ランドリー", "脱衣", "キッチン", "LDK", "WC")
UNIT = 400.0


def hatch_blocks(dark, mp):
    """Blocks of close parallel long lines. Returns [(orient, cx, cy, w, h)] in px; orient 'v' = vertical lines."""
    L = int(200 / mp)
    win = int(300 / mp)
    out = []
    for orient in ("v", "h"):
        k = (1, L) if orient == "v" else (L, 1)
        V = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        D = cv2.boxFilter(V.astype(np.float32), -1, (win, win))
        Hm = (D > 0.14).astype(np.uint8)
        Hm = cv2.morphologyEx(Hm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        n, _lab, st, _ = cv2.connectedComponentsWithStats(Hm)
        for i in range(1, n):
            x, y, w, h, _a = st[i]
            out.append((orient, x + w / 2, y + h / 2, w, h))
    return out


def diagonals(dark, mp):
    """Corner to corner lines of a 400 mm square (either way): a diagonal kernel of the square's size is
    slid over the picture (the line is 1 px wide and steps, so the kernel is 3 px wide). Returns
    [(cx, cy)] in px of the squares."""
    n = int(round(UNIT / mp))
    out = []
    d = dark.astype(np.float32)
    for flip in (False, True):
        k = np.eye(n, dtype=np.float32)
        if flip:
            k = k[:, ::-1].copy()
        k = cv2.dilate(k, np.ones((3, 3), np.uint8))
        k /= k.sum() / 3.0  # a perfect 1 px line scores about 1
        sc = cv2.filter2D(d, -1, k, anchor=(n // 2, n // 2), borderType=cv2.BORDER_CONSTANT)
        # a line, not a filled area: the surroundings of the kernel's line must be mostly empty
        peaks = (sc >= 0.8) & (sc >= cv2.dilate(sc, np.ones((n // 2, n // 2), np.uint8)) - 1e-6)
        ys, xs = np.nonzero(peaks)
        for x, y in zip(xs, ys):
            out.append((float(x), float(y)))
    # one square found several times (neighbouring peaks): keep one
    keep = []
    for c in out:
        if not any(abs(c[0] - q[0]) < n / 3 and abs(c[1] - q[1]) < n / 3 for q in keep):
            keep.append(c)
    return keep


def outline_sides(dark, cx, cy, half):
    """How many of the 4 sides of the square (centre, half side px) are drawn (a dark run along >= 70%)."""
    n = 0
    x0, x1, y0, y1 = int(cx - half), int(cx + half), int(cy - half), int(cy + half)
    H, W = dark.shape
    if x0 < 2 or y0 < 2 or x1 >= W - 2 or y1 >= H - 2:
        return 0
    for sl in (dark[y0 - 2:y0 + 3, x0:x1 + 1].max(axis=0), dark[y1 - 2:y1 + 3, x0:x1 + 1].max(axis=0),
               dark[y0:y1 + 1, x0 - 2:x0 + 3].max(axis=1), dark[y0:y1 + 1, x1 - 2:x1 + 3].max(axis=1)):
        if sl.mean() >= 0.7:
            n += 1
    return n


def detect(img_path, det, names, ops, cfg, verbose=False):
    mp = det["mm_per_px"]
    M = Mapper(det, cfg["input"]["grid_mm"])
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = (g < 110).astype(np.uint8)
    glo, ghi = det["gray_range"]
    gray = ((g >= glo) & (g <= ghi)).astype(np.uint8)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    keys = [n for n in names if n.get("name") and any(a in n["name"] for a in ALLOW)]
    bad = [n for n in names if n.get("name") and any(b in n["name"] for b in NEAR_BAD)]
    if not keys:
        return []
    # openings (found first): a box along each, +-250 mm across
    boxes = []
    for o in ops:
        if o.get("rejected"):
            continue
        (x1, y1), (x2, y2) = o["p1_mm"], o["p2_mm"]
        px1, px2 = sorted((M.x(x1), M.x(x2)))
        py1, py2 = sorted((M.y(y1), M.y(y2)))
        pad = 60 / mp  # (the storage stands right beside a window's wall: only the opening's own band)
        boxes.append((px1 - 4, py1 - pad, px2 + 4, py2 + pad) if o["horizontal"] else (px1 - pad, py1 - 4, px2 + pad, py2 + 4))

    def ok_place(cx, cy):
        d_ok = min(math.hypot(n["x"] - cx, n["y"] - cy) for n in keys)
        d_bad = min((math.hypot(n["x"] - cx, n["y"] - cy) for n in bad), default=1e9)
        if d_ok * mp > 2800 or d_bad < d_ok * 0.9:
            return False
        return not any(a <= cx <= c and b <= cy <= d for a, b, c, d in boxes)

    def plain_floor(cx, cy, hw, hh):
        """The inside shows the floor (gray tiles or white): a filled colour (the orange of a 物入, a pink room) is
        something else. 玄関収納 in a ウォークインホーム is not filled (user 11/7)."""
        x0, x1, y0, y1 = int(cx - hw), int(cx + hw), int(cy - hh), int(cy + hh)
        a = sat[max(0, y0):y1, max(0, x0):x1]
        m = dark[max(0, y0):y1, max(0, x0):x1] == 0
        if not m.any():
            return False
        return (a[m] > 60).mean() < 0.25

    def touches_wall(cx, cy, hw, hh):
        """A gray band (>= 60%) in a thin strip along any of the 4 sides, just outside the box."""
        t = max(3, int(40 / mp))
        x0, x1, y0, y1 = int(cx - hw), int(cx + hw), int(cy - hh), int(cy + hh)
        for sl in (gray[max(0, y0 - t - 4):max(0, y0 - 3), x0:x1], gray[y1 + 3:y1 + t + 4, x0:x1],
                   gray[y0:y1, max(0, x0 - t - 4):max(0, x0 - 3)], gray[y0:y1, x1 + 3:x1 + t + 4]):
            if sl.size and sl.mean() >= 0.6:
                return True
        return False

    units = []  # (kind, cx, cy, horizontal, length_mm)
    for orient, cx, cy, w, h in hatch_blocks(dark, mp):
        wm, hm = w * mp, h * mp
        s, l = min(wm, hm), max(wm, hm)
        if not (300 <= s <= 600 and 330 <= l <= 2000):
            continue
        k = max(1, round((l + 60) / UNIT))  # (the measured box is a little smaller than the unit)
        horizontal = (w > h) if k > 1 else (orient == "h")
        if not ok_place(cx, cy):
            continue
        hw, hh = ((k * UNIT / 2) / mp, (UNIT / 2) / mp) if horizontal else ((UNIT / 2) / mp, (k * UNIT / 2) / mp)
        if not touches_wall(cx, cy, hw, hh) or not plain_floor(cx, cy, hw, hh):
            continue
        units.append(("低", cx, cy, horizontal, k * UNIT))
        if verbose:
            print("ハッチング", round(M.xmm(cx)), round(M.ymm(cy)), "横" if horizontal else "縦", k * UNIT)
    half = UNIT / 2 / mp
    dark_d = (g < 170).astype(np.uint8)  # (the diagonal is a light, stepped line)
    for cx, cy in diagonals(dark_d, mp):
        # the kernel's peak is flat along the line: move the square to where its outline fits best
        best = (outline_sides(dark, cx, cy, half), 0.0, cx, cy)
        for dx in range(-9, 10):
            for dy in range(-9, 10):
                o_ = outline_sides(dark, cx + dx, cy + dy, half)
                if (o_, -math.hypot(dx, dy)) > best[:2]:
                    best = (o_, -math.hypot(dx, dy), cx + dx, cy + dy)
        cx, cy = best[2], best[3]
        inner = dark_d[int(cy - half) + 4:int(cy + half) - 3, int(cx - half) + 4:int(cx + half) - 3]
        extra = (inner.sum() - 3.0 * inner.shape[0]) / max(1, inner.size)  # ink left once the diagonal is taken away
        why = [nm for nm, bad_ in (("外枠", outline_sides(dark, cx, cy, half) < 3), ("場所", not ok_place(cx, cy)),
                                   ("中身", extra > 0.05),
                                   ("壁", not touches_wall(cx, cy, half, half)), ("色", not plain_floor(cx, cy, half, half))) if bad_]
        if why:
            if verbose and not why == ["場所"]:
                print("対角線を除く", round(M.xmm(cx)), round(M.ymm(cy)), why)
            continue
        if any(abs(cx - u[1]) < (u[4] / 2 / mp if u[3] else half) and abs(cy - u[2]) < (half if u[3] else u[4] / 2 / mp) for u in units if u[3] is not None):
            continue
        units.append(("トール", cx, cy, None, UNIT))
        if verbose:
            print("対角線", round(M.xmm(cx)), round(M.ymm(cy)))
    # runs: units side by side along one line. A tall unit has no direction of its own: it joins a run
    # whose end it touches (the hatched part gives the direction)
    runs = []
    for kind, cx, cy, horizontal, ln in [u for u in units if u[3] is not None]:
        hz = horizontal
        t = cx if hz else cy
        c_line = cy if hz else cx
        for r in runs:
            if r["horizontal"] == hz and abs(c_line - r["c"]) < 60 / mp and t - ln / 2 / mp <= r["b"] + 100 / mp \
                    and t + ln / 2 / mp >= r["a"] - 100 / mp:
                r["a"], r["b"] = min(r["a"], t - ln / 2 / mp), max(r["b"], t + ln / 2 / mp)
                r["units"].append((kind, t - ln / 2 / mp, t + ln / 2 / mp))
                break
        else:
            runs.append({"horizontal": hz, "c": c_line, "a": t - ln / 2 / mp, "b": t + ln / 2 / mp,
                         "units": [(kind, t - ln / 2 / mp, t + ln / 2 / mp)]})
    for kind, cx, cy, horizontal, ln in [u for u in units if u[3] is None]:
        joined = False
        for r in runs:
            hz = r["horizontal"]
            t = cx if hz else cy
            c_line = cy if hz else cx
            if abs(c_line - r["c"]) < 60 / mp and (abs(t - half - r["b"]) < 100 / mp or abs(t + half - r["a"]) < 100 / mp):
                r["a"], r["b"] = min(r["a"], t - half), max(r["b"], t + half)
                r["units"].append((kind, t - half, t + half))
                joined = True
                break
        if not joined:
            # alone: it runs along the wall that touches it
            t = half
            sides = {"上": gray[int(cy - half) - 12:int(cy - half) - 3, int(cx - half):int(cx + half)].mean(),
                     "下": gray[int(cy + half) + 3:int(cy + half) + 12, int(cx - half):int(cx + half)].mean(),
                     "左": gray[int(cy - half):int(cy + half), int(cx - half) - 12:int(cx - half) - 3].mean(),
                     "右": gray[int(cy - half):int(cy + half), int(cx + half) + 3:int(cx + half) + 12].mean()}
            wall = max(sides, key=sides.get)
            hz = wall in ("上", "下")
            t = cx if hz else cy
            c_line = cy if hz else cx
            for r in runs:  # a lone tall unit beside another lone one
                if r["horizontal"] == hz and abs(c_line - r["c"]) < 60 / mp and (abs(t - half - r["b"]) < 100 / mp or abs(t + half - r["a"]) < 100 / mp):
                    r["a"], r["b"] = min(r["a"], t - half), max(r["b"], t + half)
                    r["units"].append((kind, t - half, t + half))
                    break
            else:
                runs.append({"horizontal": hz, "c": c_line, "a": t - half, "b": t + half, "units": [(kind, t - half, t + half)]})
    out = []
    for r in runs:
        if all(k == "トール" for k, _a, _b in r["units"]) and len(r["units"]) < 2:
            continue  # a lone tall unit (one diagonal in a square) is too easily something else
        if r["horizontal"]:
            a_mm, b_mm, c_mm = M.xmm(r["a"]), M.xmm(r["b"]), M.ymm(r["c"])
            units_mm = [(k, round(M.xmm(t0)), round(M.xmm(t1))) for k, t0, t1 in r["units"]]
        else:
            a_mm, b_mm, c_mm = M.ymm(r["b"]), M.ymm(r["a"]), M.xmm(r["c"])
            units_mm = [(k, round(M.ymm(t1)), round(M.ymm(t0))) for k, t0, t1 in r["units"]]
        # which side the wall is on (model direction across the run): the gray band along a long side
        hz = r["horizontal"]
        t0_, t1_ = int(r["a"]), int(r["b"])
        c_ = int(r["c"])
        e = int(UNIT / 2 / mp)
        t = max(3, int(40 / mp))
        if hz:
            up = gray[max(0, c_ - e - t - 4):max(0, c_ - e - 3), t0_:t1_].mean() if t1_ > t0_ else 0
            dn = gray[c_ + e + 3:c_ + e + t + 4, t0_:t1_].mean() if t1_ > t0_ else 0
            side = 1 if up >= dn else -1  # image up = model +y
        else:
            lf = gray[t0_:t1_, max(0, c_ - e - t - 4):max(0, c_ - e - 3)].mean() if t1_ > t0_ else 0
            rt = gray[t0_:t1_, c_ + e + 3:c_ + e + t + 4].mean() if t1_ > t0_ else 0
            side = -1 if lf >= rt else 1  # image left = model -x
        out.append({"horizontal": bool(hz), "c_mm": round(c_mm), "a_mm": round(a_mm), "b_mm": round(b_mm), "wall_side": side,
                    "units": sorted(units_mm, key=lambda u: u[1])})
    return out


def main():
    img, walls, names_p, ops_p, cfg_p, out_png = sys.argv[1:7]
    cfg = json.load(open(cfg_p, encoding="utf-8"))
    det = json.load(open(walls, encoding="utf-8"))
    names = json.load(open(names_p, encoding="utf-8"))
    ops = json.load(open(ops_p, encoding="utf-8"))
    ops = ops["openings"] if isinstance(ops, dict) else ops
    res = detect(img, det, names, ops, cfg, verbose=True)
    M = Mapper(det, cfg["input"]["grid_mm"])
    mp = det["mm_per_px"]
    bgr = cv2.imdecode(np.fromfile(img, np.uint8), cv2.IMREAD_COLOR)
    for i, r in enumerate(res, 1):
        if r["horizontal"]:
            p0, p1 = (M.x(r["a_mm"]), M.y(r["c_mm"]) - 200 / mp), (M.x(r["b_mm"]), M.y(r["c_mm"]) + 200 / mp)
        else:
            p0, p1 = (M.x(r["c_mm"]) - 200 / mp, M.y(r["b_mm"])), (M.x(r["c_mm"]) + 200 / mp, M.y(r["a_mm"]))
        cv2.rectangle(bgr, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 0, 255), 2)
        cv2.putText(bgr, str(i), (int(p0[0]), int(p0[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print(i, r)
    cv2.imencode(".png", bgr)[1].tofile(out_png)


if __name__ == "__main__":
    main()
