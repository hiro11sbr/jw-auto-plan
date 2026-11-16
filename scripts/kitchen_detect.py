"""キッチン（I型）の位置・向き・コンロの側を、元の画像から読む（user 11/16「キッチンも置けるように」）。

usage: kitchen_detect.py <plan.png> <中間データの階のprefix> <config.json>
  → <prefix>_キッチン.json : [{"box_mm": [x0, y0, x1, y1], "horizontal", "stove", "back", "depth_mm",
                               "len_mm", "stove_face_mm", "back_wall_mm", "front_mm"}]
  向きは図の向きの単位ベクトル（+x＝右、+y＝上）。
  - stove：キッチンの中心から見て、コンロのある端の向き
  - back：背中側（作業する人の反対側。ダイニング・壁・腰壁の側）の向き
  - front_mm：作業する側の天板の端の線（横向きなら y、縦向きなら x）
  - stove_face_mm：コンロの端の先にある壁の面（壁芯から石膏ボードの面まで）。壁がなければ天板の端

見つけ方（5プラン、11/16）：
  天板＝同じ長さ（2200〜2900mm）の平行な2本の黒い線が 550〜1000mm 離れて向き合い、間が明るい
  （床の木目でも、じゅうたん・車でもない：中の明るさの中央値 150 以上）。中に 900mm 以上離れた
  2つの物（シンクの角の丸い四角、コンロの四角か暗い四角）がある。重なる候補は奥行きの浅い方
  （後ろの腰壁の帯を含めない）。
  コンロの側＝端の先が壁につながっている側（5プランとも、コンロは壁の側の端）。両端とも・どちらも
  壁でなければ、暗い方の端。
  背中側＝長い辺の外で、近い壁（350mm 以内）がある側。どちらもなければ、壁が遠い側（作業する側は
  カップボードの壁に近い）。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402
from opening_detect import Mapper  # noqa: E402


def long_lines(mask, mmpp, horiz):
    """長い黒い線（1500mm 以上）。[(行の始め, 行の終わり, 始め, 終わり)]（画素、行は線と直角の位置）"""
    m = mask if horiz else mask.T
    k = max(3, int(1500 / mmpp))
    op = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    runs = []
    for r in np.flatnonzero(op.any(1)):
        d = np.diff(np.concatenate(([0], op[r].astype(int), [0])))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            runs.append((int(r), int(s), int(e)))
    runs.sort()
    out = []
    for r, s, e in runs:  # 太い線は隣の行をまとめる
        for g in out:
            if r - g[1] <= 1 and abs(s - g[2]) <= 2 and abs(e - g[3]) <= 2:
                g[1] = r
                break
        else:
            out.append([r, r, s, e])
    return out


def detect(img, mmpp):
    """画像の上のキッチン（px）。[{box:(X0,Y0,X1,Y1) 線の中心で, horiz, objs:[長手の位置]}]"""
    v = img.astype(int).mean(2)
    dark = (v < 140).astype(np.uint8)
    cs, _ = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    boxes = [cv2.boundingRect(c) for c in cs]
    cands = []
    for horiz in (True, False):
        L = long_lines(dark, mmpp, horiz)
        for i, a in enumerate(L):
            for b in L[i + 1:]:
                if not 550 <= (b[0] - a[1]) * mmpp <= 1000:
                    continue
                s, e = max(a[2], b[2]), min(a[3], b[3])
                if not 2200 <= (e - s) * mmpp <= 2900:
                    continue
                r0, r1 = a[1], b[0]
                X0, Y0, X1, Y1 = (s, r0, e, r1) if horiz else (r0, s, r1, e)
                inner = v[Y0 + 3:Y1 - 2, X0 + 3:X1 - 2]
                if inner.size == 0 or np.median(inner) < 150:
                    continue
                objs = []
                obj_depth = []  # シンク・コンロの四角の、奥行き方向の位置（0〜1）
                for j, (x, y, w, h) in enumerate(boxes):
                    if x >= X0 and y >= Y0 and x + w <= X1 + 1 and y + h <= Y1 + 1:
                        lo, sh = max(w, h) * mmpp, min(w, h) * mmpp
                        if 400 <= lo <= 1000 and 350 <= sh <= 650 and cv2.contourArea(cs[j]) / (w * h) >= 0.75:
                            objs.append((x + w / 2) if horiz else (y + h / 2))
                            obj_depth.append(((y + h / 2 - Y0) / max(Y1 - Y0, 1)) if horiz else ((x + w / 2 - X0) / max(X1 - X0, 1)))
                fr = ((inner < 150).mean(0) if horiz else (inner < 150).mean(1)) > 0.6
                m_ = np.concatenate(([0], fr.astype(int), [0]))
                for st, en in zip(np.flatnonzero(np.diff(m_) == 1), np.flatnonzero(np.diff(m_) == -1)):
                    if (en - st) * mmpp >= 400:
                        objs.append((X0 if horiz else Y0) + 3 + (st + en) / 2)
                if len(objs) < 2 or (max(objs) - min(objs)) * mmpp < 900:
                    continue
                mid = lambda g: (g[0] + g[1]) / 2  # noqa: E731
                box = (s, mid(a), e, mid(b)) if horiz else (mid(a), s, mid(b), e)
                # 奥行き方向の「暗い所」の重心（0〜1、画像の上／左が0）。ペニンシュラの作業する側を決める
                prof = (inner < 150).mean(1 if horiz else 0)
                cen = float((prof * np.arange(len(prof))).sum() / max(prof.sum(), 1e-9) / max(len(prof) - 1, 1))
                if obj_depth:
                    cen = float(np.median(obj_depth))  # シンク・コンロの四角があれば、それの位置を使う
                cands.append(dict(box=box, horiz=horiz, depth=(b[0] - a[1]) * mmpp, work_cen=cen))
    # 重なる候補は、奥行きの浅い方だけ
    cands.sort(key=lambda c: c["depth"])
    out = []
    for c in cands:
        X0, Y0, X1, Y1 = c["box"]
        if any(min(X1, o["box"][2]) > max(X0, o["box"][0]) and min(Y1, o["box"][3]) > max(Y0, o["box"][1]) for o in out):
            continue
        out.append(c)
    return out, v


def main():
    img_p, pre, cfgp = sys.argv[1:4]
    cfg = spec.load_cfg(cfgp)
    det = json.load(open(f"{pre}_walls.json", encoding="utf-8"))
    mmpp = det["mm_per_px"]
    M = Mapper(det, cfg["input"]["grid_mm"])
    board = cfg["offsets_mm"]["board_face_inside"]
    img = cv2.imdecode(np.fromfile(img_p, np.uint8), cv2.IMREAD_COLOR)
    walls = []
    for w in det["walls"]:
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        if abs(y1 - y2) < 1:
            walls.append((True, y1, min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) < 1:
            walls.append((False, x1, min(y1, y2), max(y1, y2)))
    # 腰壁（キッチンの前の腰壁など）も、長い辺の側の「壁」として使う。I型とペニンシュラの見分けに必要
    # （プラン2：腰壁は壁の一覧になく、腰壁の一覧にだけあった、11/16）
    kp_ = Path(f"{pre}_腰壁.json")
    koshi = []
    if kp_.exists():
        for e in json.load(open(kp_, encoding="utf-8")):
            koshi.append((bool(e["horizontal"]), e["line_mm"], min(e["a_mm"], e["b_mm"]), max(e["a_mm"], e["b_mm"])))
    found, v = detect(img, mmpp)
    out = []
    for c in found:
        X0, Y0, X1, Y1 = c["box"]
        h = c["horiz"]
        xa, xb = sorted((M.xmm(X0), M.xmm(X1)))
        ya, yb = sorted((M.ymm(Y0), M.ymm(Y1)))
        # 長手 t（横なら x）、奥行き u（横なら y）
        ta, tb, ua, ub = (xa, xb, ya, yb) if h else (ya, yb, xa, xb)

        def end_wall(t_edge, sg):
            """長手の端の先（sg の向き）の壁芯（250mm 以内、天板の奥行きの真ん中を通る）"""
            um = (ua + ub) / 2
            c_ = [ln for hh, ln, a, b in walls if hh != h and a - 1 <= um <= b + 1 and -60 <= (ln - t_edge) * sg <= 250]
            return min(c_, key=lambda ln: abs(ln - t_edge)) if c_ else None
        ew = {-1: end_wall(ta, -1), 1: end_wall(tb, 1)}
        if (ew[-1] is None) != (ew[1] is None):
            ss = -1 if ew[-1] is not None else 1
            why = "壁の側の端"
        else:
            n = int((X1 - X0) / 3) if h else int((Y1 - Y0) / 3)
            sub = v[int(Y0) + 3:int(Y1) - 2, int(X0) + 3:int(X1) - 2]
            pa, pb = ((sub[:, :n], sub[:, -n:]) if h else (sub[:n], sub[-n:]))
            first_dark = pa.mean() < pb.mean()
            # 画像の向き（縦は +y が下）→ 図の向き
            ss = (-1 if first_dark else 1) if h else (1 if first_dark else -1)
            why = "暗い方の端"
        stove_face = (ew[ss] - ss * board) if ew[ss] is not None else (ta if ss < 0 else tb)

        def side_wall(u_edge, sg):
            """長い辺の外（sg の向き）で、いちばん近い平行な壁芯（天板の長さの 30% 以上重なる）"""
            c_ = [(ln, a, b) for hh, ln, a, b in walls + koshi if hh == h and (ln - u_edge) * sg >= -60
                  and min(b, tb) - max(a, ta) >= 0.3 * (tb - ta)]
            return min((abs(ln - u_edge) for ln, a, b in c_), default=None), \
                min(c_, key=lambda q: abs(q[0] - u_edge))[0] if c_ else None
        d_lo, w_lo = side_wall(ua, -1)
        d_hi, w_hi = side_wall(ub, 1)
        INF = 1e9
        dl, dh = (d_lo if d_lo is not None else INF), (d_hi if d_hi is not None else INF)
        n_end = sum(1 for s_ in (-1, 1) if ew[s_] is not None)
        if min(dl, dh) <= 350:
            # I型：長い辺の一方が壁（腰壁）に接する。背中＝壁の側
            kind = "I"
            bs = -1 if dl <= dh else 1
        elif n_end == 0:
            # アイランド：4面とも壁に接しない（見本がない。置かない、user 11/16）
            print(f"キッチン {'横' if h else '縦'} ({xa:.0f},{ya:.0f})-({xb:.0f},{yb:.0f})：4面とも壁に接しない（アイランド）。置かない")
            out.append({"box_mm": [round(xa, 1), round(ya, 1), round(xb, 1), round(yb, 1)], "horizontal": h, "kind": "island",
                        "len_mm": round(tb - ta), "depth_mm": round(ub - ua)})
            continue
        else:
            # ペニンシュラ：前面に壁も腰壁もなく、短い辺の一方だけが壁に接する。作業する側（シンク・コンロの側）は
            # 奥行き方向の暗い所の重心が寄っている側。背中＝その反対のフラットカウンター側
            kind = "peninsula"
            cen = c["work_cen"]
            # 横：画像の上＝mm の +y。縦：画像の左＝mm の -x
            work_sgn = (1 if cen < 0.5 else -1) if h else (-1 if cen < 0.5 else 1)
            bs = -work_sgn
        back_wall = (w_lo if bs < 0 else w_hi) if kind == "I" else None
        front = ub if bs < 0 else ua  # 背中の反対側の天板の端＝作業する側の端
        stove = [ss, 0] if h else [0, ss]
        back = [0, bs] if h else [bs, 0]
        item = {"kind": kind, "box_mm": [round(xa, 1), round(ya, 1), round(xb, 1), round(yb, 1)], "horizontal": h,
                "stove": stove, "back": back, "len_mm": round(tb - ta), "depth_mm": round(ub - ua),
                "front_mm": round(front, 1), "stove_face_mm": round(stove_face, 1),
                "back_wall_mm": back_wall, "why_stove": why}
        out.append(item)
        print(f"キッチン {'横' if h else '縦'} ({xa:.0f},{ya:.0f})-({xb:.0f},{yb:.0f}) 長さ{tb - ta:.0f} 奥行き{ub - ua:.0f}"
              f" コンロ{tuple(stove)}（{why}） 背中{tuple(back)}")
    json.dump(out, open(f"{pre}_キッチン.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"キッチン {len(out)}")


if __name__ == "__main__":
    main()
