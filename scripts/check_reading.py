"""読み取りの検査：できた図面（壁・建具）を元のプランの画像と突き合わせ、読み違いの疑いを探す。
（クロが入れない配布版で、人に「ここを見て」と出すため。user 11/15）

usage: check_reading.py <plan.png> <中間データの階のprefix（例 work/中間データ/1F）> <config.json> [--fix] [--out <フォルダ>]
       check_reading.py --all     全10フロア（work/中間データ）を検査し、work/検査 に結果を書く（確かめ用）
結果は <prefix>_読み取りの検査.json（疑いの一覧と自動修正）と、--out があれば <フォルダ>/<階>_読み取りの検査.png

見ること（「疑い」として人に見てもらう。--fix の時だけ、確実なもの（R7）を中間データで直す）：
  R1 壁の帯がない   読み取った壁なのに、元の画像のその位置に壁の帯（灰色の塗り）がない（建具の見落とし・余分な壁）
  R2 建具が帯の上   読み取った建具なのに、元の画像では壁の帯が途切れずに続いている（建具の読み違い）
  R3 壁の見落とし   元の画像に壁の帯（両側を黒い線ではさまれ、壁と同じ太さ・色で、端が壁につながる）が
                    あるのに、読み取った壁も建具もない
  R4〜R8 部屋と建具 出入口のない部屋・ドアの多いトイレ/浴室・便器のないトイレ・室内の窓・玄関ドアのない玄関
文字が重なっている所は見ない（文字を消した画像との差で分かる）。
確かめ（11/15）：正しい10フロアで疑い 6（家具・階段などの見間違い）。わざと入れた読み違いは
ドアを消す 19/20・壁を消す 17/20・ドアを窓に 9/10 を見つけた。正しいはずの図から、プラン5 2F の
洋室(C)のドアが窓になっていた本当の読み違いを見つけた（R7）。
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ezjww  # noqa: E402
import jp_text  # noqa: E402
from opening_detect import Mapper, profiles, wall_at  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

DEBUG = bool(__import__("os").environ.get("CR_DEBUG"))
HALF = 9            # 壁の帯を見る幅（px、壁芯から両側）
END_SKIP = 120      # 壁の端（柱・交わる壁）の近くは見ない（mm）
MIN_GAP = 350       # R1：帯のない所がこれ以上続いたら（mm）
MIN_BAND = 350      # R3：帯がこれ以上続いたら（mm）
OPEN_FILL = 0.85    # R2：建具の幅のうち、これ以上が帯なら


def load_rgb(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def runs(mask):
    m = np.concatenate(([0], mask.astype(int), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def close_gaps(mask, n=8):
    """n px 以下の途切れをつなぐ（帯の中の汚れ・細い線で切れないように）"""
    m = mask.copy()
    for i, j in runs(~mask):
        if i > 0 and j < len(mask) and j - i <= n:
            m[i:j] = True
    return m


def band_thin(img, gray_range, c, positions, horizontal, wide=16):
    """帯が細いか（家具などの広い灰色の面でない）：壁芯から wide px 離れた両側が灰色でない"""
    g, _, _ = profiles(img, gray_range, c, positions, horizontal, wide)
    return ~(g[:, 0] & g[:, -1])


def centred_band(gray, red, half, tmin=5, tmax=16, off=4, with_t=False, dark=None):
    """壁芯に中心がそろった壁の帯か：芯を通る灰色の連なりの太さが tmin〜tmax px、中心が芯から off px 以内"""
    fill = gray | red
    out = np.zeros(len(fill), bool)
    th = np.zeros(len(fill))
    for i, row in enumerate(fill):
        ks = [k for k in range(half - 2, half + 3) if row[k]]
        if not ks:
            continue
        a = e = ks[0]
        while a > 0 and row[a - 1]:
            a -= 1
        while e < len(row) - 1 and row[e + 1]:
            e += 1
        t = e - a + 1
        out[i] = tmin <= t <= tmax and abs((a + e) / 2 - half) <= off and a > 0 and e < len(row) - 1
        if out[i] and dark is not None:
            # 壁の帯は両側を黒い輪郭線ではさまれている（カウンターなどの破線・家具の面と分ける）
            d = dark[i]
            out[i] = bool(d[max(0, a - 2):a].any() and d[e + 1:e + 3].any())
        th[i] = t
    return (out, th) if with_t else out


def model(jww, meta, cfg):
    """図面の壁芯（mm の線分）"""
    S, (cx, cy) = meta["scale"], meta["center_mm"]
    cl = cfg["layers"]["center"]["layer"]
    out = []
    for e in ezjww.read_document(str(jww))["entities"]:
        if e["type"] == "LINE" and e["base"]["layer"] == cl:
            p = (e["start_x"] * S + cx, e["start_y"] * S + cy)
            q = (e["end_x"] * S + cx, e["end_y"] * S + cy)
            if abs(p[1] - q[1]) < 1:
                out.append((True, round(p[1], 1), min(p[0], q[0]), max(p[0], q[0])))
            elif abs(p[0] - q[0]) < 1:
                out.append((False, round(p[0], 1), min(p[1], q[1]), max(p[1], q[1])))
    return out


def check(img_path, pre, jww, cfg):
    det = json.load(open(f"{pre}_walls.json", encoding="utf-8"))
    meta = json.load(open(f"{pre}_meta.json", encoding="utf-8"))
    ops = json.load(open(f"{pre}_openings.json", encoding="utf-8"))
    kosh = json.load(open(f"{pre}_腰壁.json", encoding="utf-8")) if Path(f"{pre}_腰壁.json").exists() else []
    img = load_rgb(img_path)
    clean_p = Path(f"{pre}_文字削除後.png")
    text = np.zeros(img.shape[:2], bool)
    if clean_p.exists():
        clean = load_rgb(clean_p)
        if clean.shape == img.shape:
            text = np.abs(img.astype(int) - clean.astype(int)).sum(axis=2) > 60
            text = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    M = Mapper(det, cfg["input"]["grid_mm"])
    gr = det["gray_range"]
    H, W = img.shape[:2]
    walls = []  # 読み取った壁（walls.json）：(horizontal, line, a, b)
    for w in det["walls"]:
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        if abs(y1 - y2) < 1:
            walls.append((True, float(y1), min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) < 1:
            walls.append((False, float(x1), min(y1, y2), max(y1, y2)))
    # 建具（開口）と腰壁の範囲：(horizontal, line, a, b)
    opens = []
    for o in ops:
        if o.get("rejected") and o.get("rejected") != "joinery":
            continue  # 浴室の戸など（joinery）は部品でなく別の側で描くので、開口として扱う
        h = o["horizontal"]
        line = o["p1_mm"][1] if h else o["p1_mm"][0]
        a, b = (o["p1_mm"][0], o["p2_mm"][0]) if h else (o["p1_mm"][1], o["p2_mm"][1])
        opens.append((h, line, min(a, b), max(a, b), o["kind"]))
    kos = [(k["horizontal"], k["line_mm"], k["a_mm"], k["b_mm"]) for k in kosh]

    def px_of(h, line, t):
        return (M.x(t), M.y(line)) if h else (M.x(line), M.y(t))

    def sample(h, line, a, b, strict=False):
        """a〜b（mm）を 1px ごとに見る：(t_mm の列, 帯あり, 見ない所)"""
        c = M.y(line) if h else M.x(line)
        pa, pb = (M.x(a), M.x(b)) if h else (M.y(a), M.y(b))
        lo, hi = int(math.ceil(min(pa, pb))), int(math.floor(max(pa, pb)))
        if hi - lo < 2 or not (HALF + 17 <= c < (H if h else W) - HALF - 17):
            return None
        pos = np.arange(lo, hi + 1)
        lim = W if h else H
        pos = pos[(pos >= 0) & (pos < lim)]
        if len(pos) < 2:
            return None
        g, _, r = profiles(img, gr, c, pos, h, HALF)
        if strict:
            # 壁の塗りにはむらがある（161〜205 など）ので、灰色の幅を少し広げて見る
            g2, d2, r2 = profiles(img, (gr[0] - 20, gr[1] + 12), c, pos, h, 14)
            band, th = centred_band(g2, np.zeros_like(g2), 14, with_t=True, dark=d2)  # 赤（家具の茶色も入る）は使わない。文字は別に見る
        else:
            band = wall_at(g, r, HALF) & band_thin(img, gr, c, pos, h)
        ci = int(round(c))
        tx = text[ci, pos] if h else text[pos, ci]
        tmm = np.array([M.xmm(p) if h else M.ymm(p) for p in pos])
        if strict:
            colr = np.array([img[ci, p] if h else img[p, ci] for p in pos]).astype(int)
            return tmm, band, tx, th, colr
        return tmm, band, tx

    issues = []
    mmpp = det["mm_per_px"]

    def covered(h, line, t, items, pad):
        return any(hh == h and abs(ll - line) < 1 and a - pad <= t <= b + pad for hh, ll, a, b, *_ in items)

    # 外部の部屋（バルコニーなど）：手すり壁は帯でなく線で描かれることがあるので R1 で見ない
    from shapely.geometry import LineString as _LS, Point as _Pt
    from shapely.ops import polygonize as _pz, unary_union as _uu
    _rules = room_rules()
    _polys = [p for p in _pz(_uu([_LS([(a, l), (b, l)] if h else [(l, a), (l, b)]) for h, l, a, b in walls])) if p.area > 0.3e6]
    _np = Path(f"{pre}_部屋名.json")
    _outer = []
    for n in (json.load(open(_np, encoding="utf-8")) if _np.exists() else []):
        if n.get("name") in _rules and _rules[n["name"]][0] == "外部":
            pt = _Pt(M.xmm(n["x"]), M.ymm(n["y"]))
            _outer += [p for p in _polys if p.contains(pt)]

    def by_outer(h, line, t):
        return any(p.contains(_Pt(t, line + s_ * 200) if h else _Pt(line + s_ * 200, t)) for p in _outer for s_ in (-1, 1))

    # R1・R2：図面の壁芯に沿って
    for h, line, a, b in walls:
        s = sample(h, line, a, b)
        if s is None:
            continue
        tmm, band, tx = s
        inside = (tmm >= a + END_SKIP) & (tmm <= b - END_SKIP)
        op = np.array([covered(h, line, t, opens, 60) for t in tmm])
        ko = np.array([covered(h, line, t, [(x[0], x[1], x[2], x[3]) for x in kos], 60) for t in tmm])
        miss = inside & ~band & ~op & ~ko & ~tx
        for i, j in runs(miss):
            L = abs(tmm[j - 1] - tmm[i])
            if L >= MIN_GAP:
                m = (tmm[i] + tmm[j - 1]) / 2
                if by_outer(h, line, m):
                    continue
                issues.append(("R1", f"壁なのに元の図に壁の帯がない（{L:.0f}mm）。建具の見落としか、余分な壁", h, line, tmm[i], tmm[j - 1]))
    for h, line, a, b, kind in opens:
        if kind == "window":
            continue  # 窓は帯の上に描く図もあるので見ない
        s = sample(h, line, a + 60, b - 60)
        if s is None:
            continue
        tmm, band, tx = s
        ok = ~tx
        if ok.sum() < 10:
            continue
        fill = (band & ok).sum() / ok.sum()
        if fill >= OPEN_FILL:
            issues.append(("R2", f"建具（{kind}）なのに、元の図では壁の帯が続いている（{fill:.0%}）", h, line, a, b))
    # この図の壁の帯の太さと色（読み取った室内壁の帯から）。家具・浴槽の縁など、太さや色の違う帯を除くため
    ts, cs = [], []
    for h, line, a, b in walls:
        s = sample(h, line, a + END_SKIP, b - END_SKIP, strict=True)
        if s is None:
            continue
        tmm, band, tx, th, colr = s
        op = np.array([covered(h, line, t, opens, 60) for t in tmm])
        k = band & ~op & ~tx
        ts += list(th[k])
        cs += list(colr[k])
    typ_t = float(np.median(ts)) if ts else 9.0
    typ_c = np.median(np.array(cs), axis=0) if cs else None
    # R3：壁の見落とし（建物の範囲の 227.5 ごとの線）
    xs = [p for w in walls for p in ((w[2], w[3]) if w[0] else (w[1],))]
    ys = [p for w in walls for p in ((w[1],) if w[0] else (w[2], w[3]))]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    hg = cfg["input"]["half_grid_mm"]
    for h in (True, False):
        lo, hi = (y0, y1) if h else (x0, x1)
        a, b = (x0, x1) if h else (y0, y1)
        k = math.ceil(lo / hg - 1e-6)
        while k * hg <= hi + 1e-6:
            line = round(k * hg, 1)
            k += 1
            s = sample(h, line, a, b, strict=True)
            if s is None:
                continue
            tmm, band, tx, th, colr = s
            mine = [(w[0], w[1], w[2], w[3]) for w in walls]
            cov = np.array([covered(h, line, t, mine, 80) or covered(h, line, t, opens, 80) for t in tmm])
            ko = np.array([covered(h, line, t, kos, 60) for t in tmm])
            # 文字の所と、数 px の途切れは帯ありとみなしてつなぐ（短く切れて見えないように）
            raw = band
            band = close_gaps(band | tx)
            extra = band & ~cov & ~ko
            for i, j in runs(extra):
                L = abs(tmm[j - 1] - tmm[i])
                if DEBUG and L >= 300:
                    print("  R3候補", h, line, round(tmm[i]), round(tmm[j - 1]), "L", round(L), "raw", raw[i:j].mean().round(2),
                          "tx", tx[i:j].mean().round(2), "th", np.median(th[i:j][raw[i:j]]) if raw[i:j].any() else 0, "typ", typ_t,
                          "col", np.median(colr[i:j][raw[i:j]], axis=0) if raw[i:j].any() else None, typ_c)
                if L < MIN_BAND:
                    continue
                seen = (~tx[i:j]).sum()
                if seen < 0.5 * (j - i) or raw[i:j].sum() < 0.6 * seen:
                    continue  # 文字の所ばかり、または本当に帯のある所が少ない
                kk = raw[i:j] & (th[i:j] > 0)
                if kk.sum() < 5 or np.median(th[i:j][kk]) > typ_t + 3:
                    continue  # 壁より太い帯（浴槽の縁・カウンターなど）
                cc = colr[i:j][kk]
                if typ_c is not None and np.abs(np.median(cc, axis=0) - typ_c).max() > 12:
                    continue  # 壁と色の違う帯（家具など）
                # 壁は端で他の壁とつながる（T・L 字）。少なくとも片方の端が、直交する壁か同じ線の壁・建具の近く（200mm）にあること
                # （キッチンのカウンター・家具の縁は、端が部屋の中で切れる）
                def attached(t):
                    if any(hh != h and abs(ll - t) <= 200 and aa - 60 <= line <= bb + 60 for hh, ll, aa, bb in mine):
                        return True
                    near = np.abs(tmm - t) <= 200
                    return bool((cov & near).any())
                if not (attached(tmm[i]) or attached(tmm[j - 1])):
                    continue
                if True:
                    issues.append(("R3", f"元の図に壁の帯（{L:.0f}mm）があるのに、図面に壁がない", h, line, tmm[i], tmm[j - 1]))
    issues += room_checks(pre, det, M, walls, ops, cfg)
    issues = [(c, m, h, line, float(min(a, b)), float(max(a, b))) for c, m, h, line, a, b in issues]
    return issues, img, M


DOOR_KINDS = {"swing", "sliding", "sliding_pocket", "storage", "folding", "ub_door", "entrance_door", "joinery",
              "double_swing", "outset"}


def room_rules():
    """部屋名リスト.csv：{部屋名: (種類, 最少のドア数, ちょうど1枚か)}"""
    import csv
    out = {}
    for r in csv.DictReader(open(ROOT / "05_設定" / "部屋名リスト.csv", encoding="utf-8-sig")):
        n = r["ドア枚数の目安"].strip()
        need = 1 if n.startswith("1") else 0
        out[r["部屋名"]] = (r["種類"], need, n == "1" and "必ず1枚" in r["確認ルール"])
    return out


def room_checks(pre, det, M, walls, ops, cfg):
    """部屋と建具の突き合わせ（壁で囲まれた区画ごと）
      R4 出入口がない  ドアが要る部屋（部屋名リストの「ドア枚数の目安」が1以上）の区画に、ドアも開口もない
      R5 ドアが多い    トイレ・浴室だけの区画に、ドアが2枚以上
      R6 便器がない    トイレの区画に便器がない
      R7 室内の窓      窓が室内の壁にある、または窓の両側がどちらも室内の部屋
      R8 玄関ドアがない 玄関の区画に玄関ドアがない"""
    from shapely.geometry import LineString, Point
    from shapely.ops import polygonize, unary_union
    rules = room_rules()
    segs = [LineString([(a, line), (b, line)] if h else [(line, a), (line, b)]) for h, line, a, b in walls]
    polys = [p for p in polygonize(unary_union(segs)) if p.area > 0.3e6]
    names_p = Path(f"{pre}_部屋名.json")
    names = json.load(open(names_p, encoding="utf-8")) if names_p.exists() else []
    rooms = [{"poly": p, "names": set(), "doors": 0, "access": 0, "toilet": False, "entrance": False} for p in polys]
    wall_types = []
    for w in det["walls"]:
        (x1, y1), (x2, y2) = w["p1"], w["p2"]
        hh = abs(y1 - y2) < 1
        wall_types.append((w, (hh, y1 if hh else x1, min(x1, x2) if hh else min(y1, y2), max(x1, x2) if hh else max(y1, y2))))

    def room_at(x, y):
        pt = Point(x, y)
        for r in rooms:
            if r["poly"].contains(pt):
                return r
        return None
    for n in names:
        if not n.get("name") or n.get("kind") == "除外語" or n["name"] not in rules:
            continue
        r = room_at(M.xmm(n["x"]), M.ymm(n["y"]))
        if r:
            r["names"].add(n["name"])
    tp = Path(f"{pre}_便器.json")
    for t in (json.load(open(tp, encoding="utf-8")) if tp.exists() else []):
        r = room_at(t["tip_mm"][0] - t["front"][0] * 300, t["tip_mm"][1] - t["front"][1] * 300)
        if r:
            r["toilet"] = True
    out = []
    for o in ops:
        if o.get("rejected") and o.get("rejected") != "joinery":
            continue
        h = o["horizontal"]
        line = o["p1_mm"][1] if h else o["p1_mm"][0]
        a, b = sorted((o["p1_mm"][0], o["p2_mm"][0]) if h else (o["p1_mm"][1], o["p2_mm"][1]))
        m = (a + b) / 2
        sides = [room_at(m, line + s * 200) if h else room_at(line + s * 200, m) for s in (-1, 1)]
        for r in {id(x): x for x in sides if x}.values():
            r["access"] += 1  # 出入口（窓・浴室の戸のかけらも入れる。外部の部屋は掃き出し窓から出る）
            if not o.get("rejected") and o["kind"] in DOOR_KINDS and o["kind"] != "ub_door":
                r["doors"] += 1  # ドアの枚数（浴室の戸はかけらが多いので数えない）
        if o["kind"] == "window":
            kinds = [{rules[n][0] for n in r["names"]} if r else None for r in sides]
            inner = {"居室", "水回り", "収納", "玄関・動線"}
            # 両側とも建物の中の区画で、外部・吹抜などの名前がなく、少なくとも片側は室内の名前
            on_inner = any(w["type"] == "interior" and wl_h == h and abs(wl_l - line) < 1 and wl_a - 1 <= m <= wl_b + 1
                           for w, (wl_h, wl_l, wl_a, wl_b) in wall_types)
            if on_inner:
                out.append(("R7", "窓が室内の壁にある。窓でなく室内の建具か", h, line, a, b))
            elif (all(k is not None and k <= inner for k in kinds) and any(kinds)
                    and sides[0] is not sides[1]):
                nm = "・".join(sorted(sides[0]["names"] | sides[1]["names"])) or "名前なし"
                out.append(("R7", f"窓の両側が、どちらも室内の部屋（{nm}）。窓でなく室内の建具か", h, line, a, b))
            continue
        if o["kind"] == "entrance_door" and not o.get("rejected"):
            for r in sides:
                if r:
                    r["entrance"] = True
    for r in rooms:
        if not r["names"]:
            continue
        need = max(rules[n][1] for n in r["names"])
        c = r["poly"].representative_point()
        bx = r["poly"].bounds
        where = (True, c.y, c.x - 150, c.x + 150)
        label = "・".join(sorted(r["names"]))
        if need and r["access"] == 0:
            out.append(("R4", f"{label}：ドアが要る部屋なのに、出入口が1つもない（建具の見落としか、余分な壁）", *where))
        if len(r["names"]) == 1 and rules[next(iter(r["names"]))][2] and r["doors"] > 1:
            out.append(("R5", f"{label}：ドアは1枚のはずが {r['doors']} 枚", *where))
        if "玄関" in r["names"] and not r["entrance"]:
            out.append(("R8", f"{label}：玄関ドアがない（玄関ドアの読み違いか）", *where))
        if "トイレ" in r["names"] and not r["toilet"]:
            out.append(("R6", f"{label}：便器が見つからない", *where))
    return out


def draw(img, M, issues, out, title):
    vis = (img[:, :, ::-1].astype(np.float32) * 0.5 + 127).astype(np.uint8)
    cols = {"R1": (0, 0, 230), "R2": (230, 0, 160), "R3": (0, 150, 255), "R4": (0, 160, 0), "R5": (160, 100, 0),
            "R6": (120, 0, 200), "R7": (200, 120, 0), "R8": (0, 120, 120)}
    labels = []
    for n, (code, msg, h, line, a, b) in enumerate(issues, 1):
        p = (M.x(a), M.y(line)) if h else (M.x(line), M.y(a))
        q = (M.x(b), M.y(line)) if h else (M.x(line), M.y(b))
        p, q = tuple(int(round(v)) for v in p), tuple(int(round(v)) for v in q)
        cv2.line(vis, p, q, cols[code], 5)
        cv2.circle(vis, ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2), 22, cols[code], 2)
        labels.append((f"{n} {code}", ((p[0] + q[0]) // 2 + 24, (p[1] + q[1]) // 2 - 10), cols[code], 18))
    vis = jp_text.put(vis, [(title, (10, 8), (0, 0, 0), 24)] + labels)
    # 下に、疑いごとの説明（番号・記号・中身）を並べる
    if issues:
        Wv = vis.shape[1]
        size = max(16, min(22, Wv // 60))
        per = max(20, int((Wv - 20) / (size * 1.02)))  # 1行の文字数
        lines = []
        for n, (code, msg, *_r) in enumerate(issues, 1):
            t = f"{n} [{code}] {msg}"
            lines += [t[k:k + per] for k in range(0, len(t), per)]
        pad = np.full((len(lines) * (size + 10) + 20, Wv, 3), 255, np.uint8)
        pad = jp_text.put(pad, [(t, (10, 10 + i * (size + 10)), (0, 0, 0), size) for i, t in enumerate(lines)])
        vis = np.vstack([vis, pad])
    cv2.imencode(".png", vis)[1].tofile(str(out))


FLOORS = [("", "プラン1", "p2_x31_1504x1504.png", "p3_x51_1504x1289.png"),
          ("プラン2", "プラン2", "p2_x39_2507x1504.png", "p3_x60_1934x1433.png"),
          ("プラン3", "プラン3", "p2_x31_4370x3725.png", "p3_x52_1862x1791.png"),
          ("プラン4", "プラン4", "p2_x32_1648x2579.png", "p3_x52_1504x1791.png"),
          ("プラン5", "プラン5", "p2_x31_3081x4155.png", "p3_x52_931x2077.png")]


def fix(pre, issues):
    """確実に直せる読み違いだけを、中間データ（json）で直す（user 11/15：配布版はクロが直せないので）。
      R7 室内の窓 → ドア（円弧があれば片開き、戸袋があれば片引戸、なければ引戸）。その壁が外壁と判定されて
                   いたら室内壁に直す（外壁と読み違えた壁の上のドアが窓になっていた。プラン5 2F 洋室(C)）
    直す前の json は <階>_openings_自動修正前.json などに残す。直したものの説明の一覧を返す"""
    import shutil
    targets = [x for x in issues if x[0] == "R7"]
    if not targets:
        return []
    op_p, wl_p = Path(f"{pre}_openings.json"), Path(f"{pre}_walls.json")
    ops = json.load(open(op_p, encoding="utf-8"))
    det = json.load(open(wl_p, encoding="utf-8"))
    done = []
    for code, msg, h, line, a, b in targets:
        for o in ops:
            if o.get("rejected") or o["kind"] != "window" or o["horizontal"] != h:
                continue
            ol = o["p1_mm"][1] if h else o["p1_mm"][0]
            oa, ob = sorted((o["p1_mm"][0], o["p2_mm"][0]) if h else (o["p1_mm"][1], o["p2_mm"][1]))
            if abs(ol - line) > 1 or abs(oa - a) > 5 or abs(ob - b) > 5:
                continue
            kind = "swing" if (o.get("arc") or 0) >= 0.5 and o.get("hinge_px") is not None else (
                "sliding_pocket" if o.get("pocket") else "sliding")
            o["kind"] = kind
            o["wall_type"] = "interior"
            o.pop("window_cues", None)
            o.pop("window_type", None)
            o.pop("label", None)
            o["note"] = f"自動修正（読み取りの検査 R7）：窓 → {kind}"
            for w in det["walls"]:
                (x1, y1), (x2, y2) = w["p1"], w["p2"]
                if h and abs(y1 - line) < 1 and abs(y2 - line) < 1 and min(x1, x2) - 1 <= (oa + ob) / 2 <= max(x1, x2) + 1 or                         not h and abs(x1 - line) < 1 and abs(x2 - line) < 1 and min(y1, y2) - 1 <= (oa + ob) / 2 <= max(y1, y2) + 1:
                    if w["type"] != "interior":
                        w["type"] = "interior"
                        w.setdefault("flags", []).append("自動修正（読み取りの検査 R7）：外壁 → 室内壁")
                        # 同じ壁の上のほかの建具も室内壁の建具に（室内建具の種類分けの対象になるように）
                        lo_, hi_ = (min(x1, x2), max(x1, x2)) if h else (min(y1, y2), max(y1, y2))
                        for o2 in ops:
                            l2 = o2["p1_mm"][1] if o2["horizontal"] else o2["p1_mm"][0]
                            m2 = (o2["p1_mm"][0] + o2["p2_mm"][0]) / 2 if o2["horizontal"] else (o2["p1_mm"][1] + o2["p2_mm"][1]) / 2
                            if o2["horizontal"] == h and abs(l2 - line) < 1 and lo_ - 1 <= m2 <= hi_ + 1:
                                o2["wall_type"] = "interior"
            at = f"({oa:.0f}〜{ob:.0f}, {line:g})" if h else f"({line:g}, {oa:.0f}〜{ob:.0f})"
            done.append(f"室内の壁の窓を {kind} に直した @{at}")
    if done:
        for src, tag in ((op_p, "openings"), (wl_p, "walls")):
            bak = Path(f"{pre}_{tag}_自動修正前.json")
            if not bak.exists():
                shutil.copy(src, bak)
        json.dump(ops, open(op_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        json.dump(det, open(wl_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return done


def report(issues, fixed):
    for n, (code, msg, h, line, a, b) in enumerate(issues, 1):
        at = f"({a:.0f}〜{b:.0f}, {line:g})" if h else f"({line:g}, {a:.0f}〜{b:.0f})"
        print(f"  {n} [{code}] {msg} @{at}")
    for f in fixed:
        print("  自動修正:", f)


def main():
    import spec
    if sys.argv[1] == "--all":
        cfg = spec.current()
        res = {}
        total = 0
        for sub, pn, i1, i2 in FLOORS:
            for fl, im in (("1F", i1), ("2F", i2)):
                pre = ROOT / "work" / "中間データ" / sub / fl
                img = ROOT / "work" / "images" / sub / im
                issues, image, M = check(img, pre, None, cfg)
                total += len(issues)
                print(f"== {pn} {fl}: 疑い {len(issues)}")
                report(issues, [])
                draw(image, M, issues, ROOT / "work" / "検査" / f"読み取り_{pn}_{fl}.png", f"{pn} {fl} 読み取りの検査：疑い {len(issues)}")
                res[f"{pn} {fl}"] = [dict(code=c, msg=m, horizontal=h, line=l, a=a, b=b) for c, m, h, l, a, b in issues]
        json.dump(res, open(ROOT / "work" / "検査" / "読み取り検査.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("疑いの合計", total)
        return
    # check_reading.py <plan.png> <中間データの階のprefix> <config.json> [--fix] [--out <出力フォルダ>]
    img, pre, cfgp = sys.argv[1:4]
    cfg = spec.load_cfg(cfgp)  # 仕様の json は _extends で default.json を引き継ぐ
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    issues, image, M = check(img, pre, None, cfg)
    fixed = []
    if "--fix" in sys.argv:
        fixed = fix(pre, issues)
        if fixed:
            issues, image, M = check(img, pre, None, cfg)  # 直したあとの疑い
    report(issues, fixed)
    fl = Path(pre).name
    res = {"疑い": [dict(code=c, msg=m, horizontal=h, line=l, a=a, b=b) for c, m, h, l, a, b in issues], "自動修正": fixed}
    json.dump(res, open(f"{pre}_読み取りの検査.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if out:
        draw(image, M, issues, Path(out) / f"{fl}_読み取りの検査.png", f"{fl} 読み取りの検査：疑い {len(issues)}" +
             (f"（自動修正 {len(fixed)}）" if fixed else ""))
    print(f"{fl}: 読み取りの疑い {len(issues)}、自動修正 {len(fixed)}")


if __name__ == "__main__":
    main()
