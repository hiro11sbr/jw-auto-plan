"""出力した .jww を「言われた条件」に照らして機械で検査する（ユーザーに送る前に必ず通す）。

usage: check_conditions.py <図.jww> <_meta.json> [検査画像.png]
       check_conditions.py --all      全10フロア（02_出力）＋プラン1の正解図を検査し、work/検査 に結果を書く

検査する条件（work/言われた条件の一覧.md）：
  A 部品    置き方（meta）どおりの位置に、部品集の線がすべてある
  B 残り    建具のレイヤーに、どの部品のものでもない線が残っていない（古い位置の部品の消し残し）
  C 紫線    石膏ボード納まりの線（色5）が、窓・建具の枠の中にはみ出していない
  D 貫通    壁の線（レイヤー3）が建具を貫通していない
  E 円弧    開き戸の円弧の両端が、扉・枠の線に付いている（中心点のずれ）
  F チリ    収納建具（折戸・両開き）の、開かない側が壁（ボード面65）から飛び出していない
  G 柱      室内建具の両脇のグリッド交点に柱がある（置けない所もあるので「注意」）
  H 青線    室内建具に青（色6）の線がない
  I 環境    環境設定（参考図面の .JWF）の色が入っている
  J 形式    部品がブロックになっていない
  K メーカー 部品は決めたメーカー（docs/メーカー品を使う手順.md）と汎用だけ
  L 窓位置  窓の枠の厚みの真ん中が、付加断熱の外面にある
NG のある所は、まわりを切り出した画像（赤丸つき）を1枚にまとめて書く。
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ezjww  # noqa: E402
import place_fittings as pf  # noqa: E402
import spec  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = HERE.parent
INTERIOR = ("片開き", "片引戸", "引違い戸", "両開き", "折戸", "アウトセット片引戸")
ENTRANCE = ("玄関片開き", "玄関親子")
STORAGE = ("両開き", "折戸")
TOL = 0.8  # mm
BGR = {1: (160, 160, 0), 2: (0, 0, 0), 3: (0, 140, 0), 4: (0, 160, 200), 5: (180, 0, 180), 6: (220, 0, 0),
       7: (0, 120, 200), 8: (120, 120, 120), 9: (170, 170, 170)}


def tr_fn(pl, S):
    ang = math.radians(pl["rot"])
    ca, sa = math.cos(ang), math.sin(ang)

    def tr(u, v):
        u, v = u * S * pl["sx"], v * S * pl["sy"]
        return (pl["pos"][0] + u * ca - v * sa, pl["pos"][1] + u * sa + v * ca)
    return tr


def seg_dist(p, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def check(jww, meta_path, cfg, lib, img_out=None, quiet=False, geom=None):
    """geom に dict を渡すと、直すための材料を入れて返す：rects（枠の四角）、missing（欠けた部品の線）、wl・fl"""
    meta = json.load(open(meta_path, encoding="utf-8"))
    S, (cx, cy) = meta["scale"], meta["center_mm"]
    doc = ezjww.read_document(str(jww))
    L = cfg["layers"]
    fl = L.get("fitting", L["wall"])["layer"]
    wl = L["wall"]["layer"]
    cl = L["center"]["layer"]
    mm = lambda x, y: (x * S + cx, y * S + cy)  # noqa: E731
    lines, arcs = [], []  # (p, q, layer, color) / (centre, r, a0, da, layer, color)
    for e in doc["entities"]:
        b = e["base"]
        if e["type"] == "LINE":
            lines.append((mm(e["start_x"], e["start_y"]), mm(e["end_x"], e["end_y"]), b["layer"], b["pen_color"]))
        elif e["type"] == "ARC":
            arcs.append((mm(e["center_x"], e["center_y"]), e["radius"] * S, e["start_angle"], e["arc_angle"], b["layer"], b["pen_color"]))
    ng, warn = [], []  # (記号, 説明, (x, y))

    fit = [l for l in lines if l[2] == fl]
    FA = np.array([[l[0][0], l[0][1], l[1][0], l[1][1]] for l in fit], float).reshape(-1, 4)
    used = np.zeros(len(fit), bool)
    board = cfg["offsets_mm"]["board_face_inside"]
    added = cfg["offsets_mm"]["exterior_side"]["added_insulation"]
    grid, hc = cfg["input"]["grid_mm"], cfg["column"]["size_mm"] / 2
    # 柱：対角線2本（長さ＝柱の対角）の交点
    diag = cfg["column"]["size_mm"] * math.sqrt(2)
    cols = {(round((p[0] + q[0]) / 2), round((p[1] + q[1]) / 2)) for p, q, la, _ in lines
            if la == wl and abs(math.hypot(q[0] - p[0], q[1] - p[1]) - diag) < 1.0 and abs(abs(q[0] - p[0]) - abs(q[1] - p[1])) < 1.0}
    # 片引戸の開口側の半分（中方立まで＋60）：ここには柱を置けないので G を見ない（user 11/13）
    mull = []
    for pl in meta["placements"]:
        if pl["word"] == "片引戸" and pl["block"] in lib:
            _, _, _, _, W_, off_, _ = lib[pl["block"]]
            t_ = tr_fn(pl, S)
            mull.append((t_(off_ / S, 0), t_((off_ + W_ / 2 + 150) / S, 0)))
    rects = []  # (pl, word, 枠の四角（縮めたもの）, 中心)
    for pl in meta["placements"]:
        name, word = pl["block"], pl["word"]
        if not name.startswith(("汎用_",)):  # 汎用＝配布用の自分で数字を入れる仕様（11/14）
            ng.append(("K", f"メーカー: {name}", tuple(pl["pos"])))
        if name not in lib:
            ng.append(("A", f"部品集にない: {name}", tuple(pl["pos"])))
            continue
        _, ents, _, _, W, off, _ = lib[name]
        tr = tr_fn(pl, S)
        P, Pb = [], []
        for e in ents:
            if e["type"] != "LINE" or (word in INTERIOR and e["base"]["pen_color"] == 6):
                continue
            P.append((tr(e["start_x"], e["start_y"]), tr(e["end_x"], e["end_y"]), e["start_y"] * S, e["end_y"] * S))
            Pb.append(e["base"])
        centre = tr((off + W / 2) / S, 0)
        # A 部品の線がある／B 用の印
        miss, missing = 0, []
        for (p, q, _, _), bb in zip(P, Pb):
            if len(FA) == 0:
                miss += 1
                missing.append((p, q, bb))
                continue
            d1 = np.abs(FA - np.array([p[0], p[1], q[0], q[1]])).max(axis=1)
            d2 = np.abs(FA - np.array([q[0], q[1], p[0], p[1]])).max(axis=1)
            hit = (d1 < TOL) | (d2 < TOL)
            if hit.any():
                used |= hit
            else:
                miss += 1
                missing.append((p, q, bb))
        if miss and miss >= 0.9 * len(P) and len(FA):
            # 丸ごと別の位置にある（ユーザーが正解図で動かした玄関ドアなど）：同じ形がずれて置いてあれば「注意」
            p0, q0 = P[0][0], P[0][1]
            v0 = np.array([q0[0] - p0[0], q0[1] - p0[1]])
            free = np.where(~used)[0]
            cand = [i for i in free if np.abs((FA[i, 2:] - FA[i, :2]) - v0).max() < TOL]
            for i in cand:
                dxy = FA[i, :2] - np.array(p0)
                hits, ok = [], 0
                for p, q, _, _ in P:
                    d1 = np.abs(FA - np.array([p[0] + dxy[0], p[1] + dxy[1], q[0] + dxy[0], q[1] + dxy[1]])).max(axis=1)
                    d2 = np.abs(FA - np.array([q[0] + dxy[0], q[1] + dxy[1], p[0] + dxy[0], p[1] + dxy[1]])).max(axis=1)
                    h = (d1 < TOL) | (d2 < TOL)
                    ok += bool(h.any())
                    hits.append(h)
                if ok >= 0.9 * len(P):
                    for h in hits:
                        used |= h
                    warn.append(("A", f"{word} W{W}: 置き方から ({dxy[0]:.0f},{dxy[1]:.0f}) ずれた位置にある（人が動かした？）", centre))
                    pl = dict(pl, pos=[pl["pos"][0] + dxy[0], pl["pos"][1] + dxy[1]])
                    if "line" in pl:
                        pl["line"] += dxy[1] if pl["horizontal"] else dxy[0]
                    tr = tr_fn(pl, S)
                    centre = tr((off + W / 2) / S, 0)
                    miss, missing = 0, []
                    break
        if geom is not None and miss and miss < 0.9 * len(P):
            geom.setdefault("missing", []).extend(missing)
        if miss:
            ng.append(("A", f"{word} W{W}: 部品の線 {miss}/{len(P)} 本が、置き方の位置にない", centre))
        ys = [y for *_, y0, y1 in P for y in (y0, y1)]
        if not ys:
            continue
        ylo, yhi = min(ys), max(ys)
        # L 窓の位置：枠の室内側の面＝付加断熱の外面−65（枠の厚み130の真ん中が付加断熱の外面）
        if word in pf.WINDOWS and abs(ylo - (added - 65.0)) > 1.0:
            ng.append(("L", f"{word} W{W}: 枠の室内側の面 {ylo:.1f}（付加断熱の外面−65＝{added - 65:g}）", centre))
        # F 収納建具：開かない側（局所 −y）がボード面から出ていない
        if word in STORAGE and ylo < -board - 0.6:
            ng.append(("F", f"{word} W{W}: 開かない側が壁から {-ylo - board:.1f} 出ている", centre))
        # 枠の四角（C・D・H 用）。扉の開いた線は入れない
        if word in pf.WINDOWS:
            r0, r1 = ylo, yhi
            w0, w1 = -board + 5, min(yhi, cfg["offsets_mm"]["exterior_side"]["exterior_face"]) - 3
        else:
            r0, r1 = max(ylo, -board - 12.5), min(yhi, board + 12.5)
            w0, w1 = -board + 5, board - 5
        # 見る幅：片引戸は開口の側（左半分。右半分の戸袋は壁の中）、玄関ドアは枠の内側（左右の耳111は壁にかぶさる）
        x0, x1 = off, off + W
        if word == "片引戸":
            x1 = off + W / 2
        if word == pf.OUTSET:  # アウトセット：開口（枠の外から外）だけ。戸は壁の表面を走る
            j = pf.outset_jambs(ents, S)
            if j:
                x0, x1 = j
        if word in ENTRANCE:
            x0, x1 = off + 111, off + W - 111
        e_ = 2
        if word in pf.WINDOWS:
            # 窓の端（12.5＋少し）は、壁の戻りの紫線が枠の線まで入ってくる所なので見ない（M で見る。11/14）
            e_ = 15
        a, b = tr((x0 + e_) / S, r0 / S), tr((x1 - e_) / S, r1 / S)
        # M 窓：壁の戻りの紫線（枠の端から12.5の所）が、窓の部品の線に接している（user 11/14：4mmくらい短い）
        # （石膏ボードなしの仕様では戻りの線を描かないので見ない）
        if word in pf.WINDOWS and cfg.get("draw_board", True):
            hz = pl["horizontal"]
            alo = lambda pt: pt[0] if hz else pt[1]  # noqa: E731
            acr = lambda pt: pt[1] if hz else pt[0]  # noqa: E731
            fc = acr(tr((off + W / 2) / S, (ylo + yhi) / 2 / S))  # 窓の枠の真ん中（壁の厚みの向き）
            for u in (off + 12.5, off + W - 12.5):
                ua = alo(tr(u / S, 0))
                # 壁を横切る向きの紫線で、along がこの位置（±20）、壁芯の±250の中にあるもの
                rets = [(p, q) for p, q, la, col in lines if la != fl and col == 5 and p != q
                        and abs(alo(p) - alo(q)) < 0.3 and abs(alo(p) - ua) < 20
                        and min(acr(p), acr(q)) < pl["line"] + 250 and max(acr(p), acr(q)) > pl["line"] - 250]
                if not rets:
                    ng.append(("M", f"{word} W{W}（置き方 {pl['pos'][0]:.0f},{pl['pos'][1]:.0f}）: 壁の戻りの紫線がない", tr(u / S, -board / S)))
                    continue
                # 窓の側の端：枠の真ん中に一番近い端
                p, q = min(rets, key=lambda r: min(abs(acr(r[0]) - fc), abs(acr(r[1]) - fc)))
                tip, other = (p, q) if abs(acr(p) - fc) < abs(acr(q) - fc) else (q, p)
                bf = other
                d = min((seg_dist(tip, a_, b_) for a_, b_, *_ in P), default=99.0)
                if d > 0.3:
                    # 直す先：戻り線の向きにのばして最初に当たる部品の線（30まで）
                    dx, dy = tip[0] - bf[0], tip[1] - bf[1]
                    n_ = math.hypot(dx, dy) or 1.0
                    ray = LineString([tip, (tip[0] + dx / n_ * 30, tip[1] + dy / n_ * 30)])
                    hits = []
                    for a_, b_, *_ in P:
                        x_ = ray.intersection(LineString([a_, b_]))
                        if not x_.is_empty and x_.geom_type == "Point":
                            hits.append((math.hypot(x_.x - tip[0], x_.y - tip[1]), (x_.x, x_.y)))
                    if hits and geom is not None:
                        geom.setdefault("short_returns", []).append((p, q, tip, min(hits)[1]))
                    ng.append(("M", f"{word} W{W}（置き方 {pl['pos'][0]:.0f},{pl['pos'][1]:.0f}）: 壁の戻りの紫線が窓の線に {d:.1f} 届いていない", tip))
        purple = box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])).buffer(-1.5)
        a, b = tr((x0 + 3) / S, w0 / S), tr((x1 - 3) / S, w1 / S)
        wallbox = box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        a, b = tr((off - 30) / S, ylo / S), tr((off + W + 30) / S, yhi / S)
        whole = box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])).buffer(3)
        rects.append((pl, word, W, purple, wallbox, whole, centre))
        # G 柱（室内建具の両脇）
        if word in INTERIOR:
            along = lambda p: p[0] if pl["horizontal"] else p[1]  # noqa: E731
            g0_, g1_ = (x0, x1) if word == pf.OUTSET else (off, off + W)  # アウトセットは開口の両脇
            e0, e1 = along(tr(g0_ / S, 0)), along(tr(g1_ / S, 0))
            lo, hi = min(e0, e1), max(e0, e1)
            for t in (math.floor((lo - hc + 0.01) / grid) * grid, math.ceil((hi + hc - 0.01) / grid) * grid):
                pt = (t, pl["line"]) if pl["horizontal"] else (pl["line"], t)
                # 壁のない所（壁芯の線が通っていない交点）は見ない（プラン5 1F (2275,10920)：壁は 2730 から）
                if not any(seg_dist(pt, p, q) < 2 for p, q, la, _ in lines if la == cl):
                    continue
                if any(seg_dist(pt, a_, b_) < hc for a_, b_ in mull):
                    continue
                if not any(abs(c[0] - pt[0]) <= 2 and abs(c[1] - pt[1]) <= 2 for c in cols):
                    warn.append(("G", f"{word} W{W}: 脇のグリッド交点 ({pt[0]:g},{pt[1]:g}) に柱がない", pt))
    # B 残り
    stray = [fit[i] for i in np.where(~used)[0]]
    if stray:
        groups = []
        for p, q, *_ in stray:
            m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            for g in groups:
                if math.hypot(g[0][0] - m[0], g[0][1] - m[1]) < 1500:
                    g[1] += 1
                    break
            else:
                groups.append([m, 1])
        for m, n in groups:
            ng.append(("B", f"どの部品のものでもない建具の線 {n} 本", m))
    # C 紫線／D 貫通／H 青線
    for p, q, la, col in lines:
        if p == q:
            continue
        seg = LineString([p, q])
        mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        if la != fl and col == 5:
            for pl, word, W, purple, *_ in rects:
                if not purple.is_empty and seg.intersects(purple) and seg.intersection(purple).length > 1.0:
                    c_ = seg.intersection(purple).centroid
                    ng.append(("C", f"{word} W{W}（置き方 {pl['pos'][0]:.0f},{pl['pos'][1]:.0f}）: 紫線が枠の中に {seg.intersection(purple).length:.0f} 入っている", (c_.x, c_.y)))
        if la == wl and col != 5:
            for pl, word, W, _, wallbox, _, _ in rects:
                if seg.intersects(wallbox) and seg.intersection(wallbox).length > 3.0:
                    along_wall = (abs(q[0] - p[0]) > abs(q[1] - p[1])) == pl["horizontal"]
                    if along_wall:
                        c_ = seg.intersection(wallbox).centroid
                        ng.append(("D", f"{word} W{W}（置き方 {pl['pos'][0]:.0f},{pl['pos'][1]:.0f}）: 壁の線が建具の中を {seg.intersection(wallbox).length:.0f} 通っている", (c_.x, c_.y)))
        if la == fl and col == 6:
            inside = [word for pl, word, W, _, _, whole, _ in rects if whole.contains(seg)]
            if inside and all(w in INTERIOR for w in inside):
                ng.append(("H", f"{inside[0]}: 青（色6）の線がある", mid))
    # E 円弧の両端
    for c, r, a0, da, la, col in arcs:
        if la != fl or r < 300 or abs(da) < math.radians(30):
            continue
        for a in (a0, a0 + da):
            pt = (c[0] + r * math.cos(a), c[1] + r * math.sin(a))
            near = np.where((np.minimum(FA[:, 0], FA[:, 2]) < pt[0] + 6) & (np.maximum(FA[:, 0], FA[:, 2]) > pt[0] - 6)
                            & (np.minimum(FA[:, 1], FA[:, 3]) < pt[1] + 6) & (np.maximum(FA[:, 1], FA[:, 3]) > pt[1] - 6))[0]
            d = min((seg_dist(pt, fit[i][0], fit[i][1]) for i in near), default=99.0)
            if d > 5.0:
                ng.append(("E", f"円弧（半径{r:.0f}）の端が、扉・枠の線から {('5以上' if d > 50 else f'{d:.0f}')} 離れている", pt))
    # I 環境／J 形式
    env = json.loads((ROOT / "05_設定" / "参考_環境.json").read_text(encoding="utf-8"))
    if doc["header"].get("palette") != env.get("palette"):
        ng.append(("I", "環境設定（参考図面の .JWF）の色が入っていない", (cx, cy)))
    if doc.get("block_defs") or any(e["type"] in ("INSERT", "BLOCK") for e in doc["entities"]):
        ng.append(("J", "部品がブロックになっている", (cx, cy)))

    if geom is not None:
        geom.update(rects=rects, wl=wl, fl=fl, S=S, center=(cx, cy))
    if img_out and (ng or warn):
        sheet(lines, arcs, ng + warn, len(ng), img_out)
    if not quiet:
        for i, (k, s, p) in enumerate(ng):
            print(f"  NG{i + 1} [{k}] {s} @({p[0]:.0f},{p[1]:.0f})")
        for i, (k, s, p) in enumerate(warn):
            print(f"  注意{i + 1} [{k}] {s}")
    return ng, warn, len(meta["placements"])


def sheet(lines, arcs, marks, n_ng, out, half=700, k=0.4, per_row=4, limit=24):
    """NG・注意の所のまわり（±half mm）を切り出して、1枚にならべる（赤丸＝NG、橙丸＝注意、番号つき）"""
    n = int(2 * half * k)
    tiles = []
    for idx, (kind, _, (mx, my)) in enumerate(marks[:limit]):
        im = np.full((n, n, 3), 255, np.uint8)
        P = lambda x, y: (int(round((x - mx + half) * k)), int(round((my + half - y) * k)))  # noqa: E731
        for p, q, _, col in lines:
            if max(p[0], q[0]) < mx - half or min(p[0], q[0]) > mx + half or max(p[1], q[1]) < my - half or min(p[1], q[1]) > my + half:
                continue
            cv2.line(im, P(*p), P(*q), BGR.get(col, (0, 0, 0)), 1, cv2.LINE_AA)
        for c, r, a0, da, _, col in arcs:
            if abs(c[0] - mx) > half + r or abs(c[1] - my) > half + r:
                continue
            pts = np.array([P(c[0] + r * math.cos(a0 + da * i / 24), c[1] + r * math.sin(a0 + da * i / 24)) for i in range(25)], np.int32)
            cv2.polylines(im, [pts], False, BGR.get(col, (0, 0, 0)), 1, cv2.LINE_AA)
        is_ng = idx < n_ng
        cv2.circle(im, (n // 2, n // 2), 26, (0, 0, 255) if is_ng else (0, 140, 255), 2, cv2.LINE_AA)
        label = f"{'NG' if is_ng else 'W'}{idx + 1 if is_ng else idx - n_ng + 1} [{kind}]"
        cv2.putText(im, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_ng else (0, 140, 255), 2, cv2.LINE_AA)
        cv2.rectangle(im, (0, 0), (n - 1, n - 1), (200, 200, 200), 1)
        tiles.append(im)
    rows = []
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        row += [np.full((n, n, 3), 255, np.uint8)] * (per_row - len(row))
        rows.append(np.hstack(row))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.vstack(rows))[1].tofile(str(out))


def load_lib(cfg):
    libs = cfg["placement"]["library"]
    libs = [libs] if isinstance(libs, str) else libs
    return pf.load_library([str(q) for g in libs for q in sorted(ROOT.glob(g.replace("\\", "/")))])


def targets():
    out = []
    for sub in ("", "プラン2", "プラン3", "プラン4", "プラン5"):
        for fl in ("1F", "2F", "3F"):
            jww = ROOT / "02_出力" / sub / f"{fl}.jww"
            if fl == "3F" and not jww.exists():
                continue  # 3階建ての図がある時だけ（11/15）
            out.append((f"{sub or 'プラン1'} {fl}", jww, ROOT / "work" / "中間データ" / sub / f"{fl}_meta.json"))
    out.append(("プラン1 1F 正解図", ROOT / "06_正解図面（採点用）" / "プラン1_1F_正解.jww", ROOT / "work" / "中間データ" / "1F_meta.json"))
    return out


def main():
    cfg = spec.current()
    lib = load_lib(cfg)
    if sys.argv[1:2] == ["--all"]:
        outdir = ROOT / "work" / "検査"
        outdir.mkdir(parents=True, exist_ok=True)
        rep, total = [], 0
        for name, jww, meta in targets():
            print(f"■ {name}")
            if not (jww.exists() and meta.exists()):
                print("  ファイルがない")
                continue
            img = outdir / f"{name.replace(' ', '_')}.png"
            if img.exists():
                img.unlink()
            ng, warn, n = check(jww, meta, cfg, lib, img)
            total += len(ng)
            print(f"  建具 {n}、NG {len(ng)}、注意 {len(warn)}")
            rep.append({"name": name, "fittings": n, "ng": [[k, s, list(p)] for k, s, p in ng], "warn": [[k, s, list(p)] for k, s, p in warn]})
        (outdir / "検査結果.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"NG の合計 {total}")
        sys.exit(1 if total else 0)
    jww, meta = sys.argv[1:3]
    ng, warn, n = check(jww, meta, cfg, lib, sys.argv[3] if len(sys.argv) > 3 else None)
    print(f"建具 {n}、NG {len(ng)}、注意 {len(warn)}")
    sys.exit(1 if ng else 0)


if __name__ == "__main__":
    main()
