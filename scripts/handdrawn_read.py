"""手書き平面図（写真）から、壁・建具の開口・部屋の区画を読み取る（独立プログラム）。
使い方:  python handdrawn_read.py 写真.jpg [出力フォルダ]
出力:  <名前>_読取.png（確認用の重ね図） と <名前>_読取.json（壁・開口・部屋）
AI の判断は使わない。画像処理だけで読む。
"""
import sys, os, json
import cv2
import numpy as np



# ---------- 0. 方眼紙の1マスの大きさから、写真の拡大率をそろえる ----------
REF_CELL = 64.0      # 基準（試作で調整した写真の1マス）


def grid_cell(im):
    g = im.min(axis=2)
    bg = cv2.medianBlur(g, 51).astype(np.float32)
    d = bg - g.astype(np.float32)
    m = ((d > 6) & (d < 40)).astype(np.float32)
    per = []
    for ax in (0, 1):
        p = m.mean(axis=ax); p = p - p.mean()
        N = len(p) * 8
        f = np.abs(np.fft.rfft(p * np.hanning(len(p)), n=N)); fr = np.fft.rfftfreq(N)
        ff = f * ((fr > 1 / 220) & (fr < 1 / 15))
        i = int(np.argmax(ff)); P = 1 / fr[i]
        # 半分の周期の山が最大なら、本当の1マスはその2倍
        j = int(np.argmin(np.abs(fr - 1 / (2 * P)))) if P > 0 else i
        lo, hi = max(j - 40, 0), j + 40
        if ff[lo:hi].max() >= 0.75 * ff[i] and 2 * P < 220:
            P = 2 * P if P < 55 else P
        per.append(P)
    return float(np.mean(per))


# ---------- 1. 鉛筆の線だけ取り出す ----------
def read_ink(im):
    m = im.max(axis=2).astype(np.float32)           # 青い方眼・赤い寸法線は max が明るいので消える
    bg = cv2.medianBlur(m.astype(np.uint8), 51).astype(np.float32)
    return ((bg - m) > 45).astype(np.uint8) * 255


def deskew(im, ink):
    """水平・垂直になるように、傾きを直す（長い線の角度の中央値）。"""
    ls = cv2.HoughLinesP(ink, 1, np.pi / 720, 200, minLineLength=250, maxLineGap=10)
    if ls is None:
        return im, ink, 0.0
    a = []
    for x1, y1, x2, y2 in np.asarray(ls).reshape(-1, 4):
        t = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        t = (t + 45) % 90 - 45
        a.append(t)
    ang = float(np.median(a))
    if abs(ang) < 0.15:
        return im, ink, 0.0
    h, w = ink.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return (cv2.warpAffine(im, M, (w, h), borderValue=(255, 255, 255)),
            cv2.warpAffine(ink, M, (w, h)), ang)


def runs(mask, L):
    h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    return h, v


# ---------- 2. 家（平面図）の範囲を見つける ----------
def find_plans(ink):
    dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)
    thick = cv2.dilate(((dt >= 2) * 255).astype(np.uint8), np.ones((3, 3), np.uint8))
    h, v = runs(thick, 45)
    w = cv2.bitwise_or(h, v)
    big = cv2.dilate(w, np.ones((31, 31), np.uint8))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(big, connectivity=8)
    out = []
    for i in range(1, nl):
        x, y, ww, hh, a = st[i]
        if ww >= 250 and hh >= 250:
            n_px = int((w[y:y + hh, x:x + ww] > 0).sum())
            if n_px > 3000:
                out.append((x + 15, y + 15, x + ww - 15, y + hh - 15))
    return out


# ---------- 3. 壁 ----------
WALL_DT = 2.0
THIN_ADD = True
MIN_COMP = 110


def find_walls(ink, box, margin=12):
    x0, y0, x1, y1 = box
    X0, Y0 = max(x0 - margin, 0), max(y0 - margin, 0)
    X1, Y1 = min(x1 + margin, ink.shape[1]), min(y1 + margin, ink.shape[0])
    k = ink[Y0:Y1, X0:X1]
    dt = cv2.distanceTransform((k > 0).astype(np.uint8), cv2.DIST_L2, 5)
    core = cv2.dilate(((dt >= WALL_DT) * 255).astype(np.uint8), np.ones((3, 3), np.uint8))
    th, tv = runs(core, 45)
    thick = cv2.bitwise_or(th, tv)
    # 孤立した短い塊（階段・設備・家具）を外す
    nl, lab, st, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    keep = np.zeros_like(thick)
    for i in range(1, nl):
        if max(st[i][2], st[i][3]) >= MIN_COMP:
            keep[lab == i] = 255
    thick = keep
    # 長い細い線：太い壁につながるものだけ壁にする（薄く描いた間仕切り）
    lh, lv = runs(k, 90)
    thin = cv2.dilate(cv2.bitwise_or(lh, lv), np.ones((3, 3), np.uint8))
    walls = thick.copy()
    for _ in range(3 if THIN_ADD else 0):
        near = cv2.dilate(walls, np.ones((15, 15), np.uint8))
        nl, lab, st, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
        add = False
        for i in range(1, nl):
            m = lab == i
            if (near[m] > 0).any() and not (walls[m] > 0).all():
                walls[m] = 255; add = True
        if not add:
            break
    return walls, thick, (X0, Y0, X1, Y1), k


# ---------- 4. 壁の切れ目（開口）と、建具かどうか ----------
def find_openings(walls, ink_k, house):
    """壁の線の端から、同じ向きに最大100px 先まで伸ばし、次の壁に当たれば、その間を開口の候補にする。"""
    H, V = runs(walls, 25)
    Hh, Ww = walls.shape
    wl = cv2.dilate(walls, np.ones((3, 3), np.uint8))
    ys, xs = np.where(walls > 0)
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    cand = []

    def segs(M, horiz):
        nl, lab, st, _ = cv2.connectedComponentsWithStats(M, connectivity=8)
        r = []
        for i in range(1, nl):
            x, y, w, h, a = st[i]
            if horiz and w >= 40 and w >= h * 3:
                r.append((x, x + w, int(y + h / 2)))
            if (not horiz) and h >= 40 and h >= w * 3:
                r.append((y, y + h, int(x + w / 2)))
        return r

    for horiz, M in ((True, H), (False, V)):
        for (s0, e0, p) in segs(M, horiz):
            for end, d in ((e0, 1), (s0, -1)):
                hit = None
                for t in range(4, 101):          # 線の端のにじみ（1〜3px）は飛ばす
                    q = end + d * t
                    if horiz:
                        if not (0 <= q < Ww): break
                        band = wl[max(p - 2, 0):p + 3, q]
                    else:
                        if not (0 <= q < Hh): break
                        band = wl[q, max(p - 2, 0):p + 3]
                    if band.any():
                        hit = t; break
                if hit is None or hit < 24:
                    continue
                a_, b_ = sorted((end, end + d * hit))
                cand.append(dict(horiz=horiz, s=float(a_), e=float(b_), p=float(p)))
    uniq = []
    for r in cand:
        if not any(u["horiz"] == r["horiz"] and abs(u["p"] - r["p"]) < 8 and abs(u["s"] - r["s"]) < 8 and abs(u["e"] - r["e"]) < 8 for u in uniq):
            uniq.append(r)
    wd = cv2.dilate(walls, np.ones((9, 9), np.uint8))
    free = cv2.bitwise_and(ink_k, cv2.bitwise_not(wd))
    dti = cv2.distanceTransform(cv2.bitwise_not(free), cv2.DIST_L2, 3)     # 線までの距離
    for r in uniq:
        s, e, p = int(r["s"]), int(r["e"]), int(r["p"])
        g = e - s
        arc = 0.0
        for hinge, dirn in ((s, 1), (e, -1)):             # 蝶番の側（開口の左右どちらか）
            for side in (-1, 1):                          # 扉の開く側
                for rr in (0.75, 0.9, 1.0, 1.1, 1.25):
                    R = rr * g
                    for dh in (-6, 0, 6):
                        pts = []
                        for t in np.linspace(0, np.pi / 2, 36):
                            u = (hinge + dh) + dirn * R * np.cos(t)     # 開口の方向
                            v = p + side * R * np.sin(t)                # 開口に直角の方向
                            pts.append((u, v) if r["horiz"] else (v, u))
                        ok = 0
                        for (x, y) in pts:
                            xi, yi = int(round(x)), int(round(y))
                            if 0 <= xi < dti.shape[1] and 0 <= yi < dti.shape[0] and dti[yi, xi] <= 3.5:
                                ok += 1
                        arc = max(arc, ok / 36.0)
        # 引戸など：開口に沿った線（壁とは別の細い線）が開口の大半を覆う
        if r["horiz"]:
            band = free[max(p - 8, 0):p + 9, s:e]
            cover = float((band > 0).any(axis=0).mean()) if band.size else 0.0
        else:
            band = free[s:e, max(p - 8, 0):p + 9]
            cover = float((band > 0).any(axis=1).mean()) if band.size else 0.0
        r["score"] = [round(arc, 2), round(cover, 2)]
        ex = (r["horiz"] and (abs(p - by0) < 25 or abs(p - by1) < 25)) or ((not r["horiz"]) and (abs(p - bx0) < 25 or abs(p - bx1) < 25))
        r["on_edge"] = bool(ex)
        r["door"] = bool(arc >= 0.6 or cover >= 0.85)
    return uniq


# ---------- 5. 部屋に分ける ----------
def split_rooms(walls, thick, openings, min_inner=10, thick_min=0.35, outline=None):
    closed = walls.copy()
    dl = np.zeros_like(walls)                       # 建具・窓でふさいだ線
    for r in openings:
        if r["door"] or r["on_edge"]:
            s, e, p = int(r["s"]), int(r["e"]), int(r["p"])
            if r["horiz"]:
                cv2.line(closed, (s, p), (e, p), 255, 3); cv2.line(dl, (s, p), (e, p), 255, 3)
            else:
                cv2.line(closed, (p, s), (p, e), 255, 3); cv2.line(dl, (p, s), (p, e), 255, 3)
    closed = cv2.dilate(closed, np.ones((3, 3), np.uint8))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))   # 壁のつなぎ目の小さな抜け（20px未満）はふさぐ
    dl = cv2.dilate(dl, np.ones((3, 3), np.uint8))
    free = (closed == 0).astype(np.uint8)
    # 外側の判定：壁を大きく閉じて、枠に届く空きを「家の外」とする（外周の小さな抜けで部屋が外とつながるのを防ぐ）
    if outline is not None:
        closed_o = cv2.bitwise_or(closed, outline)
    else:
        closed_o = closed
    PAD = 130                                        # 家の外に余白をつけてから閉じる（余白が細いと、外側が埋まってしまう）
    padded = cv2.copyMakeBorder(closed_o, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    big = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, np.ones((101, 101), np.uint8))
    n2, l2 = cv2.connectedComponents((big == 0).astype(np.uint8), connectivity=4)
    ext = (l2 == l2[0, 0])[PAD:-PAD, PAD:-PAD]
    free[ext] = 0
    nl, lab, st, cen = cv2.connectedComponentsWithStats(free, connectivity=4)
    H, W = free.shape
    valid = {}
    for i in range(1, nl):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H or a < 1500:
            continue
        valid[i] = True
    # 壁1本をはさんで隣り合う部屋の組を数え、境が「細い線だけ」なら家具・設備の線とみなして合体する
    lab16 = lab.astype(np.uint16)
    mx = cv2.dilate(lab16, np.ones((11, 11), np.uint8))
    lab_nz = np.where(lab16 == 0, 65535, lab16).astype(np.uint16)
    mn = cv2.erode(lab_nz, np.ones((11, 11), np.uint8))
    wallpx = (closed > 0) & (lab16 == 0)
    sel = wallpx & (mx != mn) & (mn != 65535) & (mx > 0)
    ys, xs = np.where(sel)
    pair = {}
    thk = cv2.dilate(thick, np.ones((3, 3), np.uint8))
    for y, x in zip(ys, xs):
        k = (int(mn[y, x]), int(mx[y, x]))
        c = pair.setdefault(k, [0, 0, 0])
        c[0] += 1
        c[1] += 1 if thk[y, x] else 0
        c[2] += 1 if dl[y, x] else 0
    parent = {i: i for i in valid}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for (a_, b_), (n, t, dd) in pair.items():
        if a_ in valid and b_ in valid and n >= 20 and dd < 6 and t / n < thick_min:
            parent[find(a_)] = find(b_)
    groups = {}
    for i in valid:
        groups.setdefault(find(i), []).append(i)
    newlab = np.zeros_like(lab)
    rooms = []
    dt = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    for g, mem in groups.items():
        m = np.isin(lab, mem)
        a = int(m.sum())
        if dt[m].max() < min_inner:
            continue
        ys, xs = np.where(m)
        newlab[m] = g
        rooms.append(dict(label=int(g), area=a, cx=float(xs.mean()), cy=float(ys.mean()),
                          bbox=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]))
    return rooms, newlab


def main():
    global WALL_DT, THIN_ADD, MIN_COMP
    src = sys.argv[1]
    outd = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(src))
    os.makedirs(outd, exist_ok=True)
    name = os.path.splitext(os.path.basename(src))[0]
    im = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_COLOR)
    cell = grid_cell(im)
    sc = REF_CELL / cell if 20 < cell < 220 else 1.0
    if abs(sc - 1) > 0.06:
        im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    ink = read_ink(im)
    # 線の太さで、図の種類を判定：壁だけが太いペン書き（清書）なら、太い線だけを壁にする
    _dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)
    _v = _dt[_dt > 1]
    if len(_v) and np.percentile(_v, 90) >= 4.5:
        WALL_DT, THIN_ADD, MIN_COMP = 2.6, False, 60
    im, ink, ang = deskew(im, ink)
    plans = find_plans(ink)
    report = dict(source=src, deskew_deg=ang, grid_cell_px=cell, scale=sc, plans=[])
    rng = np.random.RandomState(5)
    for pi, box in enumerate(plans, 1):
        walls, thick, (X0, Y0, X1, Y1), k = find_walls(ink, box)
        ops = find_openings(walls, k, (0, 0, X1 - X0, Y1 - Y0))
        _lh, _lv = runs(k, 90)
        rooms, lab = split_rooms(walls, thick, ops, outline=cv2.dilate(cv2.bitwise_or(_lh, _lv), np.ones((3, 3), np.uint8)))
        crop = im[Y0:Y1, X0:X1].copy()
        col = crop.copy()
        for n, r in enumerate(rooms, 1):
            c = rng.randint(70, 255, 3).tolist()
            col[lab == r["label"]] = c
        out = cv2.addWeighted(crop, 0.5, col, 0.5, 0)
        out[walls > 0] = (0, 0, 220)
        for r in ops:
            s, e, p = int(r["s"]), int(r["e"]), int(r["p"])
            colr = (0, 200, 0) if (r["door"] or r["on_edge"]) else (0, 200, 255)
            a_, b_ = ((s, p), (e, p)) if r["horiz"] else ((p, s), (p, e))
            cv2.line(out, a_, b_, colr, 5)
        for n, r in enumerate(rooms, 1):
            cv2.putText(out, str(n), (int(r["cx"]) - 10, int(r["cy"]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.imencode(".png", cv2.resize(out, None, fx=1.5, fy=1.5))[1].tofile(os.path.join(outd, f"{name}_読取{pi}.png"))
        report["plans"].append(dict(box=[X0, Y0, X1, Y1], rooms=len(rooms),
                                    openings=[dict(horiz=r["horiz"], s=r["s"], e=r["e"], p=r["p"], door=r["door"], on_edge=r["on_edge"], score=r["score"]) for r in ops],
                                    room_list=rooms))
        print(f"平面図{pi}: 部屋 {len(rooms)}、開口 {len(ops)}（建具/外周 {sum(1 for r in ops if r['door'] or r['on_edge'])}）")
    with open(os.path.join(outd, f"{name}_読取.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
