"""2Fの階段まわり（ユーザー 2026-10-26）：階段のまわりは上がり切り（DNの側）以外、壁か腰壁。壁・腰壁の
どちらとも判定できない辺は腰壁にする。

階段の範囲は階段の芯（grid_detect が stair 印を付けた辺）から決める：芯の両側に1グリッド（910）ずつ、
長さは芯の両端からそれぞれ外側の壁まで。上がり切りは DN の文字がある側の流れの端（DNが読めなければ、
芯の端のうち外の壁から遠い側の、壁が無い半分）。

usage: stair_rail.py <walls.json> <edges.json> <部屋名.json> <腰壁.json> <config.json>
<腰壁.json> に足す（source="階段まわり"）。2F だけで使う。
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main():
    walls_path, edges_path, names_path, kos_path, cfg_path = sys.argv[1:6]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    grid = cfg["input"]["grid_mm"]
    det = json.load(open(walls_path, encoding="utf-8"))
    edges = json.load(open(edges_path, encoding="utf-8"))
    kos = json.load(open(kos_path, encoding="utf-8"))
    names = json.load(open(names_path, encoding="utf-8"))
    (sx, ix, nx0, _), (sy, iy, _, ny1) = det["fits"]["x"], det["fits"]["y"]
    X = lambda n: (n - nx0) * grid  # noqa: E731  grid index -> mm
    Y = lambda n: (ny1 - n) * grid  # noqa: E731
    walls = det["walls"]

    def covered(h, c, a, b):
        """A wall or 腰壁 already runs along (h, c) over most of a..b (mm)."""
        a, b = sorted((a, b))
        tot = 0.0
        for w in walls:
            (x1, y1), (x2, y2) = w["p1"], w["p2"]
            if h and abs(y1 - c) < 1 and abs(y2 - c) < 1:
                tot += max(0, min(b, max(x1, x2)) - max(a, min(x1, x2)))
            if not h and abs(x1 - c) < 1 and abs(x2 - c) < 1:
                tot += max(0, min(b, max(y1, y2)) - max(a, min(y1, y2)))
        for k in kos:
            if k["horizontal"] == h and abs(k["line_mm"] - c) < 120:
                tot += max(0, min(b, k["b_mm"]) - max(a, k["a_mm"]))
        return tot >= 0.6 * (b - a)

    def wall_hit(h, c, t, d):
        """From (c, t) walk along the spine direction d until a perpendicular wall line; its t."""
        best = None
        for w in walls:
            (x1, y1), (x2, y2) = w["p1"], w["p2"]
            wh = abs(y1 - y2) < 1
            if wh == h:
                continue  # need a wall across the spine
            wc = y1 if wh else x1
            lo, hi = sorted((x1, x2) if wh else (y1, y2))
            if lo - 1 <= c <= hi + 1 and (wc - t) * d > 1:
                if best is None or abs(wc - t) < abs(best - t):
                    best = wc
        for k in kos:  # a 腰壁 across the spine closes the stair too (the foot of the flight)
            if k["horizontal"] == h:
                continue
            if k["a_mm"] - 150 <= c <= k["b_mm"] + 150 and (k["line_mm"] - t) * d > -1:
                if best is None or abs(k["line_mm"] - t) < abs(best - t):
                    best = k["line_mm"]
        return best

    # stair spines: runs of stair-marked edges on one line
    runs = {}
    for e in edges:
        if e.get("stair"):
            runs.setdefault((e["h"], e["line"]), []).append(e["step"])
    dn = [n for n in names if n.get("name") == "階段" and str(n.get("matched", "")).upper() == "DN"]
    added = 0
    for (h, ln), steps in runs.items():
        if len(steps) < 2:
            continue
        s0, s1 = min(steps), max(steps) + 1
        # spine as mm: horizontal spine runs along x at y=Y(ln); vertical along y at x=X(ln)
        c = Y(ln) if h else X(ln)
        t0, t1 = sorted((X(s0), X(s1)) if h else (Y(s0), Y(s1)))
        e0, e1 = wall_hit(h, c, t0, -1), wall_hit(h, c, t1, 1)
        if e0 is None or e1 is None or e1 - e0 > cfg["stair_rail"]["max_len_mm"]:
            continue  # the stair's ends could not be closed: leave it to the person
        sides = (c - 2 * grid / 2 * 2, c + 2 * grid / 2 * 2)  # one flight (910) each side
        # the four outline pieces: two long sides, and the two ends split at the spine
        pieces = [(h, sides[0], e0, e1), (h, sides[1], e0, e1)]
        for t in (e0, e1):
            pieces += [(not h, t, sides[0], c), (not h, t, c, sides[1])]
        # 上がり切り: the end half nearest the DN label
        top = None
        if dn:
            px = (dn[0]["x"] - ix) / sx
            py = (dn[0]["y"] - iy) / sy
            dmm = (X(px), Y(py))
            # horizontal piece: across = y, along = x; vertical: across = x, along = y
            top = min(pieces[2:], key=lambda p: abs((dmm[1] if p[0] else dmm[0]) - p[1])
                      + abs((dmm[0] if p[0] else dmm[1]) - (p[2] + p[3]) / 2))
        for p in pieces:
            ph, pc, a, b = p
            if p is top or covered(ph, pc, a, b):
                continue
            kos.append({"horizontal": ph, "line_mm": pc, "a_mm": min(a, b), "b_mm": max(a, b),
                        "len_px": 0, "source": "階段まわり"})
            added += 1
            print(f"階段まわりに腰壁: {'横' if ph else '縦'} {pc} {min(a, b)}〜{max(a, b)}")
    json.dump(kos, open(kos_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"階段まわりの腰壁を足した: {added}")


if __name__ == "__main__":
    main()
