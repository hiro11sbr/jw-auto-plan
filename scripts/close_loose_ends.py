"""外壁の「L字のすき間」を壁でつなぐ（user 11/11, プラン4 1F：玄関ドアのある壁と袖壁が読み取れず、外周が開いていた）。

外壁の端のうち、ほかの外壁にふれていない端を「あき端」と呼ぶ。縦の壁のあき端 p と横の壁のあき端 q があり、
p から壁に直角に（壁の先へ）のばした線と、q から壁に直角にのばした線が、角 c で交わるとき、
p-c と c-q の2本を外壁として足す（どちらも max_mm 以下で、ほかの壁を横切らないこと）。
足した壁は flags に "L字つなぎ" を付ける。建具の判定（opening_detect）の前に動かすので、
そこにある玄関ドアなどは足した壁の上で見つかる。

usage: close_loose_ends.py <walls.json> <config.json>
"""
import json
import sys

from shapely.geometry import LineString, Point

sys.stdout.reconfigure(encoding="utf-8")


def run(walls_path, cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    mx = cfg.get("exterior_close", {}).get("l_close_max_mm")
    if not mx:
        return 0
    data = json.load(open(walls_path, encoding="utf-8"))
    ws = data["walls"]
    segs = [LineString([w["p1"], w["p2"]]) for w in ws]
    ext = [i for i, w in enumerate(ws) if w["type"] == "exterior"]
    loose = []
    for i in ext:
        w = ws[i]
        horizontal = abs(w["p1"][1] - w["p2"][1]) < 1.0
        for e, o in ((w["p1"], w["p2"]), (w["p2"], w["p1"])):
            pt = Point(*e)
            if any(j != i and segs[j].distance(pt) < 1.0 for j in ext):
                continue
            # the direction the wall runs, away from this end: the closing wall must leave the end at a right angle
            loose.append({"i": i, "p": tuple(e), "h": horizontal, "o": tuple(o)})
    added = []
    used = set()
    cands = []
    for a in loose:
        for b in loose:
            if a["i"] >= b["i"] or a["h"] == b["h"]:
                continue
            v, h = (a, b) if not a["h"] else (b, a)  # v: vertical wall's end, h: horizontal wall's end
            c = (h["p"][0], v["p"][1])
            l1 = abs(c[0] - v["p"][0])  # p(v) -> c runs along x
            l2 = abs(c[1] - h["p"][1])  # c -> q(h) runs along y
            if min(l1, l2) < 1.0 or max(l1, l2) > mx:
                continue
            # the corner must lie beyond the ends, not back along the walls
            if (c[0] - v["p"][0]) * (v["o"][0] - v["p"][0]) > 0:
                continue
            if (c[1] - h["p"][1]) * (h["o"][1] - h["p"][1]) > 0:
                continue
            s1, s2 = LineString([v["p"], c]), LineString([c, h["p"]])
            bad = False
            for j, l in enumerate(segs):
                for s in (s1, s2):
                    inter = s.intersection(l)
                    if inter.is_empty:
                        continue
                    # touching at the ends we join is fine
                    if inter.length > 0 or not (inter.distance(Point(*v["p"])) < 1.0 or inter.distance(Point(*h["p"])) < 1.0):
                        bad = True
            if bad:
                continue
            cands.append((l1 + l2, id(v), id(h), v, h, c))
    for tot, _, _, v, h, c in sorted(cands, key=lambda t: t[0]):
        if (v["i"], tuple(v["p"])) in used or (h["i"], tuple(h["p"])) in used:
            continue
        used.add((v["i"], tuple(v["p"])))
        used.add((h["i"], tuple(h["p"])))
        for p, q in ((v["p"], c), (c, h["p"])):
            added.append({"type": "exterior", "p1": [p[0], p[1]], "p2": [q[0], q[1]], "fit_edges": 0, "exact": True,
                          "source": "L字つなぎ", "flags": ["L字つなぎ"]})
        print(f"外壁のL字のすき間をつなぐ: {tuple(v['p'])} -> {c} -> {tuple(h['p'])}")
    if added:
        data["walls"] = [w for w in ws if "L字つなぎ" not in w.get("flags", [])] + added
        json.dump(data, open(walls_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return len(added)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
