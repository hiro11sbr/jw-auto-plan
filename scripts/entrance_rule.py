"""玄関ドアの決め方（ユーザー 2026-10-23）: 玄関とポーチ（外）の間にある建具が1つなら、それが玄関ドア。
複数あれば、そのうちのどれか — 大きさと開き方（開き戸の円弧）で選ぶ。

usage: entrance_rule.py <openings.json> <walls.json> <部屋名.json> <config.json> [<plan.png>]
The 玄関 label must see the opening's middle with no wall in between (it is the 玄関's own
outside wall), within entrance.max_reach_mm. Rewrites kind to "entrance_door" for the chosen one.
"""
import json
import sys
from pathlib import Path

from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opening_detect import Mapper  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def read_near(img_path, ops, ec):
    """The whole-plan OCR misses 玄関 written over floor tiles (プラン2). Read again around each
    door-sized opening in an outside wall: enlarged, and with the light tile lines dropped."""
    import subprocess
    import tempfile

    import cv2
    import numpy as np
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    H, W = bgr.shape[:2]
    R, k = ec["ocr_reach_px"], ec["ocr_scale"]
    ps1 = str(Path(__file__).resolve().parent / "ocr_windows.ps1")
    found = []
    tmp = Path(tempfile.mkdtemp(prefix="jw_entrance_"))
    for i, o in enumerate(ops):
        if o.get("rejected") or o["wall_type"] != "exterior" \
                or not ec["width_mm"][0] <= o["width_mm"] <= ec["width_mm"][1]:
            continue
        p0, p1, c = o["px"]
        m = (p0 + p1) / 2
        cx, cy = (m, c) if o["horizontal"] else (c, m)
        x0, y0 = int(max(0, cx - R)), int(max(0, cy - R))
        x1, y1 = int(min(W, cx + R)), int(min(H, cy + R))
        g = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        g = np.where(g < ec["ocr_binarize_max"], 0, 255).astype(np.uint8)
        png, js = tmp / f"{i}.png", tmp / f"{i}.json"
        cv2.imencode(".png", g)[1].tofile(str(png))
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1, str(png), str(js)],
                       check=True, capture_output=True)
        for ln in json.load(open(js, encoding="utf-8-sig"))["lines"]:
            if ln["text"].replace(" ", "") != "玄関":
                continue
            ws = ln["words"]
            xs = [w["x"] for w in ws] + [w["x"] + w["w"] for w in ws]
            ys = [w["y"] for w in ws] + [w["y"] + w["h"] for w in ws]
            found.append({"name": "玄関", "x": x0 + (min(xs) + max(xs)) / 2 / k, "y": y0 + (min(ys) + max(ys)) / 2 / k})
    # the same label read from several crops
    uniq = []
    for f in found:
        if all(abs(f["x"] - u["x"]) > 10 or abs(f["y"] - u["y"]) > 10 for u in uniq):
            uniq.append(f)
    print(f"玄関の文字（建具まわりを読み直し）: {len(uniq)}")
    return uniq


def main():
    ops_path, walls_path, names_path, cfg_path = sys.argv[1:5]
    img_path = sys.argv[5] if len(sys.argv) > 5 else None
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    ec = cfg["entrance"]
    det = json.load(open(walls_path, encoding="utf-8"))
    M = Mapper(det, cfg["input"]["grid_mm"])
    ops = json.load(open(ops_path, encoding="utf-8"))
    labels = [n for n in json.load(open(names_path, encoding="utf-8")) if n.get("name") == "玄関"]
    if not labels and img_path:
        labels = read_near(img_path, ops, ec)
        if labels:
            # keep it with the room names: the 框 test and the region check use it too
            names = json.load(open(names_path, encoding="utf-8"))
            for lab in labels:
                names.append({"raw": "玄関", "x": lab["x"], "y": lab["y"],
                              "box": [lab["x"] - 15, lab["y"] - 8, lab["x"] + 15, lab["y"] + 8], "name": "玄関",
                              "kind": "玄関・動線", "score": 1.0, "matched": "玄関",
                              "source": "建具まわりの読み直し"})
            json.dump(names, open(names_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not labels:
        print("玄関の文字なし: 玄関ドアの決め直しなし")
        return
    walls = [LineString([w["p1"], w["p2"]]) for w in det["walls"]]
    cands = []
    for lab in labels:
        lp = (M.xmm(lab["x"]), M.ymm(lab["y"]))
        for i, o in enumerate(ops):
            if o.get("rejected") or o["wall_type"] != "exterior":
                continue
            (x1, y1), (x2, y2) = o["p1_mm"], o["p2_mm"]
            mid = ((x1 + x2) / 2, (y1 + y2) / 2)
            sight = LineString([lp, mid])
            if sight.length > ec["max_reach_mm"]:
                continue
            # the sight line may touch the opening's own wall line only at its end
            own = LineString([o["p1_mm"], o["p2_mm"]])
            blocked = any(w.crosses(sight) and w.distance(own.centroid) > 1 for w in walls
                          if not w.buffer(1).contains(own))
            if not blocked:
                cands.append((i, sight.length))
    cands = sorted(set(cands), key=lambda c: c[1])
    if not cands:
        print("玄関から見える外壁の建具なし")
        return

    def door_like(o):
        # a door-sized opening with a swing arc beats a window
        w_ok = ec["width_mm"][0] <= o["width_mm"] <= ec["width_mm"][1]
        return (o["kind"] == "entrance_door", w_ok, o.get("arc", 0))

    best = cands[0][0] if len(cands) == 1 else max((i for i, _ in cands), key=lambda i: door_like(ops[i]))
    o = ops[best]
    if o["kind"] != "entrance_door":
        print(f"玄関ドアに変更: {best} {o['kind']} 幅{round(o['width_mm'])}（候補{len(cands)}）")
        o["kind_before_entrance_rule"] = o["kind"]
        o["kind"] = "entrance_door"
        o.pop("window_type", None)
    else:
        print(f"玄関ドア: {best} 幅{round(o['width_mm'])}（候補{len(cands)}）")
    json.dump(ops, open(ops_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
