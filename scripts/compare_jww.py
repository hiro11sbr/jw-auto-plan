"""Compare an automatic drawing with the user's hand-corrected one (both .jww, same paper setup).

usage: compare_jww.py <auto.jww> <corrected.jww> <meta.json> <out.png> [x0 y0 x1 y1 (model mm crop)]

Blocks are exploded. Line/arc pieces are matched after rounding to 0.5 mm:
  gray = in both, red = only in the automatic one (removed by the user),
  green = only in the corrected one (added by the user). Texts are drawn in blue.
Prints the block references and texts of the corrected file in model mm.
"""
import json
import math
import sys
from pathlib import Path

import cv2
import ezjww
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jp_text  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def pieces(path, S, center):
    """[(kind, key, pts model mm, layer, color)] with blocks exploded; texts separately."""
    doc = ezjww.read_document(path)
    defs = {b["name"]: b for b in doc["block_defs"]}
    cx, cy = center
    out, texts, refs = [], [], []

    def emit(e, tr, layer, color):
        if e["type"] == "LINE":
            p, q = tr(e["start_x"], e["start_y"]), tr(e["end_x"], e["end_y"])
            out.append(("L", [p, q], layer, color))
        elif e["type"] in ("ARC", "CIRCLE"):
            a0 = e.get("start_angle", 0.0) or 0.0
            sw = e.get("arc_angle", 2 * math.pi) or 2 * math.pi
            r = e["radius"]
            pts = [tr(e["center_x"] + r * math.cos(a0 + sw * i / 16), e["center_y"] + r * math.sin(a0 + sw * i / 16))
                   for i in range(17)]
            out.append(("A", pts, layer, color))
        elif e["type"] == "BLOCK":
            b = defs.get(e["block_name"])
            if not b:
                return
            ang = e["rotation"]  # radians when read back
            ca, sa = math.cos(ang), math.sin(ang)

            def tr2(u, v, e=e, ca=ca, sa=sa):
                u, v = u * e["scale_x"], v * e["scale_y"]
                return tr(e["ref_x"] + u * ca - v * sa, e["ref_y"] + u * sa + v * ca)

            for c in b["entities"]:
                emit(c, tr2, layer, color)

    def top(x, y):
        return (x * S + cx, y * S + cy)

    for e in doc["entities"]:
        b = e.get("base", {})
        if e["type"] == "TEXT":
            texts.append((e.get("text") or e.get("content") or "", top(e["start_x"], e["start_y"])))
        elif e["type"] == "BLOCK":
            refs.append((e["block_name"].split("@@")[0], top(e["ref_x"], e["ref_y"]), e["scale_x"], e["scale_y"], e["rotation"]))
        emit(e, top, b.get("layer"), b.get("pen_color"))
    return out, texts, refs


def key(pts):
    r = [(round(x * 2) / 2, round(y * 2) / 2) for x, y in pts]
    return tuple(r) if r[0] <= r[-1] else tuple(reversed(r))


def main():
    auto, corr, meta_path, out_png = sys.argv[1:5]
    crop = list(map(float, sys.argv[5:9])) if len(sys.argv) > 8 else None
    meta = json.load(open(meta_path, encoding="utf-8"))
    S, center = meta["scale"], meta["center_mm"]
    A, _, _ = pieces(auto, S, center)
    B, texts, refs = pieces(corr, S, center)
    ka = {key(p[1]) for p in A}
    kb = {key(p[1]) for p in B}
    allp = [pt for p in A + B for pt in p[1] if abs(pt[0]) < 1e5]
    if crop:
        x0, y0, x1, y1 = crop
    else:
        xs, ys = [p[0] for p in allp], [p[1] for p in allp]
        x0, x1 = np.percentile(xs, 1) - 500, np.percentile(xs, 99) + 500
        y0, y1 = np.percentile(ys, 1) - 500, np.percentile(ys, 99) + 500
    k = 2400 / max(x1 - x0, y1 - y0)  # px per mm
    W, H = int((x1 - x0) * k) + 1, int((y1 - y0) * k) + 1
    img = np.full((H, W, 3), 255, np.uint8)

    def P(p):
        return int(round((p[0] - x0) * k)), int(round((y1 - p[1]) * k))

    for kind, pts, _l, _c in A:
        if key(pts) not in kb:
            cv2.polylines(img, [np.array([P(p) for p in pts])], False, (0, 0, 230), 2, cv2.LINE_AA)
    for kind, pts, _l, _c in B:
        col = (150, 150, 150) if key(pts) in ka else (0, 170, 0)
        cv2.polylines(img, [np.array([P(p) for p in pts])], False, col, 1 if col[0] == 150 else 2, cv2.LINE_AA)
    items = [(t, P(p), (200, 80, 0), 14) for t, p in texts if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
    img = jp_text.put(img, items)
    img = jp_text.legend(img, "手直し前後の比較", [((150, 150, 150), "変わらない線"),
                                            ((0, 0, 230), "手直しで消えた線"),
                                            ((0, 170, 0), "手直しで足された線"),
                                            ((200, 80, 0), "文字")])
    cv2.imencode(".png", img)[1].tofile(out_png)
    print(f"lines auto={len(A)} corrected={len(B)} removed={len(ka - kb)} added={len(kb - ka)}")
    print("blocks:")
    for r in refs:
        print("  ", r[0], tuple(round(v, 1) for v in r[1]), "sx", r[2], "sy", r[3], "rot", r[4])
    print("texts:")
    for t, p in texts:
        print("  ", t, tuple(round(v, 1) for v in p))


if __name__ == "__main__":
    main()
