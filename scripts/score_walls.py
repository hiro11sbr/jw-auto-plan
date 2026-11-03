"""Score detected walls against a corrected drawing, per 455 grid edge.

usage: score_walls.py <config.json> <detected_walls.json> <correct.jww> <meta.json>

The correct drawing is an output .jww whose wall-center layer was fixed by hand in Jw_cad
(delete wrong lines, draw missing ones). Only that layer is read.
"""
import json
import sys

import ezjww

sys.stdout.reconfigure(encoding="utf-8")


def unit_edges(p1, p2, grid):
    """Split an axis-aligned segment into 455 unit edges: (h, line, step)."""
    (x1, y1), (x2, y2) = p1, p2
    out = set()
    if abs(y1 - y2) < 1:
        ln = round(y1 / grid)
        a, b = sorted((round(x1 / grid), round(x2 / grid)))
        out |= {(True, ln, s) for s in range(a, b)}
    elif abs(x1 - x2) < 1:
        ln = round(x1 / grid)
        a, b = sorted((round(y1 / grid), round(y2 / grid)))
        out |= {(False, ln, s) for s in range(a, b)}
    return out


def main():
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
    grid = cfg["input"]["half_grid_mm"]  # score on 227.5 units so closet walls count too
    det = json.load(open(sys.argv[2], encoding="utf-8"))
    meta = json.load(open(sys.argv[4], encoding="utf-8"))
    cx, cy = meta["center_mm"]
    S = meta["scale"]
    layer = cfg["layers"]["center"]["layer"]
    style = cfg["layers"]["center"]["style"]  # 窓台・笠木・寸法 share the layer: wall centres only
    group = cfg["layers"]["group"]

    pred = set()
    for w in det["walls"]:
        pred |= unit_edges(w["p1"], w["p2"], grid)

    truth = set()
    doc = ezjww.read_document(sys.argv[3])
    for e in doc["entities"]:
        b = e["base"]
        if e["type"] != "LINE" or b["layer"] != layer or b["layer_group"] != group or b["pen_style"] != style:
            continue
        p1 = (e["start_x"] * S + cx, e["start_y"] * S + cy)
        p2 = (e["end_x"] * S + cx, e["end_y"] * S + cy)
        truth |= unit_edges(p1, p2, grid)

    tp, fp, fn = pred & truth, pred - truth, truth - pred
    prec = len(tp) / len(pred) if pred else 0
    rec = len(tp) / len(truth) if truth else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"edges: truth={len(truth)} detected={len(pred)} correct={len(tp)}")
    print(f"precision={prec:.3f} (誤検出 {len(fp)})  recall={rec:.3f} (見落とし {len(fn)})  F1={f1:.3f}")

    def where(k):
        h, ln, st = k
        return f"{'横' if h else '縦'} {'Y' if h else 'X'}={ln * grid} {st * grid}-{(st + 1) * grid}"

    for name, s in (("誤検出", fp), ("見落とし", fn)):
        for k in sorted(s):
            print(f"  {name}: {where(k)}")


if __name__ == "__main__":
    main()
