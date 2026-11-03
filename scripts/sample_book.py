"""見本帳: every opening of the given plans cut out on one sheet, with its id and current judgement,
so the drawing conventions of each fitting type can be learnt (and labelled).

usage: sample_book.py <out.png> <plan.png> <openings.json> <tag> [<plan.png> <openings.json> <tag> ...]
also writes <out>.json: [{id, tag, index, kind, window_type, label, px}] for labelling
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jp_text  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
KIND_JA = {"window": "窓", "swing": "片開き", "ub_door": "UBドア", "sliding_pocket": "片引戸",
           "sliding": "引違い戸", "storage": "両開き", "entrance_door": "玄関ドア", "folding": "折戸",
           "kamachi": "框"}
CELL = 380  # px per cell side
CONTEXT = 12  # px of faded plan around the fitting's own box
FADE = 0.8  # how far everything outside the fitting's own box is faded toward white
BAND = 14  # px across the wall either side: the frame and sash lines
FAN = 45  # px a window fan / dashed 横すべり line reaches outside
MM_PER_PX = 12.5


def reach(o):
    """(px toward image -side, px toward +side) across the wall that the fitting's symbol uses."""
    leaf = o["width_mm"] / MM_PER_PX
    kind, side = o["kind"], o.get("swing_side")
    if kind == "window":
        return FAN, FAN
    if kind in ("swing", "entrance_door", "ub_door", "storage"):
        r = leaf / 2 if kind == "storage" else leaf
        r = int(r) + 4
        if side == 1:
            return BAND, r
        if side == -1:
            return r, BAND
        return r, r
    return BAND, BAND  # sliding doors, 折戸 drawn inside the band


def main():
    out = sys.argv[1]
    args = sys.argv[2:]
    cells, index = [], []
    n = 0
    for k in range(0, len(args), 3):
        img_path, ops_path, tag = args[k:k + 3]
        bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        H, W = bgr.shape[:2]
        for i, o in enumerate(json.load(open(ops_path, encoding="utf-8"))):
            if o.get("rejected") and o["kind"] != "kamachi":
                continue
            p0, p1, c = o["px"]
            c = int(round(c))
            p0, p1 = int(round(p0)), int(round(p1))
            # the fitting's own box: its span along the wall, and across the wall only as far as
            # its symbol reaches (the door leaf on its swing side, a window fan) — cut in both
            # directions, so a neighbouring fitting is not in the picture (user, 2026-10-23)
            lo, hi = reach(o)
            if o["horizontal"]:
                bx0, bx1, by0, by1 = p0, p1, c - lo, c + hi
            else:
                bx0, bx1, by0, by1 = c - lo, c + hi, p0, p1
            x0, x1, y0, y1 = bx0 - CONTEXT, bx1 + CONTEXT, by0 - CONTEXT, by1 + CONTEXT
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            crop = bgr[y0:y1, x0:x1].copy()
            # a thin faded margin of plan around the box, for orientation only
            keep = np.zeros(crop.shape[:2], bool)
            keep[max(0, by0 - y0):by1 - y0 + 1, max(0, bx0 - x0):bx1 - x0 + 1] = True
            crop[~keep] = (crop[~keep] * (1 - FADE) + 255 * FADE).astype(np.uint8)
            # the opening's ends as red ticks on both sides of the wall
            red = (0, 0, 230)
            if o["horizontal"]:
                for p in (p0 - x0, p1 - x0):
                    cv2.line(crop, (p, 0), (p, 8), red, 2)
                    cv2.line(crop, (p, crop.shape[0] - 9), (p, crop.shape[0] - 1), red, 2)
                cv2.line(crop, (p0 - x0, 2), (p1 - x0, 2), red, 2)
            else:
                for p in (p0 - y0, p1 - y0):
                    cv2.line(crop, (0, p), (8, p), red, 2)
                    cv2.line(crop, (crop.shape[1] - 9, p), (crop.shape[1] - 1, p), red, 2)
                cv2.line(crop, (2, p0 - y0), (2, p1 - y0), red, 2)
            s = (CELL - 40) / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            cell = np.full((CELL, CELL, 3), 255, np.uint8)
            cell[40:40 + crop.shape[0], :crop.shape[1]] = crop
            n += 1
            kind = KIND_JA.get(o["kind"], o["kind"])
            wt = o.get("window_type") or ""
            lab = (o.get("label") or {}).get("code", "")
            text = f"{n} {tag}-{i} {kind} {wt} {lab}".strip()
            cell = jp_text.put(cell, [(text, (4, 4), (0, 0, 0), 15),
                                      (f"幅{round(o['width_mm'])}", (4, 22), (120, 60, 0), 13)])
            cv2.rectangle(cell, (0, 0), (CELL - 1, CELL - 1), (200, 200, 200), 1)
            cells.append(cell)
            index.append({"id": n, "tag": tag, "index": i, "kind": o["kind"], "window_type": wt,
                          "label": lab, "width_mm": round(o["width_mm"]), "px": o["px"],
                          "horizontal": o["horizontal"], "wall_type": o["wall_type"]})
    cols = 6
    rows = (len(cells) + cols - 1) // cols
    sheet = np.full((rows * CELL, cols * CELL, 3), 255, np.uint8)
    for k, cell in enumerate(cells):
        r, cc = divmod(k, cols)
        sheet[r * CELL:(r + 1) * CELL, cc * CELL:(cc + 1) * CELL] = cell
    cv2.imencode(".png", sheet)[1].tofile(out)
    json.dump(index, open(out.rsplit(".", 1)[0] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(out, len(cells), "openings")


if __name__ == "__main__":
    main()
