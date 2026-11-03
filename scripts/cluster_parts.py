"""List groups of touching lines/arcs on a maker sheet, to find plan (横断面) pieces automatically.

usage: cluster_parts.py <sheet.jww> [min_aspect]
Prints each flat group (width >= min_aspect x height): bbox in paper mm, width at the sheet's
own scale, and the nearest W= text. A plan piece is a flat group whose width matches a W= text.
"""
import re
import sys
import unicodedata

import ezjww

TOL = 0.25  # paper mm: pieces closer than this belong together
SKIP_LAYERS = ("POINT", "MOJI")  # maker reference points and dimension/text layers


def bbox(e):
    if e["type"] == "LINE":
        xs, ys = (e["start_x"], e["end_x"]), (e["start_y"], e["end_y"])
    elif e["type"] == "CIRCLE":
        r = e["radius"]
        xs, ys = (e["center_x"] - r, e["center_x"] + r), (e["center_y"] - r, e["center_y"] + r)
    elif e["type"] == "ARC":  # a swing arc must not join pieces: use its centre only
        xs, ys = (e["center_x"],) * 2, (e["center_y"],) * 2
    else:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def sheet(path):
    """(entities, layer names, sheet scale) of a maker sheet."""
    doc = ezjww.read_document(path)
    g = doc["header"]["layer_groups"][0]
    return doc["entities"], [l["name"] for l in g["layers"]], g["scale"]


def groups(path):
    """[(bbox, count)] of touching drawn pieces, and [(W, x, y)] of W= texts."""
    ents, names, _ = sheet(path)
    boxes = []
    for e in ents:
        if any(s in names[e["base"]["layer"]] for s in SKIP_LAYERS):
            continue
        b = bbox(e)
        # drop sheet frame/table lines (long lines spanning most of the sheet)
        if b and max(abs(v) for v in b) < 1e5 and (b[2] - b[0]) < 60 and (b[3] - b[1]) < 60:
            boxes.append(b)
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    for k, i in enumerate(order):
        bi = boxes[i]
        for j in order[k + 1:]:
            bj = boxes[j]
            if bj[0] > bi[2] + TOL:
                break
            if bj[1] <= bi[3] + TOL and bi[1] <= bj[3] + TOL:
                parent[find(i)] = find(j)
    gs = {}
    for i, b in enumerate(boxes):
        g = gs.setdefault(find(i), [1e9, 1e9, -1e9, -1e9, 0])
        g[0], g[1] = min(g[0], b[0]), min(g[1], b[1])
        g[2], g[3] = max(g[2], b[2]), max(g[3], b[3])
        g[4] += 1
    texts = []
    for e in ents:
        if e["type"] == "TEXT":
            t = unicodedata.normalize("NFKC", e.get("text") or e.get("content") or "").replace(" ", "")
            if re.match(r"W=\d+$", t):
                texts.append((int(t[2:]), e["start_x"], e["start_y"]))
    return [(g[:4], g[4]) for g in gs.values()], texts


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1]
    min_aspect = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    S = sheet(path)[2]
    gs, texts = groups(path)
    for g, n in sorted(gs, key=lambda t: (-round(t[0][1]), t[0][0])):
        w, h = g[2] - g[0], g[3] - g[1]
        if n < 4 or w < 3 or w < min_aspect * h:
            continue
        near = min(texts, key=lambda t: (t[1] - g[0]) ** 2 + (t[2] - g[1]) ** 2) if texts else None
        print(f"bbox [{g[0]:.1f}, {g[1]:.1f}, {g[2]:.1f}, {g[3]:.1f}]  W{w * S:.0f} D{h * S:.0f}  n{n}  near {near}")


if __name__ == "__main__":
    main()
