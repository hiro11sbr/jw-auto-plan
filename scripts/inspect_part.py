"""Inspect a JWW part file: layers, entity types, blocks, texts, and render a PNG preview.

usage: inspect_part.py <file.jww> <out.png>
"""
import io
import sys
from collections import Counter

import cv2
import ezdxf
import ezjww
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
path, out_png = sys.argv[1], sys.argv[2]

doc = ezjww.read_document(path)
hdr = doc["header"]
print("version", hdr.get("version"), "paper", hdr.get("paper_size"))
for gi, g in enumerate(hdr["layer_groups"]):
    if g.get("state", 0):
        print(f"  G{gi:X} name={g.get('name')!r} scale={g.get('scale')}")
types = Counter(e["type"] for e in doc["entities"])
print("entities:", dict(types))
by_layer = Counter((e["base"]["layer_group"], e["base"]["layer"]) for e in doc["entities"])
print("by group-layer:", {f"{g:X}-{l:X}": n for (g, l), n in sorted(by_layer.items())})
print("blocks:", len(doc.get("blocks", []) or []))
for b in (doc.get("blocks") or [])[:20]:
    print("   block:", b.get("name"), "entities:", len(b.get("entities", [])))
texts = [e for e in doc["entities"] if e["type"] == "TEXT"]
print("texts (first 40):")
for t in texts[:40]:
    print("   ", repr(t.get("text") or t.get("content")), round(t.get("start_x", 0), 1), round(t.get("start_y", 0), 1))

# render via DXF (blocks exploded)
tmp = out_png + ".dxf"
ezjww.write_dxf(path, tmp)
d = ezdxf.readfile(tmp)
segs = []


def flat(entities):
    for e in entities:
        if e.dxftype() == "INSERT":  # block references: draw their contents
            yield from flat(e.virtual_entities())
        else:
            yield e


for e in flat(d.modelspace()):
    t = e.dxftype()
    if t == "LINE":
        segs.append([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)])
    elif t in ("ARC", "CIRCLE"):
        c, r = e.dxf.center, e.dxf.radius
        a0, a1 = (e.dxf.start_angle, e.dxf.end_angle) if t == "ARC" else (0, 360)
        if a1 < a0:
            a1 += 360
        ang = np.radians(np.linspace(a0, a1, 24))
        pts = [(c.x + r * np.cos(a), c.y + r * np.sin(a)) for a in ang]
        segs += [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
    elif t in ("LWPOLYLINE", "POLYLINE"):
        pts = [(p[0], p[1]) for p in (e.get_points() if t == "LWPOLYLINE" else e.points())]
        segs += [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
if not segs:
    print("nothing to render")
    sys.exit()
a = np.array(segs)
# drop stray far-away entities (some files carry a point at 1e20) and use robust extents
ok = (np.abs(a) < 1e6).all(axis=(1, 2))
a = a[ok]
segs = a.tolist()
pts = a.reshape(-1, 2)
mn, mx = np.percentile(pts, 0.2, axis=0), np.percentile(pts, 99.8, axis=0)
print("extent:", mn.round(1), mx.round(1))
W = 2400
s = (W - 40) / max(mx[0] - mn[0], 1e-6)
H = int((mx[1] - mn[1]) * s) + 40
img = np.full((H, W, 3), 255, np.uint8)
for (x1, y1), (x2, y2) in segs:
    p = (int((x1 - mn[0]) * s) + 20, int(H - 20 - (y1 - mn[1]) * s))
    q = (int((x2 - mn[0]) * s) + 20, int(H - 20 - (y2 - mn[1]) * s))
    cv2.line(img, p, q, (0, 0, 0), 1, cv2.LINE_AA)
cv2.imencode(".png", img)[1].tofile(out_png)
print("png", W, H)
