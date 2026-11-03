"""Summarize layer structure, scales and entity attributes of a JWW file."""
import sys
from collections import Counter

import ezjww

sys.stdout.reconfigure(encoding="utf-8")
path = sys.argv[1]
doc = ezjww.read_document(path)
hdr = doc["header"]
print("== header keys:", list(hdr.keys()))
print("paper_size:", hdr.get("paper_size"), "write_group:", hdr.get("write_layer_group"))
for gi, g in enumerate(hdr["layer_groups"]):
    names = [l.get("name") for l in g["layers"]]
    print(f"[G{gi:X}] name={g.get('name')!r} scale={g.get('scale')} state={g.get('state')} write={g.get('write_layer')}")
    print("     layers:", {f"{i:X}": n for i, n in enumerate(names) if n})

ents = doc["entities"]
print("== entity count:", len(ents))
print("first entity:", ents[0])
c = Counter()
for e in ents:
    b = e.get("base", e)
    c[(b.get("layer_group"), b.get("layer"), e.get("type"), b.get("pen_color"), b.get("pen_style"))] += 1
print("== (group, layer, type, color, style): count")
for k, v in sorted(c.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1])):
    print(f"  G{k[0]:X}-L{k[1]:X} {k[2]:<8} color={k[3]} style={k[4]} : {v}")
