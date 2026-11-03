"""Diff two JWW files by LINE entities (model mm, rounded): what the user removed / added."""
import json
import sys

import ezjww

sys.stdout.reconfigure(encoding="utf-8")


def lines(path, S, center):
    cx, cy = center
    out = {}
    for e in ezjww.read_document(path)["entities"]:
        if e["type"] != "LINE":
            continue
        b = e["base"]
        p = (round(e["start_x"] * S + cx, 1), round(e["start_y"] * S + cy, 1))
        q = (round(e["end_x"] * S + cx, 1), round(e["end_y"] * S + cy, 1))
        key = (b["layer_group"], b["layer"], b["pen_color"], b["pen_style"], *sorted((p, q)))
        out[key] = out.get(key, 0) + 1
    return out


meta = json.load(open(sys.argv[3], encoding="utf-8"))
a = lines(sys.argv[1], meta["scale"], meta["center_mm"])
b = lines(sys.argv[2], meta["scale"], meta["center_mm"])
removed = sorted(k for k in a if k not in b)
added = sorted(k for k in b if k not in a)
print(f"before={sum(a.values())} after={sum(b.values())} removed={len(removed)} added={len(added)}")
for name, s in (("removed", removed), ("added", added)):
    for g, l, c, st, p, q in s:
        print(f"  {name}: L{l:X} color{c} style{st} {p} -> {q}")
