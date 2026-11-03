"""List construction lines (color 9 / style 9) per layer with coordinates, plus overall bbox."""
import sys
from collections import defaultdict

import ezjww

sys.stdout.reconfigure(encoding="utf-8")
doc = ezjww.read_document(sys.argv[1])
ents = doc["entities"]
xs, ys = [], []
by_layer = defaultdict(list)
for e in ents:
    if e["type"] != "LINE":
        continue
    xs += [e["start_x"], e["end_x"]]
    ys += [e["start_y"], e["end_y"]]
    b = e["base"]
    if b["pen_color"] == 9 and b["pen_style"] == 9:
        by_layer[(b["layer_group"], b["layer"])].append(
            (round(e["start_x"], 2), round(e["start_y"], 2), round(e["end_x"], 2), round(e["end_y"], 2)))
print("LINE bbox:", min(xs), min(ys), max(xs), max(ys))
for k, v in sorted(by_layer.items()):
    print(f"G{k[0]:X}-L{k[1]:X}: {len(v)} lines")
    for s in sorted(v):
        print("   ", s)
