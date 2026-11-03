"""Cut a part the user drew in a corrected plan out into its own .jww, in library form.

usage: extract_user_part.py <corrected.jww> <meta.json> <out.jww> x0 y0 x1 y1 colors
                            origin_x origin_y along swing
  x0..y1   model-mm box around the part
  colors   pen colours that belong to the part, e.g. 2,3,4 (wall lines are 1 and 5)
  origin   model mm of the column centre the part is measured from (the block base point)
  along    model direction of the part's local +x: +x, -x, +y or -y (from the base column
           toward the part)
  swing    model direction the door swings to = local +y: +x, -x, +y or -y

Library form (same as build_parts_library.py): 1/50 paper mm, base point at the column centre on
the wall centre line, part along +x, swinging toward +y.
"""
import json
import math
import sys
from pathlib import Path

import ezjww

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_jww  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
DIRS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}


def main():
    src, meta_path, out = sys.argv[1:4]
    x0, y0, x1, y1 = map(float, sys.argv[4:8])
    colors = {int(c) for c in sys.argv[8].split(",")}
    ox, oy = float(sys.argv[9]), float(sys.argv[10])
    ex, ey = DIRS[sys.argv[11]]
    fx, fy = DIRS[sys.argv[12]]
    meta = json.load(open(meta_path, encoding="utf-8"))
    S, (cx, cy) = meta["scale"], meta["center_mm"]
    doc = ezjww.read_document(src)
    d = ezjww.new(paper_size=3)
    msp = d.modelspace()

    def loc(x, y):  # model mm -> local paper mm
        dx, dy = x - ox, y - oy
        return ((dx * ex + dy * ey) / S, (dx * fx + dy * fy) / S)

    def ang(t):  # model direction angle -> local angle (degrees)
        c, s = math.cos(t), math.sin(t)
        return math.degrees(math.atan2(c * fx + s * fy, c * ex + s * ey))

    mirror = ex * fy - ey * fx < 0  # local frame is a reflection of the model frame
    n = 0
    for e in doc["entities"]:
        b = e.get("base", {})
        if e["type"] not in ("LINE", "ARC") or b.get("pen_color") not in colors:
            continue
        attrs = {"layer": b["layer"], "pen_color": b["pen_color"], "pen_style": b["pen_style"]}
        if e["type"] == "LINE":
            p = (e["start_x"] * S + cx, e["start_y"] * S + cy)
            q = (e["end_x"] * S + cx, e["end_y"] * S + cy)
            if not all(x0 <= v[0] <= x1 and y0 <= v[1] <= y1 for v in (p, q)):
                continue
            msp.add_line(loc(*p), loc(*q), jwwattribs=attrs)
        else:
            c = (e["center_x"] * S + cx, e["center_y"] * S + cy)
            r = e["radius"]
            a0, sw = e["start_angle"], e["arc_angle"]
            ends = [(c[0] + r * S * math.cos(a), c[1] + r * S * math.sin(a)) for a in (a0, a0 + sw)]
            if not all(x0 <= v[0] <= x1 and y0 <= v[1] <= y1 for v in ends):
                continue
            start = ang(a0 + sw) if mirror else ang(a0)
            msp.add_arc(loc(*c), r, start_angle=start % 360, sweep_angle=math.degrees(sw), jwwattribs=attrs)
        n += 1
    d.saveas(out)
    print("saved", out, n, "entities")


if __name__ == "__main__":
    main()
