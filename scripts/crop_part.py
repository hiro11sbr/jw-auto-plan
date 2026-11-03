"""Render the lines/arcs of a sheet inside a bbox (paper mm) to a PNG, with a 1 mm grid.

usage: crop_part.py <sheet.jww> x0 y0 x1 y1 <out.png> [px_per_mm]
"""
import math
import sys

import cv2
import ezjww
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
path = sys.argv[1]
x0, y0, x1, y1 = map(float, sys.argv[2:6])
out = sys.argv[6]
S = float(sys.argv[7]) if len(sys.argv) > 7 else 40
ents = ezjww.read_document(path)["entities"]
img = np.full((int((y1 - y0) * S) + 1, int((x1 - x0) * S) + 1, 3), 255, np.uint8)


def P(x, y):
    return int(round((x - x0) * S)), int(round((y1 - y) * S))


for gx in range(math.ceil(x0), int(x1) + 1):
    cv2.line(img, P(gx, y0), P(gx, y1), (235, 225, 210) if gx % 5 else (220, 180, 150), 1)
for gy in range(math.ceil(y0), int(y1) + 1):
    cv2.line(img, P(x0, gy), P(x1, gy), (235, 225, 210) if gy % 5 else (220, 180, 150), 1)
n = 0
for e in ents:
    if e["type"] == "LINE":
        xs, ys = (e["start_x"], e["end_x"]), (e["start_y"], e["end_y"])
        if x0 <= min(xs) and max(xs) <= x1 and y0 <= min(ys) and max(ys) <= y1:
            cv2.line(img, P(xs[0], ys[0]), P(xs[1], ys[1]), (0, 0, 0), 1, cv2.LINE_AA)
            n += 1
    elif e["type"] in ("ARC", "CIRCLE"):
        cx, cy, r = e["center_x"], e["center_y"], e["radius"]
        if x0 <= cx - r and cx + r <= x1 + 50 and y0 - 50 <= cy - r and cy + r <= y1 + 50 and x0 <= cx <= x1 and y0 <= cy <= y1:
            a0 = math.degrees(e.get("start_angle", 0) or 0)
            sw = math.degrees(e.get("arc_angle", 2 * math.pi) or 2 * math.pi)
            pts = [P(cx + r * math.cos(math.radians(a0 + sw * t / 40)), cy + r * math.sin(math.radians(a0 + sw * t / 40)))
                   for t in range(41)]
            cv2.polylines(img, [np.array(pts)], False, (0, 0, 200), 1, cv2.LINE_AA)
            n += 1
cv2.imencode(".png", img)[1].tofile(out)
print(out, n, "entities")
