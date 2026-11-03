"""Print RGB along a vertical or horizontal pixel line: probe_pixels.py img x|y fixed from to"""
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
img = cv2.imdecode(np.fromfile(sys.argv[1], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
axis, fixed, a, b = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
for i in range(a, b):
    p = img[i, fixed] if axis == "x" else img[fixed, i]
    print(i, tuple(int(v) for v in p))
