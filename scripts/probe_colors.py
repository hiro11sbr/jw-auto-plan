"""Histogram of neutral-gray pixel values and sample wall band widths."""
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
img = cv2.imread(sys.argv[1])[:, :, ::-1].astype(int)  # RGB
r, g, b = img[..., 0], img[..., 1], img[..., 2]
neutral = (np.abs(r - g) < 8) & (np.abs(g - b) < 8)
v = img.mean(axis=2).astype(int)
hist = np.bincount(v[neutral].ravel(), minlength=256)
print("neutral gray histogram (value: count), 120-250:")
for i in range(120, 251, 2):
    c = hist[i] + hist[i + 1]
    if c > 500:
        print(f"  {i:3d}-{i+1:3d}: {c:7d} " + "#" * min(80, c // 2000))

# scan a few rows/cols and print runs of the dominant wall band
lo, hi = int(sys.argv[2]), int(sys.argv[3])
mask = neutral & (v >= lo) & (v <= hi)
for y in map(int, sys.argv[4].split(",")):
    row = mask[y]
    runs, x = [], 0
    while x < len(row):
        if row[x]:
            s = x
            while x < len(row) and row[x]:
                x += 1
            if x - s >= 4:
                runs.append((s, x - s))
        x += 1
    print(f"row {y}: runs(start,len) =", runs)
cv2.imwrite(sys.argv[5], (mask * 255).astype(np.uint8))
