"""Debug: show wall segments per stage near given centers."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, sys.argv[3])
import wall_detect as wd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
cfg = json.load(open(sys.argv[2], encoding="utf-8"))
bgr = cv2.imdecode(np.fromfile(sys.argv[1], np.uint8), cv2.IMREAD_COLOR)
rgb = bgr[:, :, ::-1].copy()
mask, _ = wd.wall_mask(rgb, cfg)
h = wd.detect_direction(mask, cfg, True)
v = wd.detect_direction(mask, cfg, False)
print("raw H", len(h), "V", len(v))
for s in sorted(h, key=lambda s: s["c"]):
    print(f"  H c={s['c']:.1f} {s['a']:.0f}-{s['b']:.0f} t={s['t']} ext={s['ext']}")
for s in sorted(v, key=lambda s: s["c"]):
    print(f"  V c={s['c']:.1f} {s['a']:.0f}-{s['b']:.0f} t={s['t']} ext={s['ext']}")
b = wd.bridge_openings(h + v, cfg)
print("bridged", len(b))
j = wd.connect_junctions(b, cfg)
print("after junction filter", len(j))
