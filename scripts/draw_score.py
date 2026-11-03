"""Draw a wall judgement against the answer on the plan image, per 227.5 unit:
green = right, red = detected but not in the answer, blue = in the answer but missed.

usage: draw_score.py <plan.png> <detected_walls.json> <correct.jww> <meta.json> <config.json> <out.png> <title>
"""
import json
import sys
from pathlib import Path

import cv2
import ezjww
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jp_text  # noqa: E402
from opening_detect import Mapper  # noqa: E402
from score_walls import unit_edges  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
img_path, det_path, ans_path, meta_path, cfg_path, out, title = sys.argv[1:8]
cfg = json.load(open(cfg_path, encoding="utf-8"))
det = json.load(open(det_path, encoding="utf-8"))
meta = json.load(open(meta_path, encoding="utf-8"))
grid = cfg["input"]["half_grid_mm"]
M = Mapper(det, cfg["input"]["grid_mm"])
S, (cx, cy) = meta["scale"], meta["center_mm"]
pred = set()
for w in det["walls"]:
    pred |= unit_edges(w["p1"], w["p2"], grid)
truth = set()
st = cfg["layers"]["center"]
for e in ezjww.read_document(ans_path)["entities"]:
    b = e["base"]
    if e["type"] == "LINE" and b["layer"] == st["layer"] and b["pen_style"] == st["style"]:
        truth |= unit_edges((e["start_x"] * S + cx, e["start_y"] * S + cy), (e["end_x"] * S + cx, e["end_y"] * S + cy), grid)
bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
img = cv2.addWeighted(bgr, 0.45, np.full_like(bgr, 255), 0.55, 0)


def px(h, ln, s):
    a, b = s * grid, (s + 1) * grid
    if h:
        return (int(M.x(a)), int(M.y(ln * grid))), (int(M.x(b)), int(M.y(ln * grid)))
    return (int(M.x(ln * grid)), int(M.y(a))), (int(M.x(ln * grid)), int(M.y(b)))


for keys, col in ((pred & truth, (0, 160, 0)), (pred - truth, (0, 0, 230)), (truth - pred, (230, 120, 0))):
    for k in keys:
        p, q = px(*k)
        cv2.line(img, p, q, col, 3)
tp, fp, fn = len(pred & truth), len(pred - truth), len(truth - pred)
prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
f1 = 2 * prec * rec / max(1e-9, prec + rec)
img = jp_text.legend(img, f"{title}　F1 {f1:.3f}", [((0, 160, 0), f"正しく検知 {tp}"),
                                                   ((0, 0, 230), f"誤検出（正解にない） {fp}"),
                                                   ((230, 120, 0), f"見落とし（正解にある） {fn}")])
cv2.imencode(".png", img)[1].tofile(out)
print(out, tp, fp, fn, round(f1, 3))
