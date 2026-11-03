"""出力した .jww をそのまま画像にする（CAD図の見た目の確認用）。

usage: render_jww.py <in.jww> <out.png> [px_per_mm] [title]
線色ごとに色分け（線色1＝黒、3＝緑、5＝紫、9＝灰）。ブロック（建具）は展開して描く。
"""
import math
import sys
from pathlib import Path

import cv2
import ezjww
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jp_text  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
COLORS = {1: (0, 0, 0), 2: (0, 0, 0), 3: (0, 140, 0), 4: (200, 120, 0), 5: (180, 0, 180),
          6: (0, 0, 200), 7: (0, 120, 200), 8: (120, 120, 120), 9: (170, 170, 170)}


def main():
    src, out = sys.argv[1:3]
    k = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    title = sys.argv[4] if len(sys.argv) > 4 else Path(src).stem
    # optional background: the plan image, with its walls.json (grid fit) and meta.json (centre)
    bg = sys.argv[5:8] if len(sys.argv) > 7 else None
    d = ezjww.read_document(src)
    blocks = {b["name"]: b["entities"] for b in d["block_defs"]}
    if bg:
        import json
        from opening_detect import Mapper
        img_path, walls_path, meta_path = bg
        img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        img = (img.astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)  # faded plan
        det = json.load(open(walls_path, encoding="utf-8"))
        meta = json.load(open(meta_path, encoding="utf-8"))
        M = Mapper(det, 455)
        S, (cx, cy) = meta["scale"], meta["center_mm"]
        H, W = img.shape[:2]

        def px(x, y):
            return int(round(M.x(x * S + cx))), int(round(M.y(y * S + cy)))
    else:
        W, H = int(420 * k), int(297 * k)
        img = np.full((H, W, 3), 255, np.uint8)

        def px(x, y):
            return int(round(W / 2 + x * k)), int(round(H / 2 - y * k))

    def draw(e, tf):
        col = COLORS.get(e["base"]["pen_color"], (0, 0, 0))
        th = 1
        if e["type"] == "LINE":
            p, q = tf(e["start_x"], e["start_y"]), tf(e["end_x"], e["end_y"])
            cv2.line(img, px(*p), px(*q), col, th, cv2.LINE_AA)
        elif e["type"] in ("ARC", "CIRCLE"):
            r = e["radius"]
            a0 = e.get("start_angle", 0.0)
            sweep = e.get("arc_angle", 2 * math.pi) or 2 * math.pi
            n = max(8, int(abs(sweep) * r * k / 2))
            pts = [px(*tf(e["center_x"] + r * math.cos(a0 + sweep * i / n),
                          e["center_y"] + r * math.sin(a0 + sweep * i / n))) for i in range(n + 1)]
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, th, cv2.LINE_AA)

    ident = lambda x, y: (x, y)  # noqa: E731
    for e in d["entities"]:
        if e["type"] == "BLOCK":
            rot = e.get("rotation", 0.0)
            if abs(rot) > 2 * math.pi + 1e-6:  # degrees in some files
                rot = math.radians(rot)
            c, s = math.cos(rot), math.sin(rot)

            def tf(x, y, e=e, c=c, s=s):
                x, y = x * e["scale_x"], y * e["scale_y"]
                return e["ref_x"] + c * x - s * y, e["ref_y"] + s * x + c * y
            for be in blocks.get(e["block_name"], []):
                draw(be, tf)
        else:
            draw(e, ident)
    img = jp_text.put(img, [(title, (20, 14), (0, 0, 0), 30)])
    cv2.imencode(".png", img)[1].tofile(out)
    print(out)


if __name__ == "__main__":
    main()
