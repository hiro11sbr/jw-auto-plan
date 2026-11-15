"""出力した .jww を、透明の背景に白い線で描いた画像（PNG）にする（画面の「作成中」の右半分に出す）。
読むだけで、jww は変えない。壁・柱のある所にぴったり切り取る。円弧は傾きと扁平率も使う。"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ezjww  # noqa: E402


def render_png(jww_path, width=2000):
    d = ezjww.read_document(str(jww_path))
    ents = [e for e in d["entities"] if e["type"] in ("LINE", "ARC", "CIRCLE")]

    def strong(b):
        return b["layer"] == 3 and b["pen_color"] != 5

    def pts_of(e):
        if e["type"] == "LINE":
            return [(e["start_x"], e["start_y"]), (e["end_x"], e["end_y"])]
        r = e["radius"]
        return [(e["center_x"] - r, e["center_y"] - r), (e["center_x"] + r, e["center_y"] + r)]

    base = [p for e in ents if strong(e["base"]) for p in pts_of(e)] or [p for e in ents for p in pts_of(e)]
    if not base:
        return None
    xs, ys = [p[0] for p in base], [p[1] for p in base]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = max(x1 - x0, y1 - y0) * 0.04
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    k = width / (x1 - x0)
    h = max(1, int((y1 - y0) * k))
    im = np.zeros((h, width, 4), np.uint8)
    SH = 2  # 線をなめらかにするための小数ビット
    P = lambda x, y: (int(round((x - x0) * k * (1 << SH))), int(round((y1 - y) * k * (1 << SH))))

    def alpha(b):
        if b["layer"] == 4:
            return 70       # 通り芯
        if b["pen_color"] == 5:
            return 120      # ボード
        if b["layer"] == 3:
            return 235      # 壁・柱
        return 170          # 建具など

    for e in ents:
        b = e["base"]
        col = (255, 255, 255, alpha(b))
        th = 3 if strong(b) else 2
        if e["type"] == "LINE":
            cv2.line(im, P(e["start_x"], e["start_y"]), P(e["end_x"], e["end_y"]), col, th, cv2.LINE_AA, SH)
        else:
            c, r = (e["center_x"], e["center_y"]), e["radius"]
            if e["type"] == "CIRCLE" or e.get("is_full_circle"):
                a0, da = 0.0, 2 * math.pi
            else:
                a0, da = e["start_angle"], e.get("arc_angle", 0)
            tl, fl = e.get("tilt_angle", 0.0) or 0.0, e.get("flatness", 1.0) or 1.0
            ct, st = math.cos(tl), math.sin(tl)
            n = max(12, int(abs(da) * r * k / 3))
            pts = []
            for i in range(n + 1):
                th_ = a0 + da * i / n
                u, v = r * math.cos(th_), r * fl * math.sin(th_)
                pts.append(P(c[0] + u * ct - v * st, c[1] + u * st + v * ct))
            cv2.polylines(im, [np.array(pts, np.int32)], False, col, th, cv2.LINE_AA, SH)
    ok, buf = cv2.imencode(".png", im)
    return buf.tobytes() if ok else None


if __name__ == "__main__":
    data = render_png(sys.argv[1])
    Path(sys.argv[2]).write_bytes(data or b"")
    print(len(data or b""))
