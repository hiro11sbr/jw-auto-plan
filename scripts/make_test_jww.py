"""Write ezjww_test.jww using the reference drawing layer structure (group 0, 1/50)."""
import sys

import ezjww

sys.stdout.reconfigure(encoding="utf-8")
out = sys.argv[1]
S = 50.0  # 1/50

# reference drawing group 0 layer names
LAYER_NAMES = ["境界線", "敷地外部", "外構", "マーク、物", "中心線、建物基礎", "CYAN", "ORANGE", "PINK",
               "WHITE", "YELLOW", "文字", "ﾊｯﾁﾝｸﾞ", "ﾚｲﾔ-12", "ﾚｲﾔ-13", "ﾚｲﾔ-14", "図面枠"]
L_WALL, L_CENTER, L_FRAME = 0x3, 0x4, 0xF

d = ezjww.new(paper_size=3)
g0 = d.options["layer_groups"][0]
g0["name"] = ""
g0["scale"] = S
for i, name in enumerate(LAYER_NAMES):
    g0["layers"][i]["name"] = name
msp = d.modelspace()


def model_line(p, q, layer, color, style=1):
    msp.add_line((p[0] / S, p[1] / S), (q[0] / S, q[1] / S),
                 jwwattribs={"layer": layer, "pen_color": color, "pen_style": style})


def paper_line(p, q):
    msp.add_line(p, q, jwwattribs={"layer": L_FRAME, "pen_color": 9, "pen_style": 9})


# wall center 0..3640 (color 3 / style 5), column face +-52.5 (color 1), board face +-65 (color 5)
model_line((0, 0), (3640, 0), L_CENTER, 3, 5)
for off, col in ((52.5, 1), (-52.5, 1), (65.0, 5), (-65.0, 5)):
    model_line((0, off), (3640, off), L_WALL, col)
# 105 column with both diagonals at x=0
h = 52.5
c = [(-h, -h), (h, -h), (h, h), (-h, h)]
for i in range(4):
    model_line(c[i], c[(i + 1) % 4], L_WALL, 1)
model_line(c[0], c[2], L_WALL, 1)
model_line(c[1], c[3], L_WALL, 1)

# print alignment marks (paper mm, A3 center origin)
ARM = 15.9
for X, Y in ((205, 144), (209, 147)):  # ▽A3 / ▽A3範囲拡大: 4-corner L
    for sx in (1, -1):
        for sy in (1, -1):
            corner = (sx * X, sy * Y)
            paper_line(corner, (sx * (X - ARM), sy * Y))
            paper_line(corner, (sx * X, sy * (Y - ARM)))
W, H = 210, 148.5  # △A3 PDF: paper edge outline
pts = [(-W, -H), (W, -H), (W, H), (-W, H)]
for i in range(4):
    paper_line(pts[i], pts[(i + 1) % 4])

d.saveas(out)
print("saved", out)
