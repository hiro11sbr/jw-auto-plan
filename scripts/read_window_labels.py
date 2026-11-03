"""Read the window code printed beside each window (e.g. 16020H透明, 06011T, 16011TF, 11905F型).

usage: read_window_labels.py <plan.png> <openings.json> <out_prefix>
writes <out_prefix>_窓の記号.json  {opening index: {raw, code, letters, type, width_mm}} and adds
"label" to each window in openings.json (the detection itself is not changed).

The code is LIXIL-style: 3 digits of width code, 2 (or 3) of height, then the type letters.
width_mm is the frame outside width looked up from the catalogue table (WIDTH_CODE).
  H = 引違い, T = 縦すべり, F = FIX, TF = 縦すべり+FIX (連窓).
Steps: find each line of red lettering, read it several times (different enlargements), take the
code most often read, then give each window the nearest code (one code per window).
Windows whose code could not be read keep the default type and are reported as 読めず.
"""
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from read_room_names import ocr  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
TYPES = {"H": "引違い窓", "T": "縦すべり窓", "F": "FIX窓", "TF": "縦すべり+FIX"}
# width code -> frame outside width (枠外W), from the catalogue table in 03_建具部品\窓寸法の対応表.md.
# Looked up, never computed (user 10/28: a formula gets the exceptions wrong). 060 is 640, not 600
# (例1 2F: 06011T was taken as 600 and got W500, user 10/30)
WIDTH_CODE = {"021": 250, "026": 300, "036": 400, "046": 500, "060": 640, "069": 730, "074": 780,
              "096": 1000, "114": 1185, "116": 1200, "119": 1235, "146": 1500, "150": 1540, "160": 1640,
              "165": 1690, "243": 2470, "251": 2550, "256": 2600}


def frame_width(digits):
    """枠外W of a code's first 3 digits; codes missing from the table keep the old reading (cm)."""
    return WIDTH_CODE.get(digits[:3], int(digits[:3]) * 10)
DEBUG = bool(os.environ.get("WL_DEBUG"))
MAX_DIST = 220  # px: a label farther than this from a window is not its label
# common Windows-OCR misreads of these labels
FIX_TEXT = [("()", "0"), ("(", "0"), (")", "0"), ("O", "0"), ("D", "0"), ("ⅱ", "11"), ("Ⅱ", "11"),
            ("I", "1"), ("L", "1"), ("|", "1"), ("丁", "T"), ("P", "F")]


def red_only(bgr):
    """Keep reddish pixels in their own colour (anti-aliased edges help the OCR), rest white."""
    b, g, r = [bgr[..., i].astype(int) for i in range(3)]
    mask = (r - np.maximum(g, b)) > 15
    return np.where(mask[..., None], bgr, 255).astype(np.uint8)


def parse(text):
    t = unicodedata.normalize("NFKC", text).replace(" ", "").upper()
    for a, b in FIX_TEXT:
        t = t.replace(a, b)
    m = re.search(r"(\d{5})(TF|T|F|H)", t)
    if not m:
        return None
    w, h = int(m.group(1)[:3]), int(m.group(1)[3:])
    # plausible window sizes only (width 25-260 cm, height 3-23 dm)
    return (m.group(1), m.group(2)) if 25 <= w <= 260 and 3 <= h <= 23 else None


def text_lines(bgr):
    """Boxes (x0, y0, x1, y1) of lines of red lettering."""
    b, g, r = [bgr[..., i].astype(int) for i in range(3)]
    m = ((r - np.maximum(g, b)) > 30).astype(np.uint8)
    # bridge the gaps left where black lines cross the lettering
    d = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    n, _lab, st, _ = cv2.connectedComponentsWithStats(d)
    return [(x, y, x + w, y + h) for x, y, w, h, _a in st[1:] if w >= 30 and 8 <= h <= 30]


def read_codes(bgr, wins):
    """Read every red label several ways and vote: each line of lettering alone (4 enlargements),
    the whole plan at once, and the area on both sides of each window."""
    red = red_only(bgr)
    H, W = bgr.shape[:2]
    reads = []  # (centre x, centre y, raw text)

    def run(img, x0, y0, scale, pad=0):
        for ln in ocr(img, scale):
            bx0, by0, bx1, by1 = ln["box"]
            reads.append(((bx0 + bx1) / 2 + x0 - pad, (by0 + by1) / 2 + y0 - pad, ln["text"]))

    for x0, y0, x1, y1 in text_lines(bgr):
        xa, ya = max(0, x0 - 6), max(0, y0 - 4)
        crop = red[ya:min(H, y1 + 4), xa:min(W, x1 + 6)]
        crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for s in (3, 4, 5, 6):
            run(crop, xa, ya, s, pad=12)
    run(red, 0, 0, 3)
    for _i, o in wins:
        p0, p1, c = o["px"]
        c = int(round(c))
        for side in (-1, 1):
            if o["horizontal"]:
                x0, x1 = p0 - 90, p1 + 90
                y0, y1 = (c - 95, c - 6) if side < 0 else (c + 6, c + 95)
            else:
                y0, y1 = p0 - 90, p1 + 90
                x0, x1 = (c - 155, c - 6) if side < 0 else (c + 6, c + 155)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            run(red[y0:y1, x0:x1], x0, y0, 4)
    # group readings of the same label (centres within 25 px) and vote
    groups = []
    for x, y, t in reads:
        r = parse(t)
        if not r:
            continue
        for g in groups:
            if abs(g["x"] - x) < 25 and abs(g["y"] - y) < 25:
                g["votes"][r] += 1
                g["raws"].append(t)
                break
        else:
            groups.append({"x": x, "y": y, "votes": Counter([r]), "raws": [t]})
    out = []
    for g in groups:
        (digits, letters), n = g["votes"].most_common(1)[0]
        if DEBUG:
            print("   label", round(g["x"]), round(g["y"]), dict(g["votes"]))
        out.append({"box": (g["x"], g["y"], g["x"], g["y"]), "digits": digits, "letters": letters,
                    "raw": " / ".join(g["raws"]), "votes": n})
    return out


def main():
    img_path, ops_path, prefix = sys.argv[1:4]
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    ops = json.load(open(ops_path, encoding="utf-8"))
    wins = [(i, o) for i, o in enumerate(ops) if not o.get("rejected") and o["kind"] == "window"]
    codes = read_codes(bgr, wins)
    pairs = []
    for i, o in wins:
        o.pop("label", None)
        p0, p1, c = o["px"]
        mx, my = ((p0 + p1) / 2, c) if o["horizontal"] else (c, (p0 + p1) / 2)
        for k, cd in enumerate(codes):
            x0, y0, x1, y1 = cd["box"]
            d = float(np.hypot((x0 + x1) / 2 - mx, (y0 + y1) / 2 - my))
            if d <= MAX_DIST:
                pairs.append((d, i, k))
    found, used = {}, set()
    for d, i, k in sorted(pairs):
        if i in found or k in used:
            continue
        cd = codes[k]
        found[i] = {"raw": cd["raw"], "code": cd["digits"] + cd["letters"], "letters": cd["letters"],
                    "type": TYPES[cd["letters"]], "width_mm": frame_width(cd["digits"])}
        ops[i]["label"] = found[i]
        used.add(k)
    for i, o in wins:
        lab = found.get(i)
        print(i, "横" if o["horizontal"] else "縦", round(o["width_mm"]),
              f'{lab["code"]} {lab["type"]}' if lab else "読めず")
    json.dump(found, open(prefix + "_窓の記号.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(ops, open(ops_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
