"""Step 0: find and erase text on the plan image before walls / openings are judged.

usage: clean_text.py <plan.png> <config.json> <out_prefix>
writes <out_prefix>_文字削除後.png (cleaned image used by the later steps) and
       <out_prefix>_文字の検出.png (check image: what was erased)

- red text (dimensions, fitting codes): picked by colour
- black text (room names, numbers): small blobs that sit close together in a word; a row of
  dashes in one straight line is a dashed drawing line (e.g. a sliding-door pocket) and is kept
- erased pixels are filled from their surroundings (gray on a wall, white on the floor)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
import jp_text  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def red_mask(rgb, c):
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    # red / pink lettering has green ~ blue; orange or brown label boxes (物入, 可動棚) do not,
    # and erasing those smears the wall next to them
    reddish = np.abs(g - b) <= c["red_max_gb_diff"]
    strong = (r - g > c["red_min_diff"]) & (r - b > c["red_min_diff"])
    faint = (r - g > c["red_faint_diff"]) & (r - b > c["red_faint_diff"]) & (r > 120)
    m = ((strong | faint) & reddish).astype(np.uint8)
    # solid filled blobs are coloured symbols, not letters
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w * h >= c["red_blob_min_area"] and area / (w * h) >= 0.6:
            m[lab == i] = 0
    return m.astype(bool)


def black_text_mask(rgb, c):
    v = rgb.astype(int).mean(axis=2)
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    neutral = (np.abs(r - g) <= 25) & (np.abs(g - b) <= 25)
    dark = ((v < c["dark_max"]) & neutral).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    glyph = np.zeros(n, bool)
    dash = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        big = max(w, h)
        if big > c["glyph_max_px"] or area < c["glyph_min_area"]:
            continue
        thin = min(w, h) <= 2
        if thin and big >= 15:
            continue  # a short straight line, not a letter
        glyph[i] = True
        dash[i] = thin and big >= 3 and big / max(min(w, h), 1) >= 2.5
    # group glyphs that sit close together into words
    gmask = np.isin(lab, np.flatnonzero(glyph)).astype(np.uint8)
    k = c["word_join_px"]
    joined = cv2.dilate(gmask, np.ones((k, k), np.uint8))
    nw, wlab, wstats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    text = np.zeros_like(gmask, bool)
    kept_dashes = 0
    for j in range(1, nw):
        members = np.unique(lab[(wlab == j) & (gmask == 1)])
        members = members[members > 0]
        if len(members) < c["word_min_glyphs"]:
            continue
        # a row of dashes on one straight line is a dashed drawing line: keep it
        if dash[members].mean() >= 0.7:
            ys = stats[members, 1] + stats[members, 3] / 2
            xs = stats[members, 0] + stats[members, 2] / 2
            if np.ptp(ys) <= 2 or np.ptp(xs) <= 2:
                kept_dashes += 1
                continue
        text |= np.isin(lab, members)
    return text, kept_dashes


def main():
    img_path, cfg_path, prefix = sys.argv[1:4]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    c = cfg["text_cleaning"]
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    rgb = bgr[:, :, ::-1]
    red = red_mask(rgb, c)
    black, kept = black_text_mask(rgb, c)
    mask = (red | black).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    clean = cv2.inpaint(bgr, mask * 255, c["inpaint_radius"], cv2.INPAINT_TELEA)
    cv2.imencode(".png", clean)[1].tofile(prefix + "_文字削除後.png")

    check = cv2.addWeighted(bgr, 0.5, np.full_like(bgr, 255), 0.5, 0)
    check[red] = (0, 0, 230)
    check[black] = (230, 120, 0)
    name = prefix.replace("\\", "/").split("/")[-1]
    check = jp_text.legend(check, f"{name}　文字の検出", [((0, 0, 230), "赤い文字（消した）"),
                                                       ((230, 120, 0), "黒い文字（消した）")])
    cv2.imencode(".png", check)[1].tofile(prefix + "_文字の検出.png")
    print(f"{name}: red px={int(red.sum())} black text px={int(black.sum())} dashed lines kept={kept}")


if __name__ == "__main__":
    main()
