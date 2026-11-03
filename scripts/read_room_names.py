"""Read room names from the plan image (before text is erased).

usage: read_room_names.py <plan.png> <config.json> <部屋名リスト.csv> <out_prefix>
writes <out_prefix>_部屋名.json  [{name, raw, kind, x, y (px of the original image), score}]

The image is prepared for OCR (red lettering removed, enlarged), read with Windows' built-in
Japanese OCR (ocr_windows.ps1), and every text line is matched against the room-name list so
small OCR mistakes still land on the right name.
"""
import csv
import difflib
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent


def norm(s):
    s = unicodedata.normalize("NFKC", s).replace(" ", "").replace("　", "")
    return s.upper()


def load_names(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    variants = []
    for r in rows:
        for v in r["表記ゆれ（|区切り）"].split("|"):
            if v:
                variants.append((norm(v), r["部屋名"], r["種類"]))
    return variants


OCR_MAX_PX = 9000  # Windows OCR refuses larger images (プラン3 1F 4370 px x 3 read nothing)


def ocr(img_bgr, scale):
    """OCR the image enlarged by `scale`; a plan too big for the OCR engine once enlarged is read
    in overlapping tiles, and lines seen twice in the overlap are kept once."""
    H, W = img_bgr.shape[:2]
    tile = int(OCR_MAX_PX / scale)
    if max(H, W) <= tile:
        return ocr_one(img_bgr, scale)
    pad = 200
    res = []
    for y0 in range(0, H, tile - pad):
        for x0 in range(0, W, tile - pad):
            part = img_bgr[y0:y0 + tile, x0:x0 + tile]
            for r in ocr_one(part, scale):
                b = r["box"]
                r["box"] = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
                if not any(q["text"] == r["text"] and abs(q["box"][0] - r["box"][0]) < 5
                           and abs(q["box"][1] - r["box"][1]) < 5 for q in res):
                    res.append(r)
            if x0 + tile >= W:
                break
        if y0 + tile >= H:
            break
    return res


def ocr_one(img_bgr, scale):
    big = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    out = tmp + ".json"
    cv2.imencode(".png", big)[1].tofile(tmp)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(HERE / "ocr_windows.ps1"), tmp, out], check=True, capture_output=True)
    data = json.load(open(out, encoding="utf-8-sig"))
    os.remove(tmp)
    os.remove(out)
    lines = data.get("lines") or []
    if isinstance(lines, dict):
        lines = [lines]
    res = []
    for ln in lines:
        words = ln.get("words") or []
        if isinstance(words, dict):
            words = [words]
        if not words:
            continue
        xs = [w["x"] for w in words] + [w["x"] + w["w"] for w in words]
        ys = [w["y"] for w in words] + [w["y"] + w["h"] for w in words]
        res.append({"text": ln["text"], "box": [min(xs) / scale, min(ys) / scale,
                                                 max(xs) / scale, max(ys) / scale]})
    return res


def merge_fragments(lines, gap):
    """OCR sometimes returns a word as several one-character lines (洗 / 面 / 所): join pieces
    on the same row whose boxes nearly touch."""
    lines = sorted(lines, key=lambda l: (round((l["box"][1] + l["box"][3]) / 2 / 6), l["box"][0]))
    out = []
    for ln in lines:
        if out:
            p = out[-1]
            same_row = abs((p["box"][1] + p["box"][3]) / 2 - (ln["box"][1] + ln["box"][3]) / 2) <= 6
            if same_row and 0 <= ln["box"][0] - p["box"][2] <= gap:
                p["text"] += ln["text"]
                p["box"] = [min(p["box"][0], ln["box"][0]), min(p["box"][1], ln["box"][1]),
                            max(p["box"][2], ln["box"][2]), max(p["box"][3], ln["box"][3])]
                continue
        out.append(dict(ln))
    return out


def best_match(text, variants, min_score):
    t = norm(text)
    # strip trailing area like (18.7帖) / 5.2帖
    for cut in ("(", "（"):
        if cut in t:
            t = t.split(cut)[0]
    if not t:
        return None
    best = None
    for v, name, kind in variants:
        if v == t and (len(v) >= 2 or len(t) == 1 and False):
            return (name, kind, 1.0, v)
        if len(v) < 2 or len(t) < 2:
            continue  # single letters (L, D, ト ...) are too ambiguous to match
        sc = difflib.SequenceMatcher(None, t, v).ratio()
        # names written with a prefix/suffix (洋室(A), ぬくぬくヌック, トイレ...) count as contained
        if len(v) >= 2 and (v in t or t in v):
            sc = max(sc, 0.9 if v in t else 0.8)
        if best is None or sc > best[2]:
            best = (name, kind, sc, v)
    return best if best and best[2] >= min_score else None


def main():
    img_path, cfg_path, names_path, prefix = sys.argv[1:5]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    c = cfg["room_names"]
    variants = load_names(names_path)
    bgr = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    # red lettering (dimensions, fitting codes) confuses the OCR: paint it white first
    r, g, b = (bgr[..., 2].astype(int), bgr[..., 1].astype(int), bgr[..., 0].astype(int))
    red = (r - g > 25) & (r - b > 25) & (np.abs(g - b) <= 30)
    prep = bgr.copy()
    prep[red] = 255
    # several readings of the same image (as is / black-and-white / other scale): each one
    # catches names the others miss; results are merged by position
    gray = cv2.cvtColor(prep, cv2.COLOR_BGR2GRAY)
    bw = cv2.cvtColor(np.where(gray < c["binarize_max"], 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # only the darkest strokes: names written over floor tiles (玄関・土間) whose grid lines
    # break the other readings (プラン2・3)
    dark_only = cv2.cvtColor(np.where(gray < c["dark_text_max"], 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    lines = []
    for im, sc in ((prep, c["ocr_scale"]), (bw, c["ocr_scale"]), (prep, c["ocr_scale"] + 1),
                   (dark_only, c["ocr_scale"])):
        lines += merge_fragments(ocr(im, sc), c["merge_gap_px"])
    found = []
    for ln in lines:
        m = best_match(ln["text"], variants, c["match_min_score"])
        x0, y0, x1, y1 = ln["box"]
        item = {"raw": ln["text"], "x": round((x0 + x1) / 2, 1), "y": round((y0 + y1) / 2, 1),
                "box": [round(v, 1) for v in ln["box"]]}
        if m:
            item.update({"name": m[0], "kind": m[1], "score": round(m[2], 2), "matched": m[3]})
        # the same text read in several passes: keep the best-scoring one per position
        dup = next((f for f in found if abs(f["x"] - item["x"]) < 20 and abs(f["y"] - item["y"]) < 10), None)
        if dup:
            if item.get("score", 0) > dup.get("score", 0):
                found[found.index(dup)] = item
            continue
        found.append(item)
    json.dump(found, open(prefix + "_部屋名.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rooms = [f for f in found if f.get("name") and f["kind"] != "除外語"]
    name = prefix.replace("\\", "/").split("/")[-1]
    print(f"{name}: OCR lines={len(found)} room names={len(rooms)}")
    for f in found:
        print(f"   {f['raw']!r:30} -> {f.get('name', '-')} ({f.get('kind', '')} {f.get('score', '')}) at {f['x']},{f['y']}")


if __name__ == "__main__":
    main()
