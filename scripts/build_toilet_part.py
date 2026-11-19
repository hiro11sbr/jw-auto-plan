"""TOTO GG（ウォシュレット一体形便器）の平面図 → 便器の部品 03_建具部品/TOTO/便器_GG/便器_GG.json。

usage: build_toilet_part.py
ZJ1 のデータが TOTO の一括ZIPに無いため、GG の意匠図 平面図（G312898-B1.jww、CES9C10 などと共通）で代用
（user 11/13）。実寸mm、原点＝便器全体の中心、+x＝前（便器の丸い側）、+y＝左。図形は線・円弧・円。
"""
import glob
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ezjww  # noqa: E402

ROOT = HERE.parent
src = glob.glob(str(ROOT / "03_建具部品/TOTO/トイレ_jww/トイレ/住宅向け腰掛便器/GG*/G312898-B1.jww"))[0]
doc = ezjww.read_document(src)
S = doc["header"]["layer_groups"][0]["scale"]  # 1.0（実寸）
ents = []
for e in doc["entities"]:
    t = e["type"]
    if t == "LINE":
        ents.append({"t": "L", "p": [e["start_x"] * S, e["start_y"] * S, e["end_x"] * S, e["end_y"] * S]})
    elif t == "ARC":
        ents.append({"t": "A", "c": [e["center_x"] * S, e["center_y"] * S], "r": e["radius"] * S,
                     "a0": e["start_angle"], "sw": e["arc_angle"]})
    elif t == "CIRCLE":
        ents.append({"t": "C", "c": [e["center_x"] * S, e["center_y"] * S], "r": e["radius"] * S})
out = {"名前": "TOTO GG ウォシュレット一体形便器（ZJ1の代用）", "出どころ": "トイレ_jww G312898-B1（意匠図 平面図）",
       "長さ_mm": 736, "幅_mm": 386, "原点": "便器の中心", "前": "+x", "entities": ents}
p = ROOT / "03_建具部品/TOTO/便器_GG/便器_GG.json"
p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(len(ents), "図形 →", p)
