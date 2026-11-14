"""壁の厚みを数字で決める仕様（配布用）と、メーカーなしの汎用建具の部品集を作る。

使い方（関数）:
    import custom_spec
    path = custom_spec.make_spec("うちの標準", ext=16, air=18, ins=60, osb=9.5, col=105, pb=12.5,
                                 parts="generic")            # parts="generic" か、部品集 .jww のパス（リスト可）
    → 05_設定/仕様/うちの標準.json（default.json を引き継ぐ）を書いて、そのパスを返す
      parts="generic" のときは 05_設定/仕様/うちの標準_部品/部品集_汎用.jww も作る（壁の厚みに合わせた窓の位置）
使い方（コマンド）:
    py -3.13 -I scripts/custom_spec.py <保存名> [ext air ins osb col pb] [--parts generic|<部品集.jww>...]
    作図：make_plan.py に、できた json を設定として渡す（run_all.ps1 なら 05_設定/現在の仕様.txt に保存名を書く）

壁の各面（柱芯から、mm）:
    柱の面 col/2、室内のボード面 col/2+pb
    構造用面材の外面 col/2+osb、付加断熱の外面 +ins、通気層の外面 +air、外壁の面 +ext
    （ポーチ側の壁＝付加断熱なし：面材の外面 → 通気層 → 外壁）
この関数で作った仕様（"_入力" がある json）だけを上書きする。ほかの仕様ファイルには触らない。
便器も汎用の形（"toilet_part": "generic"）。メーカー品を使うときは docs/メーカー品を使う手順.md を参照。
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ezjww  # noqa: E402
from place_fittings import std_bay, std_offset  # noqa: E402

ROOT = HERE.parent
SETTINGS = ROOT / "05_設定"
DEFAULTS = {"ext": 16.0, "air": 18.0, "ins": 60.0, "osb": 9.5, "col": 105.0, "pb": 12.5}
MAKER, SERIES = "汎用", "標準"  # 部品の名前：汎用_標準_<種類>_W<幅>（メーカー名なし）

# 汎用の部品の幅（数字だけ。形はここで描く）
WIDTHS = {
    "片開き": (650, 735, 755, 780, 825, 875),
    "両開き": (735, 825, 1190),
    "折戸": (735, 825, 1190, 1320, 1645, 1820),
    "片引戸": (1190, 1445, 1645),
    "引違い戸": (1645, 2430),
    "アウトセット片引戸": (1320, 1580),
    "引違い窓": (1185, 1235, 1540, 1640, 1690, 2550, 2600),
    "縦すべり窓": (300, 405, 500, 640, 730, 780),
    "FIX窓": (300, 405, 500, 640, 730, 780, 1235, 1640, 1690),
    "横すべり窓": (405, 500, 640, 730, 780, 1000, 1235, 1690),
    "縦すべり+FIX窓": (1235, 1640, 1690),
    "玄関片開き": (1126,),
    "玄関親子": (1379,),
}


def faces(ext, air, ins, osb, col, pb):
    """offsets_mm（default.json と同じ形）"""
    h = col / 2
    sh = h + osb
    return {
        "column_face": h,
        "board_face_inside": h + pb,
        "exterior_side": {"sheathing": sh, "added_insulation": sh + ins,
                          "ventilation": sh + ins + air, "exterior_face": sh + ins + air + ext},
        "interior_wall": [h, h + pb],
        "porch_side": {"column_face": h, "sheathing": sh, "ventilation": sh + air, "exterior_face": sh + air + ext},
    }


def _safe(name):
    name = str(name).strip()
    if not name or re.search(r'[\\/:*?"<>|]', name) or name.startswith((".", "_")):
        raise ValueError(f"保存名に使えない文字があります: {name!r}")
    return name


def make_spec(name, ext=16.0, air=18.0, ins=60.0, osb=9.5, col=105.0, pb=12.5, parts="generic", comment=""):
    """05_設定/仕様/<name>.json を書いて、そのパスを返す。parts="generic" なら汎用の部品集も作る。"""
    name = _safe(name)
    vals = {"ext": ext, "air": air, "ins": ins, "osb": osb, "col": col, "pb": pb}
    for k, v in vals.items():
        v = float(v)
        if not (0 <= v <= 400) or (k == "col" and v < 60):
            raise ValueError(f"{k} の値がおかしい: {v}")
        vals[k] = v
    off = faces(**vals)
    spec_dir = SETTINGS / "仕様"
    old = spec_dir / f"{name}.json"
    if old.exists() and "_入力" not in json.loads(old.read_text(encoding="utf-8")):
        raise ValueError(f"同じ名前の仕様（手で作ったもの）があるので上書きしない: {name}")
    if parts == "generic" or parts is None:
        lib_dir = spec_dir / f"{name}_部品"
        build_generic(lib_dir, off, col_mm=vals["col"])
        library = [str((lib_dir / "部品集_汎用.jww").relative_to(ROOT))]
    else:
        library = [str(Path(p)) for p in ([parts] if isinstance(parts, (str, Path)) else parts)]
    data = {
        "_extends": "../default.json",
        "_comment": comment or f"自分で数字を入れた仕様（配布用）: {json.dumps(vals, ensure_ascii=False)}",
        "_入力": vals,
        "placement": {"library": library},
        "column": {"size_mm": vals["col"]},
        "toilet_part": "generic",
        "offsets_mm": off,
    }
    path = spec_dir / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---- 汎用の部品集 -------------------------------------------------------------------------

def _attr(color=1, style=1):
    return {"group": 0, "pen_style": style, "pen_color": color, "pen_width": 0, "layer": 3, "layer_group": 0, "flag": 0}


def _shape(word, W, off, added):
    """局所の mm（x＝左の柱芯から壁に沿って、y＝壁芯から。+y が外／扉の開く側）で線と円弧を返す。"""
    L, A = [], []
    line = lambda a, b: L.append((a, b))  # noqa: E731

    def rect(x0, y0, x1, y1):
        line((x0, y0), (x1, y0)); line((x1, y0), (x1, y1)); line((x1, y1), (x0, y1)); line((x0, y1), (x0, y0))

    x0, x1 = off, off + W
    if word.endswith("窓"):
        a, b = added - 65.0, added + 65.0  # 枠の厚み130の真ん中＝付加断熱の外面
        rect(x0, a, x1, b)
        line((x0 + 25, a), (x0 + 25, b)); line((x1 - 25, a), (x1 - 25, b))
        g0, g1 = x0 + 25, x1 - 25
        if word == "引違い窓":
            m = (x0 + x1) / 2
            line((g0, added - 12), (m + 20, added - 12)); line((m - 20, added + 12), (g1, added + 12))
        elif word == "縦すべり+FIX窓":
            m = x0 + round(W * 0.4)
            line((m, a), (m, b))
            line((g0, added), (m, added)); line((m, added), (g1, added))
            ln = (m - g0) / 2  # 開いた障子（外へ30°）
            line((m, added), (m - ln * math.cos(math.radians(30)), added + ln * math.sin(math.radians(30))))
        else:
            line((g0, added), (g1, added))
            if word == "縦すべり窓":
                ln = (g1 - g0) / 2
                line((g1, added), (g1 - ln * math.cos(math.radians(30)), added + ln * math.sin(math.radians(30))))
            elif word == "横すべり窓":
                line((g0, b), (g1, b + 60))
        return L, A
    if word in ("玄関片開き", "玄関親子"):
        a, b = added - 65.0, added + 65.0
        rect(x0, a, x1, b)
        j0, j1 = x0 + 111, x1 - 111  # 枠の内側
        line((j0, a), (j0, b)); line((j1, a), (j1, b))
        if word == "玄関片開き":
            r = j1 - j0
            line((j0, b), (j0, b + r)); A.append(((j0, b), r, 0.0, 90.0))
        else:
            r1 = round((j1 - j0) * 0.7)
            r2 = (j1 - j0) - r1
            mx = j0 + r1
            line((mx, a), (mx, b))
            line((j0, b), (j0, b + r1)); A.append(((j0, b), r1, 0.0, 90.0))
            line((j1, b), (j1, b + r2)); A.append(((j1, b), r2, 90.0, 90.0))
        return L, A
    h = 77.5  # 建具枠155
    lo = -65.0 if word in ("折戸", "両開き") else -h  # 収納建具は開かない側をボード面にそろえる
    if word == "片開き":
        rect(x0, lo, x0 + 30, h); rect(x1 - 30, lo, x1, h)
        r = W - 60
        hx = x0 + 30
        line((hx, h), (hx, h + r)); line((hx + 35, h), (hx + 35, h + r)); line((hx, h + r), (hx + 35, h + r))
        A.append(((hx, h), r, 0.0, 90.0))
    elif word == "両開き":
        rect(x0, lo, x0 + 30, h); rect(x1 - 30, lo, x1, h)
        r = (W - 60) / 2
        line((x0 + 30, h), (x0 + 30, h + r)); A.append(((x0 + 30, h), r, 0.0, 90.0))
        line((x1 - 30, h), (x1 - 30, h + r)); A.append(((x1 - 30, h), r, 90.0, 90.0))
        line((x0 + 30 + r, lo), (x0 + 30 + r, h))
    elif word == "折戸":
        from_cfg_pairs = 1 if W < 910 else 2
        rect(x0, lo, x0 + 30, h); rect(x1 - 30, lo, x1, h)
        span = (W - 60) / (2 * from_cfg_pairs)  # 1組（2枚）の幅
        starts = [x0 + 30] if from_cfg_pairs == 1 else [x0 + 30, x1 - 30 - span]
        for s_ in starts:
            p = span / 2
            peak = (s_ + p * math.cos(math.radians(60)), h + p * math.sin(math.radians(60)))
            line((s_, h), peak); line(peak, (s_ + 2 * p * math.cos(math.radians(60)), h))
        line((x0 + 30, h), (x1 - 30, h))
    elif word == "片引戸":
        m = x0 + W / 2  # 中方立（ここから右が戸袋）
        rect(x0, -h, x0 + 30, h); rect(m - 15, -h, m + 15, h); rect(x1 - 30, -h, x1, h)
        line((x0 + 30, 20), (m + 40, 20)); line((x0 + 30, 50), (m + 40, 50))
        line((m + 15, -20), (x1 - 30, -20))  # 戸袋の中の線
    elif word == "アウトセット片引戸":
        # 壁を削らず、壁の表面（-y の面）に付ける引戸。開口は枠の外から外（130〜W-540）
        j0, j1 = 130.0, W - 540.0
        rect(j0, -65, j0 + 24, h); rect(j1 - 24, -65, j1, h)
        rect(j0, -125, W + 152, -78)  # 上レール（戸が走る範囲）
        line((j0 + 24, -101), (j1 + 30, -101))  # 戸
    elif word == "引違い戸":
        rect(x0, -h, x0 + 30, h); rect(x1 - 30, -h, x1, h)
        m = (x0 + x1) / 2
        line((x0 + 30, -15), (m + 20, -15)); line((x0 + 30, -45), (m + 20, -45))
        line((m - 20, 15), (x1 - 30, 15)); line((m - 20, 45), (x1 - 30, 45))
    return L, A


def build_generic(out_dir, off, col_mm=105.0):
    """汎用の部品集（部品集_汎用.jww と 建具部品集_一覧.csv）を out_dir に作る。"""
    cfg = json.load(open(SETTINGS / "default.json", encoding="utf-8"))
    S = cfg["layers"]["scale"]
    added = off["exterior_side"]["added_insulation"]
    d = ezjww.new(paper_size=3, memo="建具部品集 汎用（メーカーなし）\r\n")
    g = d.options["layer_groups"][cfg["layers"]["group"]]
    g["name"], g["scale"] = cfg["layers"]["group_name"], S
    msp = d.modelspace()
    index, y = [], 120.0
    for word, widths in WIDTHS.items():
        x = -180.0
        for W in widths:
            o = std_offset(word, W)
            lines, arcs = _shape(word, W, o, added)
            ents = [{"type": "LINE", "base": _attr(), "start_x": a[0] / S, "start_y": a[1] / S,
                     "end_x": b[0] / S, "end_y": b[1] / S} for a, b in lines]
            ents += [{"type": "ARC", "base": _attr(), "center_x": c[0] / S, "center_y": c[1] / S, "radius": r / S,
                      "start_angle": math.radians(a0), "arc_angle": math.radians(da), "tilt_angle": 0.0,
                      "flatness": 1.0, "is_full_circle": False} for c, r, a0, da in arcs]
            name = f"{MAKER}_{SERIES}_{word}_W{W}"
            d.add_block(name, ents)
            msp.add_block_ref(name, (x, y), jwwattribs={"layer": 3})
            index.append({"ファイル": "部品集_汎用.jww", "ブロック名": name, "メーカー": MAKER, "シリーズ": SERIES,
                          "種類": word, "幅W": W, "柱芯から枠外まで": round(o, 1), "柱芯々": round(std_bay(word, W)),
                          "備考": "汎用の形（メーカーなし）", "元データ": "", "元データ取得日": "", "元データ公開日": "", "商品ページ": ""})
            x += std_bay(word, W) / S + 20
        y -= 30.0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ezjww の保存は日本語の深いフォルダで失敗することがあるので、%TEMP% に書いてから写す
    import os
    import shutil
    tmp = Path(os.environ.get("TEMP", str(out_dir))) / "jw_generic_parts.jww"
    d.saveas(str(tmp))
    shutil.copyfile(tmp, out_dir / "部品集_汎用.jww")
    with open(out_dir / "建具部品集_一覧.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    return out_dir / "部品集_汎用.jww"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    parts = "generic"
    if "--parts" in args:
        i = args.index("--parts")
        rest = args[i + 1:]
        args = args[:i]
        parts = "generic" if rest == ["generic"] else rest
    nm, nums = args[0], [float(v) for v in args[1:]]
    keys = ["ext", "air", "ins", "osb", "col", "pb"]
    kw = dict(DEFAULTS, **dict(zip(keys, nums)))
    p = make_spec(nm, parts=parts, **kw)
    print(p)
