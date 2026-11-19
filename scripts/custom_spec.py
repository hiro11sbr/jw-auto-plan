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
水回り（キッチン・洗面台・浴室UB・便器）：wet={"kitchen": {"maker","product","size"}, ...} を渡すと、選んだ部品を
kitchen_part・vanity_part・ub_part・toilet_part に入れる（size は "auto"＝図の間口にいちばん近いもの）。
一覧は wet_catalog()（コマンド：--catalog）。部品集のブロック名から作る。
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
    "2枚連動片引戸": (2430,),  # 開口2000以上の引き戸（2枚が1つの戸袋に引き込む。user 11/19 で汎用にも足す）
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


# ---- 水回り（キッチン・洗面台・浴室UB・便器）の部品の一覧と、仕様への書き込み -------------------------
# 一覧は、部品集（03_建具部品/*/部品集/*.jww）の中のブロック名から作る（手で書いた一覧にしない）。
# ブロック名の決まり：<メーカー>_<種類>_<商品>[_W<幅>][_付記]（UB は <メーカー>_UB_<4桁の大きさ>_<ドアの位置>_…）
WET_KINDS = {"キッチン": "kitchen", "洗面": "vanity", "UB": "ub"}
_WET_LABEL = {"kitchen": "キッチン", "vanity": "洗面台", "ub": "浴室（UB）", "toilet": "便器"}


def wet_entries():
    """部品集から、水回りの部品を [{kind, maker, product, size, block, lib, (side)}] で返す。便器は便器の json から"""
    out = []
    for p in sorted((ROOT / "03_建具部品").glob("*/部品集/*.jww")):
        try:
            names = [b["name"] for b in ezjww.read_document(str(p))["block_defs"]]
        except Exception:  # noqa: BLE001  読めない部品集は飛ばす
            continue
        lib = str(p.relative_to(ROOT))
        for n in names:
            t = n.split("_")
            if len(t) < 3 or t[1] not in WET_KINDS:
                continue
            kind, maker, rest = WET_KINDS[t[1]], t[0], t[2:]
            w = next((x[1:] for x in rest if re.fullmatch(r"W\d+", x)), None)
            other = [x for x in rest if not re.fullmatch(r"W\d+", x)]
            e = {"kind": kind, "maker": maker, "block": n, "lib": lib}
            if kind == "ub":
                if not other or not re.fullmatch(r"\d{4}", other[0]):
                    continue
                side = "beside" if any(x.startswith("ドア下") for x in other) else \
                    "facing" if any(x.startswith("ドア右") for x in other) else None
                if side is None:
                    continue
                e.update(product="ユニットバス", size=other[0], side=side)
            else:
                if w is None:
                    continue
                pen = kind == "kitchen" and "ペニンシュラ" in other
                e.update(product="ペニンシュラ" if pen else " ".join(other), size=w, pen=pen)
            out.append(e)
    for p in sorted((ROOT / "03_建具部品").glob("*/便器_*/*.json")):
        try:
            part = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"kind": "toilet", "maker": p.parent.parent.name, "product": str(part.get("名前") or p.stem),
                    "size": None, "block": p.stem, "lib": p.relative_to(ROOT).as_posix()})
    return out


def wet_catalog():
    """画面に出す一覧：{種類: {メーカー: {商品: [大きさ, …]}}}（持っていない部品は出ない）"""
    cat = {k: {} for k in _WET_LABEL}
    for e in wet_entries():
        sizes = cat[e["kind"]].setdefault(e["maker"], {}).setdefault(e["product"], [])
        if e["size"] and e["size"] not in sizes:
            sizes.append(e["size"])
    for k in cat.values():
        for prods in k.values():
            for s in prods.values():
                s.sort(key=int)
    return cat


def wet_spec(wet):
    """画面で選んだ水回り {種類: {maker, product, size}} を、仕様の json に入れる項目にする。
    maker が空（指定しない）の種類は入れない（そのときは汎用の形）。size は "auto"（自動）か 幅・大きさの数字"""
    if not wet:
        return {}
    ents = wet_entries()
    add = {}
    for kind, sel in wet.items():
        if kind not in _WET_LABEL or not sel or not sel.get("maker"):
            continue
        size = str(sel.get("size") or "auto")
        hit = [e for e in ents if e["kind"] == kind and e["maker"] == sel["maker"] and e["product"] == sel.get("product")]
        if not hit:
            raise ValueError(f"{_WET_LABEL[kind]}：選んだ部品が部品集にありません")
        pick = hit if size == "auto" else [e for e in hit if e["size"] == size]
        if not pick:
            raise ValueError(f"{_WET_LABEL[kind]}：選んだ大きさの部品がありません（{size}）")
        lib = pick[0]["lib"].replace("/", "\\")
        if kind == "toilet":
            add["toilet_part"] = pick[0]["lib"]
        elif kind == "vanity":
            add["vanity_part"] = {"library": lib, "blocks": {e["size"]: e["block"] for e in pick}, **({"fixed": True} if size != "auto" else {})}
        elif kind == "kitchen":
            kp = {"library": lib, "stove": [-1, 0], "back": [0, 1]}
            if pick[0]["pen"]:
                kp["peninsula"] = {"blocks": {e["size"]: e["block"] for e in pick}, "stove": [-1, 0], "back": [0, 1]}
            else:
                kp["block"] = pick[0]["block"]
            add["kitchen_part"] = kp
        else:  # ub：浴槽・ドアの向きは部品集の決まり（beside＝ドアが浴槽の横の壁、facing＝浴槽の向かいの壁）
            by = {}
            for e in pick:
                by.setdefault(e["size"], {})[e["side"]] = e["block"]
            sizes = sorted(by, key=int)
            base = by[size] if size != "auto" else by[sizes[0]]
            canon = {"beside": {"tub": [-1, 0], "door": [0, -1], "door_end": [1, 0]},
                     "facing": {"tub": [-1, 0], "door": [1, 0], "door_end": [0, -1]}}
            up = {"library": lib}
            for side in ("beside", "facing"):
                if side in base:
                    up[side] = {"block": base[side], **canon[side]}
            up["sizes"] = {s: {side: {"block": b} for side, b in by[s].items()} for s in sizes if size != "auto" or s != sizes[0]}
            add["ub_part"] = up
    return add


# ---- 詳細設定：線の色・線種・レイヤ（種類ごと） -----------------------------------------------------
# 仕様の json の "line_styles" に {種類: {"layer": 0〜15, "color": 1〜9, "style": 1〜9}} で入る。
# 書いてない種類・項目は、いまの描き方のまま（既定）。作図側（make_plan.py）が cfg["line_styles"] を読んで使う。
LINE_ITEMS = [
    # (キー, 画面の名前, 既定の取り出し方)
    ("wall", "壁線（柱面・面材・付加断熱・通気層ほか。窓の戻り線・額縁・腰壁・下がり壁も含む）"),
    ("outermost", "最外周（外壁の外面）"),
    ("innermost", "最内周（石膏ボードの面）"),
    ("column", "柱"),
    ("stud", "間柱"),
    ("center", "通り芯"),
    ("furniture", "住設と家具"),
    ("hatch", "ハッチング"),
    ("visible", "その他見え線（階段・框）"),
    ("interior_fitting", "内装建具"),
    ("exterior_fitting", "外部建具（窓）"),
    ("entrance", "玄関ドア"),
]
LINE_LIMITS = {"layer": (0, 15), "color": (1, 9), "style": (1, 9)}
# 初期設定（user 11/20）：レイヤグループは全部 □0 に統一し、レイヤ（○）は種類ごとにすべて分ける。
# ○の番号：0〜F。A＝文字、F＝図面枠は、プログラムが使うので避けた
DEFAULT_LAYERS = {
    "wall": 3, "outermost": 7, "innermost": 8, "column": 9, "stud": 12, "center": 4,
    "furniture": 6, "hatch": 11, "visible": 13, "interior_fitting": 5, "exterior_fitting": 14, "entrance": 2,
}


def line_defaults(cfg):
    """種類ごとの、既定の {layer, color, style}（レイヤは DEFAULT_LAYERS＝種類ごとにすべて別。建具の色・線種は部品のままなので None）"""
    L = cfg["layers"]
    w, c = L["wall"], L["center"]
    fd = cfg.get("furniture_draw", {})
    pc = cfg.get("placement", {})
    fit = L.get("fitting", {}).get("layer", w["layer"])
    out = {
        "wall": {"layer": w["layer"], "color": w["structure_color"], "style": w["style"]},
        "outermost": {"layer": w["layer"], "color": w["finish_color"], "style": w["style"]},
        "innermost": {"layer": w["layer"], "color": w["finish_color"], "style": w["style"]},
        "column": {"layer": w["layer"], "color": w["structure_color"], "style": w["style"]},
        "stud": {"layer": w["layer"], "color": w["structure_color"], "style": w["style"]},
        "center": {"layer": c["layer"], "color": c["color"], "style": c["style"]},
        "furniture": {"layer": fd.get("layer"), "color": fd.get("color"), "style": fd.get("style")},
        "hatch": {"layer": fd.get("built_in_layer", fd.get("layer")), "color": fd.get("color"), "style": fd.get("style")},
        "visible": {"layer": pc.get("kamachi_layer"), "color": pc.get("kamachi_color"), "style": pc.get("kamachi_style")},
        "interior_fitting": {"layer": fit, "color": None, "style": None},
        "exterior_fitting": {"layer": fit, "color": None, "style": None},
        "entrance": {"layer": fit, "color": None, "style": None},
    }
    for k, n in DEFAULT_LAYERS.items():   # レイヤは、初期設定で種類ごとにすべて分ける
        out[k]["layer"] = n
    return out


def line_env():
    """画面の詳細設定に出す一覧：色（JWの色1〜9の実際の色）・線種（1〜9）・レイヤ名（0〜F）・種類ごとの既定"""
    env = json.loads((SETTINGS / "参考_環境.json").read_text(encoding="utf-8"))
    cfg = json.loads((SETTINGS / "default.json").read_text(encoding="utf-8"))
    pens = env["palette"]["pen_colors"]
    colors = []
    for n in range(1, 10):
        v = pens[n]  # 0x00BBGGRR
        colors.append({"n": n, "hex": "#%02x%02x%02x" % (v & 255, (v >> 8) & 255, (v >> 16) & 255)})
    lt = {d["number"]: d for d in env["line_types"]["standard"]}
    styles = [{"n": 1, "segments": []}]
    for n in range(2, 10):
        if n in lt:
            styles.append({"n": n, "segments": lt[n]["segments_mm"]})
    names = [l.get("name", "") for l in env["layer_groups"][0]["layers"]]
    layers = [{"n": i, "name": names[i] if i < len(names) else ""} for i in range(16)]
    cfg_names = cfg["layers"].get("names", [])
    layers = [{"n": i, "name": cfg_names[i] if i < len(cfg_names) and cfg_names[i] else l["name"]} for i, l in enumerate(layers)]
    return {"items": [{"key": k, "label": n} for k, n in LINE_ITEMS], "defaults": line_defaults(cfg),
            "colors": colors, "styles": styles, "layers": layers}


def lines_spec(lines):
    """画面で選んだ {種類: {layer, color, style}} を、仕様に入れる形にする（空欄＝既定の項目は入れない）"""
    keys = {k for k, _ in LINE_ITEMS}
    out = {k: {"layer": DEFAULT_LAYERS[k]} for k in keys}   # 初期設定：レイヤは種類ごとにすべて別（画面で変えた分で上書き）
    for key, v in (lines or {}).items():
        if key not in keys:
            raise ValueError(f"詳細設定：知らない種類です（{key}）")
        one = {}
        for f, (lo, hi) in LINE_LIMITS.items():
            x = (v or {}).get(f)
            if x in (None, ""):
                continue
            x = int(x)
            if not lo <= x <= hi:
                raise ValueError(f"詳細設定：{key} の {f} は {lo}〜{hi} です（入力 {x}）")
            one[f] = x
        out[key].update(one)
    return out


def make_spec(name, ext=16.0, air=18.0, ins=60.0, osb=9.5, col=105.0, pb=12.5, parts="generic", comment="", wet=None, lines=None):
    """05_設定/仕様/<name>.json を書いて、そのパスを返す。parts="generic" なら汎用の部品集も作る。"""
    name = _safe(name)
    vals = {"ext": ext, "air": air, "ins": ins, "osb": osb, "col": col, "pb": pb}
    for k, v in vals.items():
        v = float(v)
        if not (0 <= v <= 400):
            raise ValueError(f"{k} の値がおかしい: {v}")
        vals[k] = v
    # 柱は105に固定、室内の石膏ボードは「なし（0）」か「12.5」だけ（厚くすると建具の枠などに障害が出る。user 11/15）
    if abs(vals["col"] - 105.0) > 0.01:
        raise ValueError(f"柱は105だけです（入力 {vals['col']:g}）")
    if not any(abs(vals["pb"] - x) < 0.01 for x in (0.0, 12.5)):
        raise ValueError(f"石膏ボードは なし（0）か 12.5 だけです（入力 {vals['pb']:g}）")
    draw_board = vals["pb"] > 0
    # 石膏ボードなしでも、寸法・建具の枠は12.5の時と同じ。石膏ボードの線を描かないだけ（user 11/15）
    off = faces(**dict(vals, pb=12.5))
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
        "draw_board": draw_board,
        "offsets_mm": off,
    }
    data.update(wet_spec(wet))  # 選んだ水回りの部品（選ばなければ汎用の形のまま）
    if wet:
        data["_入力"] = dict(vals, 水回り=wet)
    ls = lines_spec(lines)  # 詳細設定（線の色・線種・レイヤ）
    if ls:
        data["line_styles"] = ls
        data["_入力"] = dict(data["_入力"], 線=ls)
    path = spec_dir / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---- 汎用の部品集 -------------------------------------------------------------------------

def _attr(color=1, style=1):
    return {"group": 0, "pen_style": style, "pen_color": color, "pen_width": 0, "layer": 3, "layer_group": 0, "flag": 0}


def _shape(word, W, off, added, board=65.0):
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
    h = board + 12.5  # 建具枠：ボード面から12.5ずつ出す（ボード12.5なら155）
    lo = -board if word in ("折戸", "両開き") else -h  # 収納建具は開かない側をボード面にそろえる
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
    elif word == "2枚連動片引戸":
        # 2枚が1つの戸袋に重なって引き込む：開口は全体の 2/3、戸袋は 1/3（中方立から右）
        m = x0 + W * 2 / 3
        rect(x0, -h, x0 + 30, h); rect(m - 15, -h, m + 15, h); rect(x1 - 30, -h, x1, h)
        half_ = (m - 15 - (x0 + 30)) / 2
        line((x0 + 30, 20), (x0 + 30 + half_ + 40, 20)); line((x0 + 30, 50), (x0 + 30 + half_ + 40, 50))  # 1枚目
        line((x0 + 30 + half_, -10), (m + 40, -10)); line((x0 + 30 + half_, -40), (m + 40, -40))  # 2枚目
        line((m + 15, -60), (x1 - 30, -60))  # 戸袋の中の線
    elif word == "アウトセット片引戸":
        # 壁を削らず、壁の表面（-y の面）に付ける引戸。開口は枠の外から外
        j0, j1 = 65.0, 65.0 + W / 2 - 10  # 枠の外＝石膏ボード面（W1580 なら 65〜845＝910の柱間）
        rect(j0, -65, j0 + 24, h); rect(j1 - 24, -65, j1, h)
        rect(j0, -125, W + 87, -78)  # 上レール（戸が走る範囲）
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
            lines, arcs = _shape(word, W, o, added, off["board_face_inside"])
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
    if args == ["--catalog"]:  # 画面の水回りの一覧（JSON）
        print(json.dumps(wet_catalog(), ensure_ascii=False))
        sys.exit(0)
    if args == ["--lineenv"]:  # 画面の詳細設定の一覧（JSON）
        print(json.dumps(line_env(), ensure_ascii=False))
        sys.exit(0)
    lines = None
    if "--lines" in args:
        i = args.index("--lines")
        lines = json.loads(args[i + 1])
        args = args[:i] + args[i + 2:]
    wet = None
    if "--wet" in args:
        i = args.index("--wet")
        wet = json.loads(args[i + 1])
        args = args[:i] + args[i + 2:]
    parts = "generic"
    if "--parts" in args:
        i = args.index("--parts")
        rest = args[i + 1:]
        args = args[:i]
        parts = "generic" if rest == ["generic"] else rest
    nm, nums = args[0], [float(v) for v in args[1:]]
    keys = ["ext", "air", "ins", "osb", "col", "pb"]
    kw = dict(DEFAULTS, **dict(zip(keys, nums)))
    p = make_spec(nm, parts=parts, wet=wet, lines=lines, **kw)
    print(p)
