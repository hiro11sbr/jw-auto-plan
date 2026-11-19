"""線の種類ごとの色・線種・レイヤーの指定（仕様の "line_styles"。画面の「詳細設定」で選ぶ）。

仕様の json：{"line_styles": {"center": {"layer": 7, "color": 4}, "outermost": {"color": 6, "style": 4}, ...}}
書いてない種類・項目は、いまの描き方のまま。色は 1〜9、線種は 1〜9、レイヤーは 0〜15。

種類（キー）：
  wall              壁線（柱面・面材・付加断熱・通気層・ポーチ側、開口の戻り線、腰壁・下がり壁 など）
  outermost         最外周（外壁の外面）
  innermost         最内周（石膏ボードの面）
  column            柱（4辺と対角線）
  stud              間柱
  center            通り芯（手入力の通り芯も）
  furniture         住設と家具（便器・UB・キッチン・洗面台、家具の記号の枠と対角線）
  hatch             ハッチング（家具の記号・玄関収納の線）
  visible           その他の見え線（框）
  interior_fitting  内装建具（片開き・片引戸・引違い戸・両開き・折戸 など）
  exterior_fitting  外部建具（窓）
  entrance          玄関ドア
建具3種は、色・線種を書かなければ部品の線のまま。書けば全線をその色・線種にする。
"""

KINDS = ("wall", "outermost", "innermost", "column", "stud", "center", "furniture", "hatch",
         "visible", "interior_fitting", "exterior_fitting", "entrance")
WINDOW_WORDS = {"引違い窓", "縦すべり窓", "FIX窓", "横すべり窓", "縦すべり+FIX窓", "横すべり+FIX窓"}
ENTRANCE_WORDS = {"玄関片開き", "玄関親子"}


def override(cfg, kind):
    """その種類の指定（{"layer", "color", "style"} のうち書いてあるものだけ）。範囲外は無視する"""
    ov = (cfg.get("line_styles") or {}).get(kind) or {}
    out = {}
    for k, lo, hi in (("layer", 0, 15), ("color", 1, 9), ("style", 1, 9)):
        v = ov.get(k)
        if isinstance(v, int) and lo <= v <= hi:
            out[k] = v
    return out


def attr(cfg, kind, layer, color, style):
    """(layer, color, style) に、その種類の指定を重ねる"""
    ov = override(cfg, kind)
    return ov.get("layer", layer), ov.get("color", color), ov.get("style", style)


def fitting_kind(word):
    if word in ENTRANCE_WORDS:
        return "entrance"
    if word in WINDOW_WORDS or str(word).endswith("窓"):
        return "exterior_fitting"
    return "interior_fitting"


def check_values(cfg):
    """検査（check_conditions.py）と線の直し（fix_fittings.py）が線を見分けるための実効値。
    指定がなければ、いまと同じ（壁レイヤー・建具レイヤー・通り芯レイヤー・紫線＝色5）。"""
    L = cfg["layers"]
    wl, fl0, cl = L["wall"]["layer"], L.get("fitting", L["wall"])["layer"], L["center"]["layer"]
    finish = L["wall"].get("finish_color", 5)
    fit_layers = {override(cfg, k).get("layer", fl0) for k in ("interior_fitting", "exterior_fitting", "entrance")}
    purple = {override(cfg, "innermost").get("color", finish), override(cfg, "outermost").get("color", finish)}
    if "color" not in override(cfg, "wall"):
        purple.add(finish)  # 窓・開口の戻り線（壁線の仲間、色は紫）
    return {
        "wall_layer": override(cfg, "wall").get("layer", wl),
        "column_layer": override(cfg, "column").get("layer", wl),
        "center_layer": override(cfg, "center").get("layer", cl),
        "fit_layers": fit_layers,
        "purple": purple,
    }
