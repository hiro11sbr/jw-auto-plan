"""YKK AP APW 430（樹脂トリプル窓・アングル無）の意匠図･建具表（.jww）→ 窓の部品集用の「できあがり部品」。

usage: build_ykk_windows.py
  03_建具部品/YKK_APW430/元データ/APW430_*.jww の横断面図（平面の部品）を、サイズごとに取り出し、
  部品集の形（用紙1/50mm。基点＝左の柱の中心・壁芯上、+x＝壁の長手、+y＝外部）にして ready/ に置き、
  parts_spec.json を作る。そのあと
  build_parts_library.py 03_建具部品/YKK_APW430 05_設定/default.json 03_建具部品/YKK_APW430/部品集
  で 部品集_窓.jww と一覧CSV ができる。

- 窓の幅 W＝枠外寸法（YKKの表の W）。LIXILと同じ呼びの幅でも、036 だけは YKK は405（LIXILは400）。
- 横断面図の線は枠の左右に28ずつ余分に出る（外壁との納まりの線）。LIXILの部品と同じ作りで、
  幅の判定には枠外寸法を使う（クラスタの幅−56）。
- 枠の厚み（130）の真ん中を、付加断熱の外面（壁芯から）に置く（YKKの図の寸法 65＋50＋15：枠の外側65が付加断熱の外面から外、
  内側は付加断熱50＋面材15。LIXILの窓と同じ考え方）。
- 縦すべり窓は、YKKの表に平面がW640の1つだけなので、それを左右に伸縮して各幅にした。
- 利用条件：YKK AP のCADデータサイト（cad.ykkap.co.jp）の規約に従う。公開版には入れない。
"""
import collections
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ezjww  # noqa: E402
import place_fittings as pf  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = HERE.parent
BASE = ROOT / "03_建具部品" / "YKK_APW430"
import spec as _spec  # noqa: E402
# 窓の枠の厚みは130。YKKの図（65＋50＋15）で、枠の真ん中＝付加断熱の外面に当たる（枠の外側65が付加断熱の外面から外、
# 内側が付加断熱50＋面材15）。付加断熱の外面（壁芯から）に、枠の厚みの真ん中を合わせる
ADDED_OUT = _spec.current()["offsets_mm"]["exterior_side"]["added_insulation"]
FRAME_IN = ADDED_OUT - 65.0  # 枠の室内側の面（壁芯から）
STUB = 28.0  # 横断面図の線が枠の左右に出る長さ
S = 50.0


def comps(lines, tol=0.4):
    n = len(lines)
    par = list(range(n))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    bb = [(min(e["start_x"], e["end_x"]), max(e["start_x"], e["end_x"]), min(e["start_y"], e["end_y"]), max(e["start_y"], e["end_y"])) for e in lines]
    cell = collections.defaultdict(list)
    for i, b in enumerate(bb):
        for gx in range(int((b[0] - tol) // 5), int((b[1] + tol) // 5) + 1):
            for gy in range(int((b[2] - tol) // 5), int((b[3] + tol) // 5) + 1):
                cell[(gx, gy)].append(i)
    for ids in cell.values():
        for a in ids:
            for b in ids:
                if a < b and bb[a][0] <= bb[b][1] + tol and bb[b][0] <= bb[a][1] + tol and bb[a][2] <= bb[b][3] + tol and bb[b][2] <= bb[a][3] + tol:
                    par[f(a)] = f(b)
    g = collections.defaultdict(list)
    for i in range(n):
        g[f(i)].append(i)
    out = []
    for ids in g.values():
        out.append({"ids": ids, "x0": min(bb[i][0] for i in ids), "x1": max(bb[i][1] for i in ids),
                    "y0": min(bb[i][2] for i in ids), "y1": max(bb[i][3] for i in ids)})
    return out


# 種類 → (ファイル, 線の層, 部品の選び方, 備考)
SHEETS = [
    ("FIX窓", "APW430_1.jww", {"YKK_B1"}, lambda c, k: c["y0"] < -170, "FIX窓 アングル無"),
    ("引違い窓", "APW430_6.jww", {"YSS_K"}, lambda c, k: c["y0"] < -100 and (c["y1"] - c["y0"]) * k < 200, "引違い窓 2枚建 トリプルガラス アングル無"),
    ("縦すべり+FIX窓", "APW430_20.jww", {"YKK_B1"}, lambda c, k: c["y0"] < -130, "たてすべり出し窓+FIX2連窓 アングル無"),
    ("横すべり窓", "APW430_4.jww", {"YKK_B1"}, lambda c, k: abs(c["y0"] + 164.6) < 1.0, "すべり出し窓 アングル無"),
]
# 縦すべり窓（たてすべり出し窓）は、W640の平面を伸縮して作る
TATE = ("縦すべり窓", "APW430_3.jww", {"YKK_B1"}, "たてすべり出し窓 アングル無（W640から伸縮）")
TATE_WIDTHS = [300, 405, 500, 640, 730, 780]


def known_widths(doc):
    ws = set()
    for e in doc["entities"]:
        if e["type"] == "TEXT":
            m = re.match(r"^W=([\d,]+)", e["content"])
            if m:
                ws.add(int(m.group(1).replace(",", "")))
    return sorted(ws)


def to_part(doc, lines, arcs, c, k, W, y_in):
    """クラスタ c の線を、部品集の形（用紙mm、左の柱の中心が原点）にする"""
    xc = (c["x0"] + c["x1"]) / 2 * k
    fl = xc - W / 2  # 枠の左端（実mm）
    off = pf.std_offset(KIND_NOW, W)
    out = []
    for i in c["ids"]:
        e = lines[i]
        out.append(("L", [((e["start_x"] * k - fl) + off) / S, (FRAME_IN + (e["start_y"] * k - y_in)) / S,
                          ((e["end_x"] * k - fl) + off) / S, (FRAME_IN + (e["end_y"] * k - y_in)) / S]))
    # 障子の開きの円弧（縦すべり+FIX窓）：中心がこの窓の範囲にある円弧も写す（線だけ写して円弧が抜けていた、user 11/16）
    for a in arcs or []:
        if c["x0"] - 1 <= a["center_x"] <= c["x1"] + 1 and c["y0"] - 1 <= a["center_y"] <= c["y1"] + 1:
            out.append(("A", [((a["center_x"] * k - fl) + off) / S, (FRAME_IN + (a["center_y"] * k - y_in)) / S,
                              a["radius"] * k / S, math.degrees(a["start_angle"]) % 360.0, math.degrees(a["arc_angle"])]))
    return out, off


def save_ready(path, items, color=1):
    d = ezjww.new(paper_size=3)
    msp = d.modelspace()
    for t, p in items:
        at = {"layer": 5, "pen_color": color, "pen_style": 1}
        if t == "A":
            msp.add_arc((p[0], p[1]), p[2], start_angle=p[3], sweep_angle=p[4], jwwattribs=at)
        else:
            msp.add_line((p[0], p[1]), (p[2], p[3]), jwwattribs=at)
    d.saveas(str(path))


KIND_NOW = ""


def main():
    global KIND_NOW
    (BASE / "ready").mkdir(exist_ok=True)
    spec = {"_comment": "YKK AP APW 430（樹脂トリプル窓・アングル無）の窓。build_ykk_windows.py が作る。", "sheet_scale": 50, "parts": []}
    src_rows = ["ファイル,取得日,公開日,商品ページ"]
    made = {}

    def add(kind, W, items, off, note, src):
        out = BASE / "ready" / f"{kind}_W{W}.jww"
        save_ready(out, items)
        spec["parts"].append({"maker": "YKK AP", "series": "APW430", "kind": kind, "op": "window", "width": W,
                              "note": note, "file": f"ready/{out.name}", "ready": True, "offset_mm": off})
        src_rows.append(f"ready/{out.name},2026-11-13,,https://cad.ykkap.co.jp/housing/series/窓・シャッター/APW/APW 430/アングル無/プロジェクト窓・他")
        made.setdefault(kind, []).append(W)

    for kind, fname, layers, pick, note in SHEETS:
        KIND_NOW = kind
        doc = ezjww.read_document(str(BASE / "元データ" / fname))
        k = doc["header"]["layer_groups"][0]["scale"]
        names = [l["name"] for l in doc["header"]["layer_groups"][0]["layers"]]
        lines = [e for e in doc["entities"] if e["type"] == "LINE" and names[e["base"]["layer"]] in layers]
        arcs = [e for e in doc["entities"] if e["type"] == "ARC" and names[e["base"]["layer"]] in layers]
        known = known_widths(doc)
        for c in comps(lines):
            if not (c["x1"] - c["x0"] > 3 and pick(c, k)):
                continue
            Wc = (c["x1"] - c["x0"]) * k - 2 * STUB
            W = min(known, key=lambda w: abs(w - Wc)) if known else round(Wc / 5) * 5
            if abs(W - Wc) > 15:
                print(f"  幅が合わない: {kind} {Wc:.0f} → {W}（飛ばす）")
                continue
            y_in = min(lines[i]["start_y"] for i in c["ids"]) * k
            y_in = min(y_in, min(lines[i]["end_y"] for i in c["ids"]) * k)
            items, off = to_part(doc, lines, arcs, c, k, W, y_in)
            add(kind, W, items, off, f"APW430 {note}", fname)
    # 縦すべり窓
    kind, fname, layers, note = TATE
    KIND_NOW = kind
    doc = ezjww.read_document(str(BASE / "元データ" / fname))
    k = doc["header"]["layer_groups"][0]["scale"]
    names = [l["name"] for l in doc["header"]["layer_groups"][0]["layers"]]
    lines = [e for e in doc["entities"] if e["type"] == "LINE" and names[e["base"]["layer"]] in layers]
    base = next(c for c in comps(lines) if abs(c["x0"] - 76.3) < 0.5 and abs(c["y0"] + 126.5) < 0.5)
    y_in = min(min(lines[i]["start_y"], lines[i]["end_y"]) for i in base["ids"]) * k
    xc = (base["x0"] + base["x1"]) / 2 * k
    W0 = 640
    for W in TATE_WIDTHS:
        off = pf.std_offset(kind, W)
        items = []
        for i in base["ids"]:
            e = lines[i]

            def X(x):
                xr = x * k - xc  # 中心から
                xr = xr + (W - W0) / 2 if xr > 0 else xr - (W - W0) / 2  # 中心から左右へ伸縮
                return (xr + W / 2 + off) / S
            items.append(("L", [X(e["start_x"]), (FRAME_IN + (e["start_y"] * k - y_in)) / S,
                                X(e["end_x"]), (FRAME_IN + (e["end_y"] * k - y_in)) / S]))
        add(kind, W, items, off, f"APW430 {note}", fname)
    (BASE / "parts_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    (BASE / "元データ一覧.csv").write_text("\n".join(src_rows) + "\n", encoding="utf-8-sig")
    for kd, ws in made.items():
        print(kd, sorted(ws))


if __name__ == "__main__":
    main()
