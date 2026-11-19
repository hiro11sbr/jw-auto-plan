"""YKK AP イノベスト D50（意匠図･建具表の .jww）→ 部品集用の「できあがり部品」と parts_spec.json。

usage: build_ykk_parts.py
  03_建具部品/YKK_イノベスト/元データ/INBD50_*.jww の平面図（柱2本＋建具の断面）から、
  建具の線だけを取り出し、部品集の形（用紙1/50mm。基点＝左の柱の中心・壁芯上、+x＝壁の長手、
  +y＝外部（外開き））にして ready/ に置き、parts_spec.json を作る。そのあと
  build_parts_library.py 03_建具部品/YKK_イノベスト 05_設定/default.json 03_建具部品/YKK_イノベスト/部品集
  で 部品集_玄関ドア.jww と一覧CSV ができる。

- 取り出す線：建具表のレイヤ3・色1（枠・扉）。壁・寸法・ハッチはこちらで描くので入れない。
- 枠の左端を、柱芯から65（部品集の約束）に合わせる。YKKの図では枠の庇が柱の外側にかぶさるので、
  実物の柱芯からの位置とは違う（置き方の約束を既存の玄関ドアと揃えた）。
- 外開きの扉と円弧は図にないので、枠の幅から作る（吊り元は左）。
- 利用条件：YKK AP のCADデータサイト（cad.ykkap.co.jp）の規約に従う。公開版には入れない。
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ezjww  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = HERE.parent
BASE = ROOT / "03_建具部品" / "YKK_イノベスト"
S = 50.0
ATTACH_LOCAL = 65.0  # 元の図のまま写すと、付加断熱の外面に付く線（枠の耳の裏）が壁芯から65にある（正解図でユーザーが57動かして付加断熱60の外面122に合わせた、11/15）
PARTS = {  # 型番 → 種類（部品集の名前）
    "INBD50_103": ("玄関片開き", "スマートコントロールキー 片開き 形材断熱枠"),
    "INBD50_107": ("玄関親子", "スマートコントロールキー 親子 形材断熱枠"),
}


def plan_columns(doc):
    """平面図の柱（斜め線が2本ある105角）の中心を、x の小さい順に返す"""
    L = [e for e in doc["entities"] if e["type"] == "LINE" and e["base"]["layer"] == 10 and e["base"]["pen_color"] == 2]
    boxes = {}
    for e in L:
        dx, dy = e["end_x"] - e["start_x"], e["end_y"] - e["start_y"]
        if abs(abs(dx) - abs(dy)) < 1e-3 and abs(dx) * S > 100:
            cx, cy = (e["start_x"] + e["end_x"]) / 2, (e["start_y"] + e["end_y"]) / 2
            boxes[(round(cx, 2), round(cy, 2))] = (cx, cy)
    cols = sorted(boxes.values())
    return cols


def main():
    spec = {"_comment": "YKK AP イノベスト D50（意匠図･建具表）。build_ykk_parts.py が作る。", "sheet_scale": 50, "parts": []}
    (BASE / "ready").mkdir(exist_ok=True)
    src_rows = ["ファイル,取得日,公開日,商品ページ"]
    for model, (word, note) in PARTS.items():
        doc = ezjww.read_document(str(BASE / "元データ" / f"{model}.jww"))
        cols = plan_columns(doc)
        assert len(cols) >= 2, (model, cols)
        (lx, ly), (rx, _ry) = cols[0], cols[1]
        lines = [e for e in doc["entities"] if e["type"] == "LINE" and e["base"]["layer"] == 3 and e["base"]["pen_color"] == 1
                 and min(e["start_x"], e["end_x"]) > lx - 400 / S and max(e["start_x"], e["end_x"]) < rx + 400 / S
                 and ly - 300 / S < min(e["start_y"], e["end_y"]) and max(e["start_y"], e["end_y"]) < ly + 500 / S]
        xs = [v for e in lines for v in (e["start_x"], e["end_x"])]
        ys = [v for e in lines for v in (e["start_y"], e["end_y"])]
        fx0, fx1 = min(xs), max(xs)
        w = round((fx1 - fx0) * S)  # 枠の外寸（部品の幅 W）
        dx = 65.0 / S - fx0  # 枠の左端 → 柱芯から65
        # 元の図の壁は付加断熱60（外壁面154.5）の仕様で描かれている。今の仕様の付加断熱の外面との差だけ平行移動して、
        # 付加断熱の厚みを変えても玄関ドアが付加断熱の面に追従するようにする（user 11/15）
        import spec as _spec
        ADDED_NOW = _spec.current()["offsets_mm"]["exterior_side"]["added_insulation"]
        dy = -ly + (ADDED_NOW - ATTACH_LOCAL) / S  # 枠の耳の裏の線を、今の付加断熱の外面に合わせる
        d = ezjww.new(paper_size=3)
        msp = d.modelspace()
        at = {"layer": 5, "pen_color": 1, "pen_style": 1}
        for e in lines:
            msp.add_line((e["start_x"] + dx, e["start_y"] + dy), (e["end_x"] + dx, e["end_y"] + dy), jwwattribs=at)
        # 外開きの扉：吊り元は左。扉の幅＝枠の内側の幅の約9割、円弧は吊り元を中心に+y側へ90度
        # 外部側の枠の面＝枠の幅の7割以上ある横線のうち一番外（親子の図には子扉の線が外へ伸びているので、全部の線の最大は使わない）
        long_h = [e["start_y"] for e in lines if abs(e["start_y"] - e["end_y"]) < 1e-6 and abs(e["end_x"] - e["start_x"]) * S > 0.7 * w]
        y_out = max(long_h) + dy  # 外部側の枠の面（用紙mm）
        leaf = (w - 218.0 if word == "玄関片開き" else 908.0) / S  # 扉の幅（YKKの図の扉：枠の内側 111 から 111 まで＝W−218）
        hx = 65.0 / S + 111.0 / S  # 吊り元の位置（扉の端）
        msp.add_arc((hx, y_out), leaf, 0.0, 90.0, jwwattribs={"layer": 5, "pen_color": 6, "pen_style": 1})
        t = 70.0 / S  # 扉の厚み
        for p, q in (((hx, y_out), (hx, y_out + leaf)), ((hx + t, y_out), (hx + t, y_out + leaf)), ((hx, y_out + leaf), (hx + t, y_out + leaf))):
            msp.add_line(p, q, jwwattribs=at)
        out = BASE / "ready" / f"{word}_W{w}.jww"
        d.saveas(str(out))
        spec["parts"].append({"maker": "YKK AP", "series": "イノベスト", "kind": word, "op": "swing", "width": w,
                              "note": f"D50 {note}（柱芯々 {round((rx - lx) * S)}）", "file": f"ready/{out.name}",
                              "ready": True, "offset_mm": 65.0})
        src_rows.append(f"ready/{out.name},2026-11-13,,https://cad.ykkap.co.jp/housing/series/玄関ドア・引戸/玄関ドア/イノベスト InnoBest D70 D50")
        print(f"{word}: 型番 {model} 枠の外寸 W{w}、深さ {round((max(ys) - min(ys)) * S)}、柱芯々 {round((rx - lx) * S)}")
    (BASE / "parts_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    (BASE / "元データ一覧.csv").write_text("\n".join(src_rows) + "\n", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
