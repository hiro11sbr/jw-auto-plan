"""検査（check_conditions.py）で分かる、建具まわりの線の不具合を直して書き直す。

usage: fix_fittings.py <図.jww> <_meta.json> [書き出し.jww]   （書き出しを省くと上書き）

直すもの：
  C 紫線（色5）が窓・建具の枠の中に入っている → 枠の外までで切る
  D 壁の線（レイヤー3）が建具の中を通っている → 建具の所で切る
  M 窓の戻りの紫線が枠の線に届いていない → 部品の線までのばす
  A 部品の線が一部欠けている（9割以上ないときは「動かされた」とみて触らない） → 部品集の線を足す
make_plan.py が図を書いたあとに必ず通す。正解図（人が直した図）にも使える。
"""
import json
import math
import sys
from pathlib import Path

from shapely.geometry import LineString

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ezjww  # noqa: E402
import jww_env  # noqa: E402
import check_conditions as cc  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def _pieces(seg, cut):
    rest = seg.difference(cut)
    geoms = getattr(rest, "geoms", [rest])
    return [list(g.coords) for g in geoms if not g.is_empty and g.length > 0.5]


def fix(jww, meta_path, cfg=None, lib=None, out=None):
    cfg = cfg or cc.spec.current()
    lib = lib or cc.load_lib(cfg)
    geom = {}
    cc.check(jww, meta_path, cfg, lib, quiet=True, geom=geom)
    rects, wl, fl, S, (cx, cy) = geom["rects"], geom["wl"], geom["fl"], geom["S"], geom["center"]
    # 線の種類ごとの色・レイヤーの指定（line_styles）を反映した実効値：建具のレイヤー、紫線の色
    fls, purple_c = geom.get("fls", {fl}), geom.get("purple", {5})
    mm = lambda x, y: (x * S + cx, y * S + cy)  # noqa: E731
    paper = lambda p: ((p[0] - cx) / S, (p[1] - cy) / S)  # noqa: E731
    doc = ezjww.read_document(str(jww))
    ents, n_cut, n_ext = [], 0, 0
    shorts = geom.get("short_returns", [])
    close = lambda a, b: abs(a[0] - b[0]) < 0.01 and abs(a[1] - b[1]) < 0.01  # noqa: E731
    for e in doc["entities"]:
        if e["type"] == "LINE":
            p, q = mm(e["start_x"], e["start_y"]), mm(e["end_x"], e["end_y"])
            b = e["base"]
            # M 窓の戻りの紫線が枠に届いていない → 部品の線までのばす（user 11/14）
            for sp, sq, tip, tgt in shorts:
                if b["pen_color"] in purple_c and ((close(p, sp) and close(q, sq)) or (close(p, sq) and close(q, sp))):
                    p, q = (tgt, q) if close(p, tip) else (p, tgt)
                    e = dict(e, start_x=paper(p)[0], start_y=paper(p)[1], end_x=paper(q)[0], end_y=paper(q)[1])
                    n_ext += 1
                    break
            if p != q:
                seg = LineString([p, q])
                for pl, _word, _W, purple, wallbox, _whole, _c in rects:
                    cut = None
                    if b["layer"] not in fls and b["pen_color"] in purple_c and not purple.is_empty \
                            and seg.intersects(purple) and seg.intersection(purple).length > 1.0:
                        cut = purple.buffer(1.5, join_style=2)
                    elif b["layer"] == wl and b["pen_color"] not in purple_c and seg.intersects(wallbox) \
                            and seg.intersection(wallbox).length > 3.0 \
                            and (abs(q[0] - p[0]) > abs(q[1] - p[1])) == pl["horizontal"]:
                        cut = wallbox
                    if cut is not None:
                        seg = LineString(seg.coords)  # 念のため
                        parts = _pieces(seg, cut)
                        n_cut += 1
                        if len(parts) != 1:
                            for c in parts:
                                ents.append(dict(e, start_x=paper(c[0])[0], start_y=paper(c[0])[1],
                                                 end_x=paper(c[-1])[0], end_y=paper(c[-1])[1]))
                            seg = None
                            break
                        seg = LineString(parts[0])
                if seg is None:
                    continue
                (x0, y0), (x1, y1) = seg.coords[0], seg.coords[-1]
                e = dict(e, start_x=paper((x0, y0))[0], start_y=paper((x0, y0))[1], end_x=paper((x1, y1))[0], end_y=paper((x1, y1))[1])
        ents.append(e)
    import line_styles as ls_
    for p, q, bb, word in geom.get("missing", []):
        a, b2 = paper(p), paper(q)
        ov = ls_.override(cfg, ls_.fitting_kind(word))  # 建具の色・線種・レイヤーの指定（なければ部品のまま）
        base = dict(bb, layer=ov.get("layer", fl), pen_color=ov.get("color", bb["pen_color"]), pen_style=ov.get("style", bb["pen_style"]))
        ents.append({"type": "LINE", "base": base, "start_x": a[0], "start_y": a[1], "end_x": b2[0], "end_y": b2[1]})
    n_add = len(geom.get("missing", []))
    write(doc, ents, out or jww)
    print(f"建具まわりの線：切った {n_cut}、足した {n_add}、窓までのばした {n_ext} → {out or jww}")
    return n_cut, n_add


def write(doc, ents, out):
    d = ezjww.new(paper_size=3)
    jww_env.apply_env(d)
    g0, g = doc["header"]["layer_groups"][0], d.options["layer_groups"][0]
    g["name"], g["scale"] = g0["name"], g0["scale"]
    for i, lay in enumerate(g0["layers"]):
        g["layers"][i]["name"] = lay["name"]
    msp = d.modelspace()

    def attrs(e):
        bb = e["base"]
        return {"layer": bb["layer"], "pen_color": bb["pen_color"],
                "pen_style": 1 if e["type"] in ("TEXT", "POINT") else bb["pen_style"]}
    other = {}
    for e in ents:
        t = e["type"]
        if t == "LINE":
            msp.add_line((e["start_x"], e["start_y"]), (e["end_x"], e["end_y"]), jwwattribs=attrs(e))
        elif t == "ARC":
            st = round(math.degrees(e["start_angle"]), 4) % 360.0
            st = 0.0 if st >= 359.9999 else st
            msp.add_arc((e["center_x"], e["center_y"]), e["radius"], st, round(math.degrees(e["arc_angle"]), 4), jwwattribs=attrs(e))
        elif t == "CIRCLE":
            msp.add_circle((e["center_x"], e["center_y"]), e["radius"], jwwattribs=attrs(e))
        elif t == "POINT":
            msp.add_point((e["x"], e["y"]), jwwattribs=attrs(e))
        elif t == "TEXT":
            if str(e.get("content", "")).startswith(("Printer_", "View_", "Draw_")):
                continue
            msp.add_text(e["content"], (e["start_x"], e["start_y"]), (e["end_x"], e["end_y"]), size_x=e["size_x"], size_y=e["size_y"],
                         spacing=e["spacing"], angle=e["angle"], font_name=e["font_name"], text_type=e["text_type"], jwwattribs=attrs(e))
        else:
            other[t] = other.get(t, 0) + 1
    if other:
        print(f"  注意：書き写せない図形 {other}")
    d.saveas(str(out))


if __name__ == "__main__":
    fix(sys.argv[1], sys.argv[2], out=sys.argv[3] if len(sys.argv) > 3 else None)
