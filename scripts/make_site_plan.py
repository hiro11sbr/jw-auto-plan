"""配置図（敷地に建物の位置を示す図）の自動作図。平面図の make_plan.py とは別ファイル（別セッション 11/12）。

usage: make_site_plan.py <ケース名（05_設定\\配置図.json の cases）> [出力.jww]
入力 : 敷地の座標（面積計算簿・求積表の点、work\\配置図\\*.json）と、建物の平面図 .jww（外壁の仕上げ線から外形を作る）
決める : ①建物の外形と敷地を、A3の使える範囲（図枠・表題欄・凡例を除き四方に余白）に入る向きに絞る
         ②入るものの中で 北が北西・北・北東 → 建物の長手が横 の順に選ぶ（入る向きがなければ入りにくい図の向きを優先）
描く   : 敷地境界（道路境界線・隣地境界線と長さ）、建物の外形、北側の離れ、方位記号、図枠
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import ezjww
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent


def load_footprint(jww_path, fp):
    """平面図 .jww の壁芯の線（layer 4・線色3）から外形を作り、外壁面（芯から156）まで広げる（実寸mm、平面図の向きのまま）。"""
    doc = ezjww.read_document(str(jww_path))
    scale = doc["header"]["layer_groups"][fp["layer_group"]]["scale"]
    lines = []
    for e in doc["entities"]:
        b = e["base"]
        if (e["type"] == "LINE" and b["layer_group"] == fp["layer_group"]
                and b["layer"] == fp["layer"] and b["pen_color"] == fp["color"]):
            lines.append(LineString([(e["start_x"] * scale, e["start_y"] * scale),
                                     (e["end_x"] * scale, e["end_y"] * scale)]))
    if not lines:
        raise SystemExit("外壁の仕上げ線が見つからない: " + str(jww_path))
    # 線を太らせてつなぎ、外側の輪郭だけ取る（外壁の線が途切れていても閉じる）→ 同じ量だけ戻す
    r = fp["close_mm"]
    u = unary_union([ln.buffer(r) for ln in lines])
    polys = [u] if u.geom_type == "Polygon" else list(u.geoms)
    big = unary_union([Polygon(p.exterior) for p in polys])
    core = big.buffer(-r, join_style=2)
    if core.geom_type != 'Polygon':
        core = max(core.geoms, key=lambda q: q.area)
    return core.buffer(fp['outer_face_mm'], join_style=2).simplify(20)


def rot(geom, deg):
    return affinity.rotate(geom, deg, origin=(0, 0))


def norm180(a):
    return (a + 180) % 360 - 180


def main():
    case_name = sys.argv[1]
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "work" / "配置図" / f"{case_name}_配置図.jww"
    cfg = json.load(open(ROOT / "05_設定" / "配置図.json", encoding="utf-8"))
    case = cfg["cases"][case_name]
    S = cfg["scale"]
    mg = cfg["margin_paper_mm"]

    # ---- 入力：敷地（北が上、x=東・y=北、実寸mm）と建物の外形（平面図の向き）
    pts = json.load(open(ROOT / case["site_points"], encoding="utf-8"))["points"]
    names = [p[0] for p in pts]
    site_n = Polygon([(p[2] * 1000, p[1] * 1000) for p in pts])  # (東, 北)
    alpha = case["plan_up_azimuth_deg"]  # 平面図の「上」の方位角（北から時計回り）
    bld = load_footprint(ROOT / case["plan_jww"], cfg["footprint"])
    bld = affinity.translate(bld, -bld.centroid.x, -bld.centroid.y)
    bw0, bh0 = bld.bounds[2] - bld.bounds[0], bld.bounds[3] - bld.bounds[1]
    print(f"建物の外形 {bw0:.0f}×{bh0:.0f}mm（外接四角）、敷地の面積 {site_n.area / 1e6:.3f}㎡、平面図の上の方位 {alpha}°")

    # ---- 向きの候補：平面図の向きを基準に 0/90/180/270° 回す。k回すと、北は用紙の上から反時計回りに (alpha + 90k)°
    pa, sa = cfg["plan_area_paper_mm"], cfg["site_area_paper_mm"]
    plan_w = (pa["x"][1] - pa["x"][0] - 2 * mg) * pa["scale"]
    plan_h = (pa["y"][1] - pa["y"][0] - 2 * mg) * pa["scale"]
    site_w = (sa["x"][1] - sa["x"][0] - 2 * mg) * S
    site_h = (sa["y"][1] - sa["y"][0] - 2 * mg) * S
    cands = []
    for k in range(4):
        site_p = rot(affinity.rotate(site_n, alpha, origin=(0, 0)), 90 * k)
        b_p = rot(bld, 90 * k)
        sw, sh = site_p.bounds[2] - site_p.bounds[0], site_p.bounds[3] - site_p.bounds[1]
        bw, bh = b_p.bounds[2] - b_p.bounds[0], b_p.bounds[3] - b_p.bounds[1]
        plan_slack = min(plan_w - bw, plan_h - bh)
        site_slack = min(site_w - sw, site_h - sh)
        north = norm180(alpha + 90 * k)
        cands.append(dict(k=k, north=north, plan_slack=plan_slack, site_slack=site_slack,
                          long_h=bw >= bh, north_ok=abs(north) <= 45.0001))
        print(f"  向き{k}: 北は用紙の上から{north:+.0f}°　建物{'横' if bw >= bh else '縦'}長　"
              f"平面図の余り{plan_slack / 1000:+.1f}m　配置図の余り{site_slack / 1000:+.1f}m")
    both = [c for c in cands if c["plan_slack"] >= 0 and c["site_slack"] >= 0]
    if both:
        pick = max(both, key=lambda c: (c["north_ok"], c["long_h"], -abs(c["north"])))
        why = "どちらも入る向きの中から（北の方角→建物の長手が横）で選んだ"
    else:
        hard = "plan" if max(c["plan_slack"] for c in cands) < max(c["site_slack"] for c in cands) else "site"
        key = (lambda c: c["plan_slack"]) if hard == "plan" else (lambda c: c["site_slack"])
        pick = max(cands, key=lambda c: (key(c), c["north_ok"], c["long_h"]))
        why = f"両方入る向きがない。入りにくい{'平面図' if hard == 'plan' else '配置図'}を優先した（要確認）"
    print(f"→ 向き{pick['k']}：{why}")
    k = pick["k"]
    site_p = rot(affinity.rotate(site_n, alpha, origin=(0, 0)), 90 * k)
    b_p = rot(bld, 90 * k)
    n_ang = pick["north"]
    # 用紙上の北向きベクトル（上から反時計回りに n_ang°）
    nv = (-math.sin(math.radians(n_ang)), math.cos(math.radians(n_ang)))
    # 北に一番近い用紙の軸方向
    axis = max([(1, 0), (-1, 0), (0, 1), (0, -1)], key=lambda a: a[0] * nv[0] + a[1] * nv[1] + 1e-6 * a[1])  # 同点なら上
    cross = (axis[1], -axis[0])

    # ---- 建物の位置：北側の軸方向は敷地境界から gap 離す。直交方向は中央
    gap = cfg["gap_north_mm"]
    cx, cy = site_p.centroid.x, site_p.centroid.y
    b0 = affinity.translate(b_p, cx, cy)

    def inside(b, t):
        moved = affinity.translate(b, axis[0] * t, axis[1] * t)
        return moved.buffer(gap, join_style=2).within(site_p)

    def push(b, d, hi=60000):
        lo_, hi_ = -60000.0, hi
        # d 方向へ動かせる最大量
        a = (d[0], d[1])
        lo = 0.0
        if not b.within(site_p):
            return None
        lo, hi = 0.0, hi_
        for _ in range(40):
            mid = (lo + hi) / 2
            m = affinity.translate(b, a[0] * mid, a[1] * mid)
            if m.buffer(gap, join_style=2).within(site_p):
                lo = mid
            else:
                hi = mid
        return lo

    # まず中央から、直交方向の左右に動かせる範囲を求めて中央に置く
    def room(b, d):
        lo, hi = 0.0, 60000.0
        for _ in range(40):
            mid = (lo + hi) / 2
            m = affinity.translate(b, d[0] * mid, d[1] * mid)
            if m.within(site_p):
                lo = mid
            else:
                hi = mid
        return lo

    # 北側へ押し当てて gap を取る（北側が入らない場合は最大限）
    t = 0.0
    b1 = b0
    if not b1.within(site_p):
        # 重心が敷地外になる場合は、敷地内に入るまで軸方向に探す
        pass
    t = room(b1, axis)
    b1 = affinity.translate(b1, axis[0] * t, axis[1] * t)
    # gap 分だけ戻す（北側の境界から gap）
    # 北側の境界までの距離を測り直し、gap になるよう調整
    b1 = affinity.translate(b1, -axis[0] * gap, -axis[1] * gap)
    l = room(b1, (-cross[0], -cross[1]))
    r = room(b1, cross)
    shift = (r - l) / 2
    b1 = affinity.translate(b1, cross[0] * shift, cross[1] * shift)
    ok = b1.within(site_p)
    print("建物は敷地に" + ("収まっている" if ok else "収まっていない（要確認）"))

    # ---- 離れ（北側・左右・南側を、用紙の軸方向で）
    def dist_to_boundary(b, d):
        """b の外形から d 方向へ敷地境界までの距離（外接四角の中央線で測る）"""
        bx0, by0, bx1, by1 = b.bounds
        c = ((bx0 + bx1) / 2, (by0 + by1) / 2)
        far = 80000
        ray = LineString([c, (c[0] + d[0] * far, c[1] + d[1] * far)])
        inter = ray.intersection(site_p.exterior)
        pts_ = [inter] if inter.geom_type == "Point" else list(getattr(inter, "geoms", []))
        best = None
        for p in pts_:
            dd = math.hypot(p.x - c[0], p.y - c[1])
            best = dd if best is None else max(best, dd)
        return c, best

    # ---- 用紙座標へ（敷地＋建物の外接を使える範囲の中央に）
    area = (sa["x"][0], sa["y"][0], sa["x"][1], sa["y"][1])
    ac = ((area[0] + area[2]) / 2, (area[1] + area[3]) / 2)
    sb = site_p.bounds
    off = (ac[0] - (sb[0] + sb[2]) / 2 / S, ac[1] - (sb[1] + sb[3]) / 2 / S)
    P = lambda x, y: (x / S + off[0], y / S + off[1])

    d = ezjww.new(paper_size=3)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import jww_env
    jww_env.apply_env(d)  # 環境設定 参考図面の .JWF（色・線幅・線種・レイヤグループ）
    g = d.options["layer_groups"][0]
    g["scale"] = S
    for i, n in enumerate(["境界線", "敷地外部", "外構", "マーク、物", "中心線、建物基礎", "CYAN", "ORANGE", "PINK",
                           "WHITE", "YELLOW", "文字", "ﾊｯﾁﾝｸﾞ", "ﾚｲﾔ-12", "ﾚｲﾔ-13", "ﾚｲﾔ-14", "図面枠"]):
        g["layers"][i]["name"] = n
    msp = d.modelspace()

    def line(p, q, layer, color, style):
        msp.add_line(p, q, jwwattribs={"layer": layer, "pen_color": color, "pen_style": style})

    def text(s, p, size=None, ang=0.0, color=1):
        size = size or cfg["text_size"]
        msp.add_text(s, p, (p[0] + size * 0.9 * len(s), p[1]), size_x=size, size_y=size, angle=ang % 360,
                     jwwattribs={"layer": cfg["text_layer"], "pen_color": color})

    sl, bl = cfg["site_line"], cfg["building_line"]
    # 敷地境界と辺の長さ
    ring = list(site_p.exterior.coords)[:-1]
    # 回した後の頂点の並び（元の順序を保つ）
    ring = [rot(affinity.rotate(__import__("shapely.geometry", fromlist=["Point"]).Point(p[2] * 1000, p[1] * 1000),
                                alpha, origin=(0, 0)), 90 * k) for p in pts]
    ring = [(q.x, q.y) for q in ring]
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        line(P(*a), P(*b), sl["layer"], sl["color"], sl["style"])
        Lm = math.hypot(b[0] - a[0], b[1] - a[1]) / 1000
        kind = "道路境界線" if i in case["road_edges"] else "隣地境界線"
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        if ang > 90:
            ang -= 180
        if ang <= -90:
            ang += 180
        mid = P((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        # 敷地の外側へ少しずらして書く
        cxp = P(site_p.centroid.x, site_p.centroid.y)
        vx, vy = mid[0] - cxp[0], mid[1] - cxp[1]
        vl = math.hypot(vx, vy) or 1
        tp = (mid[0] + vx / vl * 3 - 15 * math.cos(math.radians(ang)), mid[1] + vy / vl * 3 - 15 * math.sin(math.radians(ang)))
        text(f"{kind}　{Lm:.3f}", tp, 2.5, ang)
    # 道路境界線：続く道路の辺の両端を、隣地側へ road_extend_mm だけ延ばして続いているように描く（実線）
    ext_len = cfg["road_extend_mm"]
    roads = set(case["road_edges"])
    for i in roads:
        a, b = ring[i], ring[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        dl = math.hypot(dx, dy)
        ux, uy = dx / dl, dy / dl
        if (i - 1) % n not in roads:  # 連なりの始まり：a から後ろへ
            line(P(*a), P(a[0] - ux * ext_len, a[1] - uy * ext_len), sl["layer"], sl["color"], 1)
        if (i + 1) % n not in roads:  # 連なりの終わり：b から先へ
            line(P(*b), P(b[0] + ux * ext_len, b[1] + uy * ext_len), sl["layer"], sl["color"], 1)
    # 建物の外形
    bring = list(b1.exterior.coords)
    for i in range(len(bring) - 1):
        line(P(*bring[i]), P(*bring[i + 1]), bl["layer"], bl["color"], bl["style"])
    # 離れの寸法（北側の軸方向＋直交の左右）
    bx0, by0, bx1, by1 = b1.bounds
    for dvec, label in ((axis, "北側"), (cross, "横"), ((-cross[0], -cross[1]), "横")):
        c, dd = dist_to_boundary(b1, dvec)
        if dd is None:
            continue
        # 建物の縁から境界までの距離
        half = abs(dvec[0]) * (bx1 - bx0) / 2 + abs(dvec[1]) * (by1 - by0) / 2
        edge = (c[0] + dvec[0] * half, c[1] + dvec[1] * half)
        far = (c[0] + dvec[0] * dd, c[1] + dvec[1] * dd)
        line(P(*edge), P(*far), 0, 1, 1)
        mid = P((edge[0] + far[0]) / 2, (edge[1] + far[1]) / 2)
        text(f"{dd - half:,.0f}", (mid[0] + 1, mid[1] + 1), 2.0, 90 if abs(dvec[0]) < 0.5 else 0)
    # 方位記号（右上）：円＋北向きの矢印＋N
    cpx, cpy = ac[0] + (area[2] - ac[0]) - 25, area[3] - 25
    msp.add_circle((cpx, cpy), 5, jwwattribs={"layer": 0, "pen_color": 1, "pen_style": 1})
    tip = (cpx + nv[0] * 14, cpy + nv[1] * 14)
    tail = (cpx - nv[0] * 8, cpy - nv[1] * 8)
    line(tail, tip, 0, 1, 1)
    for s in (150, -150):
        a_ = math.radians(s) + math.atan2(nv[1], nv[0])
        line(tip, (tip[0] + 4 * math.cos(a_), tip[1] + 4 * math.sin(a_)), 0, 1, 1)
    text("N", (tip[0] + nv[0] * 2 - 1, tip[1] + nv[1] * 2), 3.0)
    # 図面枠・表題欄・印刷位置合わせ印（default.json と同じ）
    fr = cfg["frame"]
    W_, H_, T_ = fr["half_w"], fr["half_h"], fr["title_h"]
    for p, q in (((-W_, -H_), (W_, -H_)), ((W_, -H_), (W_, H_)), ((W_, H_), (-W_, H_)), ((-W_, H_), (-W_, -H_)),
                 ((-W_, -H_ + T_), (W_, -H_ + T_))):
        line(p, q, fr["layer"], fr["color"], fr["style"])
    text(case.get("title", case_name) + f"　S=1:{S}", (-W_ + 8, -H_ + 6), 4.0)
    default = json.load(open(ROOT / "05_設定" / "default.json", encoding="utf-8"))
    pm = default["print_marks_mm"]
    for X_, Y_ in pm["l_corners"]:
        for sx in (1, -1):
            for sy in (1, -1):
                line((sx * X_, sy * Y_), (sx * (X_ - pm["arm"]), sy * Y_), 15, 9, 9)
                line((sx * X_, sy * Y_), (sx * X_, sy * (Y_ - pm["arm"])), 15, 9, 9)
    Wp, Hp = pm["outline"]
    pts4 = [(-Wp, -Hp), (Wp, -Hp), (Wp, Hp), (-Wp, Hp)]
    for i in range(4):
        line(pts4[i], pts4[(i + 1) % 4], 15, 9, 9)

    tmp = Path(tempfile.gettempdir()) / "jw_site_plan"
    tmp.mkdir(exist_ok=True)
    tmp_file = tmp / "site.jww"
    d.saveas(str(tmp_file))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp_file, out_path)
    print("出力:", out_path)


if __name__ == "__main__":
    main()
