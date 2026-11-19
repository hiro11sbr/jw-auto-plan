"""UB の部品の4つの角にある L字ガイドを探す（部品集を作る時と、図面に置く時の両方で使う）。"""
import math


def _dir(v):
    return (0 if abs(v[0]) < 1e-6 else math.copysign(1, v[0]), 0 if abs(v[1]) < 1e-6 else math.copysign(1, v[1]))


def guide_box(segs, size=(30.0, 80.0), tol=0.6):
    """UB の4つの角にある L字のガイド（2本の短い線が端で直角に会う。50mm 前後）を探し、
    その4つの角を結ぶ長方形 (x0, y0, x1, y1) を返す。そろわなければ None（L字のない部品もある。user 11/18）。
    ガイドの長方形の中心が UB の中心（user 11/18：外形の線の範囲で取ると 7.5mm ずれ、ガイドが壁に埋まった）。
    近くに別の短い直角（ドアの枠など）があっても、内側を向いた4つの角がそろう長方形のうち、いちばん大きいものを取る。"""
    lo, hi = size
    short = [(p, q) for p, q in segs if lo <= math.dist(p, q) <= hi
             and (abs(p[0] - q[0]) < 1e-6 or abs(p[1] - q[1]) < 1e-6)]  # 縦か横の線だけ
    ends = []
    for p, q in short:
        ends.append((p, _dir((q[0] - p[0], q[1] - p[1]))))
        ends.append((q, _dir((p[0] - q[0], p[1] - q[1]))))
    # 角：同じ点から、横向き1本と縦向き1本が出ている。向きの組 (sx, sy) が角の種類
    corners = {}
    for i, (c, d1) in enumerate(ends):
        for c2, d2 in ends[i + 1:]:
            if math.dist(c, c2) <= tol and d1[0] * d2[0] + d1[1] * d2[1] == 0:
                h, v = (d1, d2) if d1[1] == 0 else (d2, d1)
                corners.setdefault((h[0], v[1]), []).append(c)
    best = None
    for bl in corners.get((1, 1), []):        # 左下：右と上へ
        for tr in corners.get((-1, -1), []):  # 右上：左と下へ
            if tr[0] - bl[0] < 500 or tr[1] - bl[1] < 500:
                continue
            br = any(math.dist(c, (tr[0], bl[1])) <= tol for c in corners.get((-1, 1), []))
            tl = any(math.dist(c, (bl[0], tr[1])) <= tol for c in corners.get((1, -1), []))
            if br and tl:
                area = (tr[0] - bl[0]) * (tr[1] - bl[1])
                if best is None or area > best[0]:
                    best = (area, (bl[0], bl[1], tr[0], tr[1]))
    return best[1] if best else None
