"""測量図の座標の表（求積表・面積計算簿）を読み取り、面積で検算して敷地の座標の json にする。配置図用（別セッション 11/12）。

usage: read_survey_table.py <PDFまたは画像> [ページ番号(0始まり、既定0)] [出力.json]
Windows 標準の日本語OCR（ocr_windows.ps1）で文字の位置を取り、
  「x座標 y座標」の行（小数3桁の数字2つ）を点として拾い、行の間隔で区画ごとに分け、
  座標から面積を計算して、表に書かれた面積（小数7桁）と照合する。一致しない区画は「要確認」。
マイナスはOCRで長音「ー」などになるので、数字のすぐ左の短い横棒をマイナスとみなす（面積の一致で確かめる）。
読み取れるのは活字の表だけ。手書きの図は読めない。
"""
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
MINUS = set("-ー—−―‐-─一")


def ocr_words(img_path):
    out = tempfile.mktemp(suffix=".json")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(HERE / "ocr_windows.ps1"), str(img_path), out], check=True, capture_output=True)
    data = json.load(open(out, encoding="utf-8-sig"))
    os.remove(out)
    lines = data.get("lines") or []
    if isinstance(lines, dict):
        lines = [lines]
    words = []
    for ln in lines:
        ws = ln.get("words") or []
        if isinstance(ws, dict):
            ws = [ws]
        words += ws
    return words


def render(path, page, dpi=350):
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return path
    import pymupdf
    tmp = tempfile.mktemp(suffix=".png")
    pymupdf.open(str(path))[page].get_pixmap(dpi=dpi).save(tmp)
    return tmp


def parse_numbers(words, ps=1.0):
    """小数の数字（「70692.」＋「466」のように割れたもの、「-99854.」のように先頭にマイナスが付いたものも）を、位置・符号つきで集める。"""
    ws = sorted(words, key=lambda w: (round((w["y"] + w["h"] / 2) / 10), w["x"]))
    nums = []
    i = 0
    while i < len(ws):
        w = ws[i]
        t = w["text"]
        cy = w["y"] + w["h"] / 2
        neg = False
        while t and t[0] in MINUS:
            neg = True
            t = t[1:]
        m = re.fullmatch(r"(\d+)\.(\d*)", t)
        if m:
            frac = m.group(2)
            xr = w["x"] + w["w"]
            j = i + 1
            if not frac:
                # 割れた小数部：同じ行で右隣にある数字だけの文字（行の分け方に左右されないよう全体から探す）
                xr = w["x"] + w["w"]
                cand = [o for o in words if re.fullmatch(r"\d+", o["text"]) and abs(o["y"] + o["h"] / 2 - cy) < 12 * ps
                        and 0 <= o["x"] - xr < 40 * ps]
                if cand:
                    frac = min(cand, key=lambda o: o["x"])["text"]
            if frac:
                if not neg:  # 左隣の短い横棒（長音など）＝マイナス
                    neg = any(len(o["text"]) <= 2 and o["text"] and set(o["text"]) <= MINUS | {" "}
                              for o in words
                              if abs(o["y"] + o["h"] / 2 - cy) < 14 * ps and 0 <= w["x"] - (o["x"] + o["w"]) < 30 * ps)
                nums.append(dict(v=float(f"{m.group(1)}.{frac}") * (-1 if neg else 1), dec=len(frac),
                                 ints=len(m.group(1)), x=w["x"], y=cy, text=f"{m.group(1)}.{frac}"))
        i += 1
    return nums


def poly_area(pts):
    a = 0.0
    for k in range(len(pts)):
        p, q = pts[k], pts[(k + 1) % len(pts)]
        a += p[2] * q[1] - q[2] * p[1]
    return abs(a) / 2


def main():
    src = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE.parent / "work" / "配置図" / "座標読取.json"
    # OCRは解像度で読み落としが変わるので、解像度を変えて何度も読み、行ごとに多数決で合わせる（位置は用紙のpt単位にそろえる）
    from collections import Counter
    obs = []  # (y_pt, x_pt, v)
    allw = []
    for dpi in (250, 300, 350, 400):
        img = render(src, page, dpi)
        words = ocr_words(img)
        k = 72.0 / dpi
        for n in parse_numbers(words):
            if n["dec"] == 3 and n["ints"] >= 4:
                obs.append((n["y"] * k, n["x"] * k, n["v"]))
        if dpi == 250:
            for w in words:
                allw.append(dict(text=w["text"], x=w["x"] * k, y=w["y"] * k, w=w["w"] * k, h=w["h"] * k))
    # 小さく区切って拡大して読み直す（全体では読み落とす行がある）。区切りの境目で切れないよう少し重ねる
    if Path(src).suffix.lower() == ".pdf":
        import pymupdf
        pg = pymupdf.open(str(src))[page]
        H, W = pg.rect.height, pg.rect.width
        y0 = 0.0
        while y0 < H:
            # 幅の広いままだと数字の縦の列を縦書きの文字として読むことがあるので、幅も細かく区切る
            x0 = 0.0
            while x0 < W:
                clip = pymupdf.Rect(x0, max(0, y0 - 15), min(W, x0 + 280), min(H, y0 + 95))
                tmp = tempfile.mktemp(suffix=".png")
                pg.get_pixmap(dpi=432, clip=clip).save(tmp)
                k = 72.0 / 432
                for n in parse_numbers(ocr_words(tmp), 432 / 250):
                    if n["dec"] == 3 and n["ints"] >= 4:
                        obs.append((clip.y0 + n["y"] * k, clip.x0 + n["x"] * k, n["v"]))
                os.remove(tmp)
                x0 += 200
            y0 += 80
    # 列（x座標・y座標）：数字の左端 x_pt を近いものどうしでまとめ、観測の多い上位2列
    xs = sorted(o[1] for o in obs)
    cols = []
    for x in xs:
        if cols and x - cols[-1][-1] < 8:
            cols[-1].append(x)
        else:
            cols.append([x])
    cols = sorted(sorted(cols, key=len, reverse=True)[:2], key=lambda c: c[0])
    cx = [statistics.mean(c) for c in cols]
    rowvals = {}
    for y, x, v in obs:
        ci = min(range(len(cx)), key=lambda i: abs(cx[i] - x))
        if abs(cx[ci] - x) > 10:
            continue
        rowvals.setdefault(round(y / 4), [[], []] if False else {0: [], 1: [], "y": []})
        rowvals[round(y / 4)][ci].append(v)
        rowvals[round(y / 4)]["y"].append(y)
    # 隣の行キーに割れた観測をまとめる（行間は約10pt）
    pts = []
    for key in sorted(rowvals):
        r = rowvals[key]
        if not r[0] or not r[1]:
            continue
        vx = Counter(r[0]).most_common(1)[0][0]
        vy = Counter(r[1]).most_common(1)[0][0]
        ym = statistics.mean(r["y"])
        left = sorted([w for w in allw if abs(w["y"] + w["h"] / 2 - ym) < 4 and w["x"] + w["w"] < cx[0] - 2], key=lambda w: w["x"])
        name = "".join(w["text"] for w in left[-1:]) if left else "?"
        pts.append((name, vx, vy, ym))
    nums = []
    for dpi in (250, 300, 350):
        ww = ocr_words(render(src, page, dpi))
        k = 72.0 / dpi
        for n in parse_numbers(ww):
            if n["dec"] == 7:
                nums.append(dict(v=n["v"], ints=n["ints"], dec=7, y=n["y"] * k))
    if not pts:
        raise SystemExit("座標の行が見つからない")
    # 区画ごとに分ける：行の間隔が普通の2.2倍を超えたら別の区画
    gaps = [b[3] - a[3] for a, b in zip(pts, pts[1:])]
    pitch = statistics.median(gaps) if gaps else 40
    blocks, cur = [], [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if b[3] - a[3] > pitch * 2.2:
            blocks.append(cur)
            cur = []
        cur.append(b)
    blocks.append(cur)
    # 表の面積：小数7桁の数字（面積計算簿・求積表の「合計面積」）
    stated = sorted({(round(n["y"] / 4), n["v"]): n for n in nums if n["ints"] >= 2}.values(), key=lambda n: n["y"])
    result = []
    for blk in blocks:
        # 最後の点が最初の点と同じ（閉じている）なら落とす
        if len(blk) > 2 and abs(blk[0][1] - blk[-1][1]) < 1e-3 and abs(blk[0][2] - blk[-1][2]) < 1e-3:
            blk = blk[:-1]
        if len(blk) < 3:
            continue
        calc = poly_area(blk)
        y_end = blk[-1][3]
        # このブロックの下にある最初の面積（7桁）
        st = next((n for n in stated if n["y"] > y_end - 5), None)
        ok = st is not None and abs(st["v"] - calc) < 0.02
        result.append(dict(points=[[p[0], p[1], p[2]] for p in blk], area_calc=round(calc, 4),
                           area_stated=st["v"] if st else None, match=ok))
    for i, r in enumerate(result, 1):
        print(f"区画{i}: 点{len(r['points'])}個 計算面積 {r['area_calc']} ㎡ / 表の面積 {r['area_stated']} → "
              + ("一致" if r["match"] else "要確認（符号か数字の読み違い）"))
        for p in r["points"]:
            print("   ", p[0], p[1], p[2])
    json.dump(dict(_note="read_survey_table.py の読み取り結果。x=北、y=東（m）。match が false の区画は人が確認する",
                   parcels=result), open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("出力:", out)


if __name__ == "__main__":
    main()
