r"""Keep 03_建具部品\元データ一覧.csv: where each downloaded maker sheet came from, when it was
fetched, and whether the maker has published a newer one.

usage: check_parts_updates.py <parts_dir>          (record + check every sheet)
       check_parts_updates.py <parts_dir> --offline (record only, no network)

For each <maker>\<file>.jww the list holds: download URL, product page, the date it was
downloaded, size, MD5 of the local file, and the server's 公開日 (Last-Modified) and ETag.
Checking sends one HEAD request per file (no download) and reports 最新 / 更新あり / 確認できず.
A newer file is never downloaded automatically: the user asks for updates (2026-10-22).
"""
import csv
import datetime
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = {"LIXIL": "https://cad.biz-lixil.com/prod_data/2dcad/cadds/jww/"}
PRODUCT = "https://cad.biz-lixil.com/product/"
# product pages the sheets were found on (LIXIL CAD site, 2026-10-22)
PAGES = {
    "LILAC004": "CBMP_4359473", "LILAC005": "CBMP_4359470", "LILAC003": "CBMP_4359475",
    "LSMKE001": "CBMP_5087143", "LSMKE055": "CBMP_5087143", "LSMKE022": "CBMP_4359657",
    "LSMKE023": "CBMP_5087127", "LSMKE028": "CBMP_4359648", "LSMKE062": "CBMP_4359648",
    "LDGDG032": "CBMP_5085430", "LDGDG038": "CBMP_5085431",
}
NAMES = {
    "LILAD009": "ラシッサ 標準ドア 片開き", "LILAH001": "ラシッサ Vレール 片引戸",
    "LILAC004": "ラシッサ クローゼット 両開き戸", "LILAC005": "ラシッサ クローゼット 引違い戸", "LILAC003": "ラシッサ クローゼット 折戸",
    "LSMKE001": "EW 単体引違い窓 半外付 HH アングル無 マド ~W160",
    "LSMKE055": "EW 単体引違い窓 半外付 HH アングル無 マド W165~",
    "LSMKE022": "EW 縦すべり出し窓 単窓T型 アングル無", "LSMKE023": "EW 縦すべり出し窓 連窓TF型ほか アングル無",
    "LSMKE028": "EW FIX窓 F型・FF型 ~W119 アングル無", "LSMKE062": "EW FIX窓 F型・FF型 W133~ アングル無",
    "LDGDG032": "ジエスタ2 非防火 k2仕様 片開き C型", "LDGDG038": "ジエスタ2 非防火 k2仕様 親子 C型",
}
FIELDS = ["ファイル", "メーカー", "商品", "商品ページ", "ダウンロードURL", "取得日", "サイズ", "MD5",
          "公開日", "ETag", "確認日", "確認結果"]


def head(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.headers.get("Last-Modified", ""), (r.headers.get("ETag") or "").strip('"'), r.headers.get("Content-Length", "")


def main():
    parts_dir = Path(sys.argv[1])
    offline = "--offline" in sys.argv
    out = parts_dir / "元データ一覧.csv"
    old = {}
    if out.exists():
        old = {r["ファイル"]: r for r in csv.DictReader(open(out, encoding="utf-8-sig"))}
    rows = []
    today = datetime.date.today().isoformat()
    for maker, base in BASE.items():
        for f in sorted((parts_dir / maker).glob("*.jww")):
            key = f"{maker}/{f.name}"
            md5 = hashlib.md5(f.read_bytes()).hexdigest()
            r = dict(old.get(key, {}))
            r.update({"ファイル": key, "メーカー": maker, "商品": NAMES.get(f.stem, r.get("商品", "")),
                      "商品ページ": PRODUCT + PAGES[f.stem] if f.stem in PAGES else r.get("商品ページ", ""),
                      "ダウンロードURL": base + f.name, "サイズ": f.stat().st_size, "MD5": md5})
            if not r.get("取得日"):
                r["取得日"] = datetime.date.fromtimestamp(f.stat().st_mtime).isoformat()
            if not offline:
                try:
                    lm, etag, size = head(base + f.name)
                    r["確認日"] = today
                    if etag == md5 and str(size) == str(f.stat().st_size):
                        r["確認結果"] = "最新"
                        r["公開日"], r["ETag"] = lm, etag
                    else:
                        r["確認結果"] = f"更新あり（公開日 {lm}、サイズ {size}）"
                except Exception as e:  # noqa: BLE001
                    r["確認日"], r["確認結果"] = today, f"確認できず（{type(e).__name__}）"
            rows.append(r)
            print(key, r.get("確認結果", "記録のみ"))
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
