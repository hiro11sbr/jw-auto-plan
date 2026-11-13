"""図面づくりの画面（ブラウザで開く）。「図面を作る.bat」をダブルクリックで起動。

1. 仕様を選ぶ（05_設定/仕様/*.json → 05_設定/現在の仕様.txt に書く）
2. 入力図（PDF）をドラッグ＆ドロップ（01_入力 に保存し、図の画像を取り出す。1F・2F の画像は自動で選び、クリックで変えられる）
3. 「作業開始」で run_all.ps1 を走らせ、終わったら機械検査（check_conditions.py）までする
結果：02_出力/<プラン名>/ の .jww と確認画像。外には出ない（127.0.0.1 だけで待ち受ける）。
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps" / "py.exe"
PORT = 8765
DEFAULT_PDF = "plan_20261012.pdf"  # プラン1（出力は 02_出力 の直下）

job = {"running": False, "log": [], "done": False, "ok": None, "name": None, "started": 0.0, "result": None}
lock = threading.Lock()


def specs():
    names = sorted(p.stem for p in (ROOT / "05_設定" / "仕様").glob("*.json"))
    f = ROOT / "05_設定" / "現在の仕様.txt"
    cur = f.read_text(encoding="utf-8").strip() if f.exists() else ""
    return {"names": names, "current": cur}


def plan_name(pdf_name):
    stem = Path(pdf_name).stem
    return "" if pdf_name == DEFAULT_PDF else stem


def images_of(name):
    d = ROOT / "work" / "images" / name if name else ROOT / "work" / "images"
    out = []
    for p in sorted(d.glob("p*_x*_*.png")):
        m = re.match(r"p(\d+)_x(\d+)_(\d+)x(\d+)\.png", p.name)
        if m:
            out.append({"file": p.name, "page": int(m[1]), "xref": int(m[2]), "w": int(m[3]), "h": int(m[4]),
                        "url": "/file?p=" + urllib.parse.quote(str(p.relative_to(ROOT)))})
    out.sort(key=lambda r: (r["page"], r["xref"]))
    return out


def guess(imgs):
    """1F＝2ページ目、2F＝3ページ目の最初の図（今までのプラン1〜5はすべてこの並び）"""
    pick = {}
    for fl, page in (("1F", 2), ("2F", 3)):
        c = [r for r in imgs if r["page"] == page]
        if c:
            pick[fl] = c[0]["file"]
    if len(pick) < 2 and len(imgs) >= 2:  # 並びが違うPDF：大きい順の2枚をページ順に
        big = sorted(sorted(imgs, key=lambda r: -r["w"] * r["h"])[:2], key=lambda r: (r["page"], r["xref"]))
        pick = {"1F": big[0]["file"], "2F": big[1]["file"]}
    return pick


def extract(pdf_name):
    name = plan_name(pdf_name)
    outdir = ROOT / "work" / "images" / name if name else ROOT / "work" / "images"
    r = subprocess.run([str(PY), "-3.13", "-I", str(HERE / "extract_plans.py"), str(ROOT / "01_入力" / pdf_name), str(outdir)],
                       capture_output=True, env=dict(os.environ, PYTHONUTF8="1"))
    imgs = images_of(name)
    done = (ROOT / "02_出力" / name / "1F.jww").exists() if name else (ROOT / "02_出力" / "1F.jww").exists()
    return {"pdf": pdf_name, "name": name or "プラン1", "images": imgs, "guess": guess(imgs), "exists": done,
            "error": None if r.returncode == 0 else r.stderr.decode("utf-8", "replace")[-800:]}


def save_custom(c):
    """画面で入れた壁の厚みから仕様を保存し、保存名を返す。会社の仕様（数字を入れて作ったものでない json）は上書きしない"""
    name = str(c.get("name", "")).strip()
    if not name:
        raise ValueError("仕様の名前を入れてください")
    old = ROOT / "05_設定" / "仕様" / f"{Path(name).name}.json"
    if old.exists():
        try:
            mine = "_入力" in json.loads(old.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            mine = False
        if not mine:
            raise ValueError(f"「{name}」は前からある仕様なので上書きしません。別の名前にしてください")
    nums = [str(float(c.get(k, 0))) for k in ("ext", "air", "ins", "osb", "col", "pb")]
    parts = c.get("parts") or "generic"
    cmd = [str(PY), "-3.13", "-I", str(HERE / "custom_spec.py"), name, *nums, "--parts", *([parts] if isinstance(parts, str) else parts)]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, env=dict(os.environ, PYTHONUTF8="1"))
    if r.returncode != 0:
        err = decode(r.stderr).strip().splitlines()
        raise ValueError("仕様を保存できませんでした：" + (err[-1] if err else "不明"))
    return name


def decode(b):
    for enc in ("utf-8", "cp932"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", "replace")


def run_job(pdf_name, img1, img2, spec):
    name = plan_name(pdf_name)
    with lock:
        job.update(running=True, log=[], done=False, ok=None, name=name or "プラン1", started=time.time(), result=None)

    def say(s):
        with lock:
            job["log"].append(s)
    try:
        if spec:
            (ROOT / "05_設定" / "現在の仕様.txt").write_text(spec, encoding="utf-8")
            say(f"仕様：{spec}")
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "run_all.ps1"),
               "-Pdf", f"01_入力\\{pdf_name}", "-Img1F", img1, "-Img2F", img2]
        if name:
            cmd += ["-Name", name]
        say("図面づくりを始めます（数分かかります）")
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONUTF8="1"))
        for line in p.stdout:
            t = decode(line).rstrip()
            if t:
                say(t)
        p.wait()
        out = ROOT / "02_出力" / name if name else ROOT / "02_出力"
        mid = ROOT / "work" / "中間データ" / name if name else ROOT / "work" / "中間データ"
        res = {"out": str(out), "floors": []}
        say("機械検査をしています")
        for fl in ("1F", "2F"):
            jww, meta = out / f"{fl}.jww", mid / f"{fl}_meta.json"
            f = {"fl": fl, "jww": jww.exists(), "images": [], "ng": None, "check_img": None, "check_log": ""}
            for nm in (f"{fl}_建具の配置.png", f"{fl}_壁と柱.png"):
                if (out / nm).exists():
                    f["images"].append({"label": nm[3:-4], "url": "/file?p=" + urllib.parse.quote(str((out / nm).relative_to(ROOT)))})
            if jww.exists() and meta.exists():
                ci = ROOT / "work" / "検査" / f"{name or 'プラン1'}_{fl}.png"
                if ci.exists():
                    ci.unlink()
                r = subprocess.run([str(PY), "-3.13", "-I", str(HERE / "check_conditions.py"), str(jww), str(meta), str(ci)],
                                   capture_output=True, env=dict(os.environ, PYTHONUTF8="1"))
                f["check_log"] = decode(r.stdout)
                m = re.search(r"NG (\d+)、注意 (\d+)", f["check_log"])
                f["ng"], f["warn"] = (int(m[1]), int(m[2])) if m else (None, None)
                if ci.exists():
                    f["check_img"] = "/file?p=" + urllib.parse.quote(str(ci.relative_to(ROOT)))
            res["floors"].append(f)
        ok = p.returncode == 0 and all(f["jww"] for f in res["floors"])
        say("できました" if ok else "途中で止まりました（下の記録を見てください）")
        with lock:
            job.update(result=res, ok=ok)
    except Exception as e:  # noqa: BLE001
        say(f"エラー：{e}")
        with lock:
            job["ok"] = False
    finally:
        with lock:
            job.update(running=False, done=True)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            return self.send(200, (HERE / "app.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            pdfs = sorted(p.name for p in (ROOT / "01_入力").glob("*.pdf"))
            return self.send(200, {"specs": specs(), "pdfs": pdfs})
        if u.path == "/api/status":
            with lock:
                j = dict(job, log=job["log"][-400:], elapsed=round(time.time() - job["started"]) if job["started"] else 0)
            return self.send(200, j)
        if u.path.startswith("/ui/"):  # 画面の部品（ロゴ・背景）。scripts\ui のファイルを替えれば差し替わる
            p = (HERE / "ui" / Path(urllib.parse.unquote(u.path[4:])).name)
            ct = {".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg"}.get(p.suffix.lower())
            if not ct or not p.is_file():
                return self.send(404, {"error": "not found"})
            return self.send(200, p.read_bytes(), ct)
        if u.path == "/file":
            p = (ROOT / q.get("p", [""])[0]).resolve()
            if ROOT not in p.parents or not p.is_file() or p.suffix.lower() not in (".png", ".jpg"):
                return self.send(404, {"error": "not found"})
            return self.send(200, p.read_bytes(), "image/png")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        if u.path == "/api/upload":
            fn = Path(urllib.parse.unquote(self.headers.get("X-Filename", ""))).name
            if not fn.lower().endswith(".pdf") or not body.startswith(b"%PDF"):
                return self.send(400, {"error": "PDF ファイルを入れてください"})
            (ROOT / "01_入力" / fn).write_bytes(body)
            return self.send(200, extract(fn))
        data = json.loads(body or b"{}")
        if u.path == "/api/select":  # 前に入れたPDFを選ぶ
            fn = Path(data.get("pdf", "")).name
            if not (ROOT / "01_入力" / fn).exists():
                return self.send(404, {"error": "ない"})
            return self.send(200, extract(fn))
        if u.path == "/api/parts":  # 自分の部品集（.jww）を取り込む
            fn = Path(urllib.parse.unquote(self.headers.get("X-Filename", ""))).name
            if not fn.lower().endswith(".jww") or not body:
                return self.send(400, {"error": ".jww の部品集を入れてください"})
            d = ROOT / "05_設定" / "仕様" / "部品_取り込み"
            d.mkdir(parents=True, exist_ok=True)
            (d / fn).write_bytes(body)
            return self.send(200, {"path": str((d / fn).relative_to(ROOT)), "name": fn})
        if u.path == "/api/start":
            with lock:
                if job["running"]:
                    return self.send(409, {"error": "いま作業中です"})
            fn = Path(data["pdf"]).name
            spec = data.get("spec")
            c = data.get("custom")
            if c:  # 自分で数字を入れた仕様：05_設定/仕様/<保存名>.json を作ってから作図する（custom_spec.py）
                try:
                    spec = save_custom(c)
                except ValueError as e:
                    return self.send(400, {"error": str(e)})
            threading.Thread(target=run_job, args=(fn, data["img1"], data["img2"], spec), daemon=True).start()
            return self.send(200, {"ok": True, "spec": spec})
        if u.path == "/api/open":  # 出力フォルダを開く
            p = Path(data.get("path", "")).resolve()
            if p == ROOT or ROOT in p.parents:
                os.startfile(str(p))
                return self.send(200, {"ok": True})
            return self.send(400, {"error": "開けない"})
        self.send(404, {"error": "not found"})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"画面：{url}（この黒い窓を閉じると終わります）")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
