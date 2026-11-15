"""図面づくりの画面（ブラウザで開く）。「図面を作る.bat」をダブルクリックで起動。

1. 仕様を選ぶ（05_設定/仕様/*.json → 05_設定/現在の仕様.txt に書く）
2. 入力図（PDF）をドラッグ＆ドロップ（01_入力 に保存し、図の画像を取り出す。1F・2F の画像は自動で選び、クリックで変えられる）
3. 「作業開始」で run_all.ps1 を走らせ、終わったら機械検査（check_conditions.py）までする
結果：02_出力/<プラン名>/ の .jww と確認画像。外には出ない（127.0.0.1 だけで待ち受ける）。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # 「-I」で起動すると、この場所が自動では入らない（ui_plot を読み込むため）
ROOT = HERE.parent
PY = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps" / "py.exe"
PORT = 8765
DEFAULT_PDF = "plan_20261012.pdf"  # プラン1（出力は 02_出力 の直下）

FLOORS = ("1F", "2F", "3F")  # 扱える階。実際に作る階は、作図ごとに job["floors"]（2階建て＝1F・2F、3階建て＝1F・2F・3F）
job = {"floors": ("1F", "2F"), "running": False, "log": [], "done": False, "ok": None, "name": None, "started": 0.0, "result": None,
       "plan": "", "img": {}, "T": {}, "ended": None, "saved_to": None}
lock = threading.Lock()
_plot_cache = {}
TMP_SPEC = "作図用_保存なし"  # 保存しないで作図する時の、一時の仕様の名前（毎回上書き・一覧には出さない）


def specs():
    names = sorted(p.stem for p in (ROOT / "05_設定" / "仕様").glob("*.json") if p.stem != TMP_SPEC)  # 保存しないで作図した時の一時の仕様は、一覧に出さない
    f = ROOT / "05_設定" / "現在の仕様.txt"
    cur = f.read_text(encoding="utf-8").strip() if f.exists() else ""
    return {"names": names, "current": cur if cur in names else ""}


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
    c3 = [r for r in imgs if r["page"] == 4]
    if c3:
        pick["3F"] = c3[0]["file"]   # 3階建ての時だけ使う（画面で「3階建て」を選んだ時）
    if len(pick) < 2 and len(imgs) >= 2:  # 並びが違うPDF：大きい順の2枚をページ順に
        big = sorted(sorted(imgs, key=lambda r: -r["w"] * r["h"])[:2], key=lambda r: (r["page"], r["xref"]))
        pick = {"1F": big[0]["file"], "2F": big[1]["file"]}
    return pick


IMG_EXT = (".png", ".jpg", ".jpeg")
_IMG2PDF = r"""
import sys, pymupdf as fitz
src, dst = sys.argv[1], sys.argv[2]
pix = fitz.Pixmap(src)
doc = fitz.open(); pg = doc.new_page(width=pix.width, height=pix.height)
pg.insert_image(pg.rect, filename=src)
doc.save(dst)
print(pix.width, pix.height)
"""


def img_to_pdf(data):
    """画像（PNG・JPG）を、その画像だけが入った1ページの PDF にする（作図は PDF から図を取り出す決まりのため）"""
    tmp = Path(os.environ.get("TEMP", "")) / "jw_by_claude_img"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    src, dst = tmp / ("in.jpg" if data[:3] == b"\xff\xd8\xff" else "in.png"), tmp / "out.pdf"
    src.write_bytes(data)
    r = subprocess.run([str(PY), "-3.13", "-I", "-c", _IMG2PDF, str(src), str(dst)], capture_output=True, env=dict(os.environ, PYTHONUTF8="1"))
    if r.returncode != 0 or not dst.exists():
        raise ValueError("画像を読み込めませんでした")
    w, h = (int(x) for x in decode(r.stdout).split()[:2])
    if w * h < 1_000_000 or min(w, h) < 600:   # extract_plans.py が図として拾う大きさ
        raise ValueError(f"画像が小さすぎます（{w}×{h}）。寸法の数字が読める大きさの画像を入れてください")
    out = dst.read_bytes()
    shutil.rmtree(tmp, ignore_errors=True)
    return out


ADD_PAGE = 100  # あとから足した PDF の図は、ページの番号を 100 番台・200 番台…にして、最初の PDF の図と名前がぶつからないようにする


def add_pdf(plan_pdf, pdf_name):
    """図を選ぶ画面で、別の PDF を足す。図を取り出して、最初の PDF の図と同じフォルダに写す（作図は1つのフォルダの図を使うため）"""
    name = plan_name(plan_pdf)
    outdir = ROOT / "work" / "images" / name if name else ROOT / "work" / "images"
    tmp = Path(os.environ.get("TEMP", "")) / "jw_by_claude_add"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    r = subprocess.run([str(PY), "-3.13", "-I", str(HERE / "extract_plans.py"), str(ROOT / "01_入力" / pdf_name), str(tmp)],
                       capture_output=True, env=dict(os.environ, PYTHONUTF8="1"))
    top = max([i["page"] for i in images_of(name)] + [0])
    off = (top // ADD_PAGE + 1) * ADD_PAGE
    for p in sorted(tmp.glob("p*_x*_*.png")):
        m = re.match(r"p(\d+)_(x\d+_\d+x\d+\.png)", p.name)
        if m:
            shutil.copy2(p, outdir / f"p{int(m[1]) + off}_{m[2]}")
    shutil.rmtree(tmp, ignore_errors=True)
    return {"images": images_of(name), "error": None if r.returncode == 0 else r.stderr.decode("utf-8", "replace")[-800:]}


def extract(pdf_name):
    name = plan_name(pdf_name)
    outdir = ROOT / "work" / "images" / name if name else ROOT / "work" / "images"
    for p in outdir.glob("p*_x*_*.png"):   # 前に足した PDF の図（100 番台より上）は、PDF を置き直したら消す
        m = re.match(r"p(\d+)_", p.name)
        if m and int(m[1]) >= ADD_PAGE:
            p.unlink()
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


def dirs(name):
    """プランの 中間データ・出力・一時（run_all.ps1 が make_plan の結果を一時に書く所）"""
    mid = ROOT / "work" / "中間データ" / name if name else ROOT / "work" / "中間データ"
    out = ROOT / "02_出力" / name if name else ROOT / "02_出力"
    tmp = Path(os.environ.get("TEMP", "")) / "jw_by_claude"
    return mid, out, (tmp / name if name else tmp)


def _mtime(p, t0):
    try:
        m = p.stat().st_mtime
    except OSError:
        return None
    return m if m >= t0 else None


def watch(stop, name, t0, T, floors):
    """作図の進み具合を、できるファイルの時刻で見る（run_all.ps1 には手を入れない）。
    部屋名の読み取り → 壁の判定（腰壁）→ 建具の判定（区画の確認）→ 階ごとの図面（一時の建具の配置）の順に、ファイルができる"""
    mid, out, tmp = dirs(name)
    t0 -= 1.0

    def once():
        probes = {"p0": mid / f"{floors[0]}_部屋名.json"}
        for fl in floors:
            probes[f"walls_{fl}"] = mid / f"{fl}_腰壁.json"
            probes[f"done_{fl}"] = mid / f"{fl}_区画の確認.png"
            probes[f"draw_{fl}"] = tmp / f"{fl}_建具の配置.png"
        for k, pth in probes.items():
            if k not in T:
                m = _mtime(pth, t0)
                if m:
                    T[k] = m
    while not stop.wait(0.25):
        once()
    once()


def progress(j, now):
    """作成中の画面に渡す、フェーズごとの状態と時間"""
    fls = tuple(j.get("floors") or ("1F", "2F"))
    T = dict(j.get("T") or {})
    t_start, running, done, ok, ended = j["started"], j["running"], j["done"], j["ok"], j.get("ended")
    failed = bool(done and not ok)
    firsts = [v for k, v in T.items() if k.startswith(("walls_", "done_"))]
    p0 = T.get("p0") or (min(firsts) if firsts else None)
    segs, prev = [("extract", t_start, p0)], p0
    for fl in fls:
        d = T.get(f"done_{fl}")
        w = T.get(f"walls_{fl}") or d
        if w and prev and w < prev:
            w = prev
        segs.append((f"walls_{fl}", prev, w))
        segs.append((f"open_{fl}", w, d))
        prev = d
    judged = prev
    pd = judged
    for fl in fls:
        e = T.get(f"draw_{fl}")
        segs.append((f"draw_{fl}", pd, e))
        pd = e
    segs.append(("check", T.get("check_start"), T.get("check_end")))
    if done and ok:  # 終わったのに印のファイルが見つからなかった所は、終わった扱いにする（表示だけ）
        fixed, last = [], t_start
        for k, s, e in segs:
            s = s or last
            e = e or max(s, ended or s)
            fixed.append((k, s, e))
            last = e
        segs = fixed
    cur = next((k for k, s, e in segs if e is None), None)
    seg = {k: (s, e) for k, s, e in segs}

    def tot(prefix):
        t_, live = 0.0, None
        for k, s, e in segs:
            if k.startswith(prefix):
                if s and e:
                    t_ += max(0.0, e - s)
                elif k == cur and s:
                    live = s
        return round(t_, 1), live

    def state(prefix):
        ks = [k for k, s, e in segs if k.startswith(prefix)]
        if all(seg[k][1] for k in ks):
            return "done"
        if cur in ks:
            return "fail" if failed else "now"
        return "part" if any(seg[k][1] for k in ks) else "pend"

    def dots(prefix):
        return [{"fl": fl, "on": bool(seg[f"{prefix}_{fl}"][1]), "go": cur == f"{prefix}_{fl}" and not failed} for fl in fls]

    rows = []

    def add(rid, label, prefix, dts=None):
        s_, live = tot(prefix)
        rows.append({"kind": "row", "id": rid, "label": label, "state": state(prefix), "sec": s_,
                     "live": live if not failed else None, "dots": dts})
    add("extract", "図を読み込む", "extract")
    add("walls", "壁を判定", "walls_", dots("walls"))
    add("open", "建具を判定", "open_", dots("open"))
    rows.append({"kind": "tr", "id": "tr1", "label": "判定が終わり、作図へ", "on": bool(judged)})
    for fl in fls:
        add(f"draw_{fl}", f"{fl[:-1]}階を作図", f"draw_{fl}")
    rows.append({"kind": "tr", "id": "tr2", "label": "作図が終わり、検査へ", "on": bool(pd) and (T.get("check_start") is not None or done)})
    add("check", "検査", "check")
    res = j.get("result") or {}
    ngs = sum((f.get("ng") or 0) for f in res.get("floors", []))
    finished = bool(done and ok)
    rows.append({"kind": "row", "id": "fin", "state": "done" if finished else "pend",
                 "sec": round((ended or now) - t_start, 1) if finished else 0,
                 "label": ("完了" if not ngs else f"完了（直す所 {ngs}）") if finished else "完了", "live": None, "dots": None})

    # 右半分に出す図：描き終わった階があればその図面（白線）、なければ読んでいる階の元の図
    plan = j.get("plan", "")
    last_plot = next((fl for fl in reversed(fls) if T.get(f"draw_{fl}") or (finished and seg[f"draw_{fl}"][1])), None)
    if last_plot:
        prev_ = {"kind": "plot", "fl": last_plot,
                 "url": f"/api/plot?plan={urllib.parse.quote(plan)}&fl={last_plot}&v={int(T.get('draw_' + last_plot) or ended or 0)}"}
    else:
        tail = cur.rsplit("_", 1)[-1] if cur and "_" in cur else ""
        fl = tail if tail in fls else fls[0]
        im = (j.get("img") or {}).get(fl)
        d_ = ROOT / "work" / "images" / plan if plan else ROOT / "work" / "images"
        prev_ = {"kind": "scan", "fl": fl, "url": ("/file?p=" + urllib.parse.quote(str((d_ / im).relative_to(ROOT)))) if im else None}
    files = []
    if finished:
        for f in res.get("floors", []):
            if f.get("jww"):
                pth = dirs(plan)[1] / f"{f['fl']}.jww"
                files.append({"fl": f["fl"], "name": deliver_name(f["fl"]), "size": pth.stat().st_size if pth.exists() else 0,
                              "ng": f.get("ng"), "warn": f.get("warn"), "read": f.get("read", 0),
                              "url": f"/download?plan={urllib.parse.quote(plan)}&fl={f['fl']}"})
    return {"rows": rows, "title": "途中で止まりました" if failed else ("完了" if finished else "作成中"), "failed": failed,
            "finished": finished, "total": round((ended or now) - t_start, 1) if t_start else 0, "live_total": bool(running),
            "now": now, "preview": prev_, "files": files, "saved_to": j.get("saved_to")}


def pick_folder():
    """Windows のフォルダ選択の窓を出して、選んだフォルダを返す（やめたら None）"""
    ps = ("Add-Type -AssemblyName System.Windows.Forms; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
          "$f=New-Object System.Windows.Forms.Form; $f.TopMost=$true; $f.ShowInTaskbar=$false; $f.StartPosition='CenterScreen'; "
          "$f.Opacity=0; $f.Show(); $d=New-Object System.Windows.Forms.FolderBrowserDialog; "
          "$d.Description='図面を入れるフォルダを選んでください'; $d.ShowNewFolderButton=$true; "
          "if($d.ShowDialog($f) -eq 'OK'){ $d.SelectedPath }; $f.Close()")
    r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps], capture_output=True, timeout=900)
    return decode(r.stdout).strip() or None


def deliver_name(fl):
    """納品のファイルの名前（02_出力 の中は 1F.jww のまま。渡す時だけ「1階平面詳細図.jww」にする）"""
    return f"{fl[:-1]}階平面詳細図.jww"


def deliver(dest_root):
    """納品のファイルを、選んだフォルダの中の「プラン名」フォルダにコピーする（前のファイルを上書きしない）"""
    plan = job.get("plan", "")
    out = dirs(plan)[1]
    base = plan or "プラン1"
    dest, k = Path(dest_root) / base, 2
    while dest.exists():
        dest = Path(dest_root) / f"{base}_{k}"
        k += 1
    dest.mkdir(parents=True)
    names = []
    for fl in job.get("floors") or ("1F", "2F"):
        src = out / f"{fl}.jww"
        if src.exists():
            shutil.copy2(src, dest / deliver_name(fl))
            names.append(deliver_name(fl))
    with lock:
        job["saved_to"] = str(dest)
    return {"dir": str(dest), "files": names}


def decode(b):
    for enc in ("utf-8", "cp932"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", "replace")


def run_job(pdf_name, imgs, spec):
    floors = tuple(fl for fl in FLOORS if imgs.get(fl))
    name = plan_name(pdf_name)
    with lock:
        job.update(running=True, log=[], done=False, ok=None, name=name or "プラン1", started=time.time(), result=None,
                   plan=name, img=dict(imgs), floors=floors, T={}, ended=None, saved_to=None)
    stop = threading.Event()
    threading.Thread(target=watch, args=(stop, name, job["started"], job["T"], floors), daemon=True).start()

    def say(s):
        with lock:
            job["log"].append(s)
    try:
        if spec:
            (ROOT / "05_設定" / "現在の仕様.txt").write_text(spec, encoding="utf-8")
            say(f"仕様：{spec}")
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "run_all.ps1"),
               "-Pdf", f"01_入力\\{pdf_name}", "-Img1F", imgs["1F"], "-Img2F", imgs["2F"]]
        if imgs.get("3F"):
            cmd += ["-Img3F", imgs["3F"]]
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
        stop.set()
        job["T"]["proc_end"] = time.time()
        job["T"]["check_start"] = time.time()
        out = ROOT / "02_出力" / name if name else ROOT / "02_出力"
        mid = ROOT / "work" / "中間データ" / name if name else ROOT / "work" / "中間データ"
        res = {"out": str(out), "floors": []}
        say("機械検査をしています")
        for fl in floors:
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
            # 読み取りの検査（run_all.ps1 が判定のあとに出す）：確定ではなく「見てほしい所」
            rj, rp = mid / f"{fl}_読み取りの検査.json", out / f"{fl}_読み取りの検査.png"
            f.update(read=0, read_msgs=[], read_fix=[], read_img=None)
            if rj.exists():
                try:
                    d = json.loads(rj.read_text(encoding="utf-8"))
                    f["read_msgs"] = [str(x.get("msg", "")) for x in d.get("疑い", [])]
                    f["read"], f["read_fix"] = len(f["read_msgs"]), [str(x) for x in d.get("自動修正", [])]
                except Exception:
                    pass
            if f["read"] and rp.exists():
                f["read_img"] = "/file?p=" + urllib.parse.quote(str(rp.relative_to(ROOT)))
            res["floors"].append(f)
        job["T"]["check_end"] = time.time()
        ok = p.returncode == 0 and all(f["jww"] for f in res["floors"])
        say("できました" if ok else "途中で止まりました（下の記録を見てください）")
        with lock:
            job.update(result=res, ok=ok)
    except Exception as e:  # noqa: BLE001
        say(f"エラー：{e}")
        with lock:
            job["ok"] = False
    finally:
        stop.set()
        with lock:
            job.update(running=False, done=True, ended=time.time())


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k_, v_ in (headers or {}).items():
            self.send_header(k_, v_)
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
            if j["started"]:
                j["view"] = progress(j, time.time())
            return self.send(200, j)
        if u.path == "/api/plot":  # 作った図面（jww）を白線の画像にして返す（作成中の右半分）
            plan, fl = q.get("plan", [""])[0], q.get("fl", [""])[0]
            if Path(plan).name != plan or fl not in FLOORS:
                return self.send(404, {"error": "not found"})
            mid, out, tmp = dirs(plan)
            pth = next((x for x in (tmp / f"{fl}.jww", out / f"{fl}.jww") if x.is_file() and x.stat().st_mtime >= job["started"] - 1), None) \
                or (out / f"{fl}.jww" if (out / f"{fl}.jww").is_file() else None)
            if not pth:
                return self.send(404, {"error": "not found"})
            key = (str(pth), pth.stat().st_mtime_ns)
            if key not in _plot_cache:
                import ui_plot
                if len(_plot_cache) >= 6:
                    _plot_cache.clear()
                _plot_cache[key] = ui_plot.render_png(pth)
            data = _plot_cache[key]
            return self.send(200, data, "image/png") if data else self.send(404, {"error": "empty"})
        if u.path == "/download":  # 納品のファイル（ドラッグして好きな所へ落とせる）
            plan, fl = q.get("plan", [""])[0], q.get("fl", [""])[0]
            pth = dirs(plan)[1] / f"{fl}.jww" if Path(plan).name == plan and fl in FLOORS else None
            if not pth or not pth.is_file():
                return self.send(404, {"error": "not found"})
            return self.send(200, pth.read_bytes(), "application/octet-stream",
                             {"Content-Disposition": "attachment; filename*=UTF-8''" + urllib.parse.quote(deliver_name(fl))})
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
            if Path(fn).suffix.lower() in IMG_EXT:   # 画像：1ページの PDF に包んでから、PDF と同じ流れにする
                try:
                    body, fn = img_to_pdf(body), Path(fn).stem + ".pdf"
                except ValueError as e:
                    return self.send(400, {"error": str(e)})
            if not fn.lower().endswith(".pdf") or not body.startswith(b"%PDF"):
                return self.send(400, {"error": "PDF か画像（PNG・JPG）を入れてください"})
            (ROOT / "01_入力" / fn).write_bytes(body)
            plan = Path(urllib.parse.unquote(self.headers.get("X-Plan", ""))).name
            if plan and plan != fn and (ROOT / "01_入力" / plan).exists():   # 図を選ぶ画面で足した PDF：今の PDF の図に加える
                return self.send(200, add_pdf(plan, fn))
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
        if u.path == "/api/save":  # 納品のファイルの保存先を選ばせて、コピーする
            with lock:
                ready_ = job["done"] and job["ok"]
            if not ready_:
                return self.send(400, {"error": "まだ作図が終わっていません"})
            try:
                d = pick_folder()
            except Exception as e:  # noqa: BLE001
                return self.send(500, {"error": f"フォルダを選ぶ窓を出せませんでした：{e}"})
            if not d:
                return self.send(200, {"canceled": True})
            try:
                return self.send(200, deliver(d))
            except OSError as e:
                return self.send(500, {"error": f"保存できませんでした：{e}"})
        if u.path == "/api/open_saved":  # 保存したフォルダを開く
            d = job.get("saved_to")
            if d and Path(d).is_dir():
                os.startfile(d)
                return self.send(200, {"ok": True})
            return self.send(400, {"error": "保存先がありません"})
        if u.path == "/api/save_spec":  # 自分で入れた数字を、名前を付けて仕様として保存する（作図とは別）
            if str(data.get("name", "")).strip() == TMP_SPEC:
                return self.send(400, {"error": "この名前は使えません。別の名前にしてください"})
            try:
                return self.send(200, {"ok": True, "name": save_custom(data), "specs": specs()})
            except ValueError as e:
                return self.send(400, {"error": str(e)})
        if u.path == "/api/start":
            with lock:
                if job["running"]:
                    return self.send(409, {"error": "いま作業中です"})
            fn = Path(data["pdf"]).name
            spec = data.get("spec")
            c = data.get("custom")
            if c:  # 自分で数字を入れた仕様：保存しない（一時の「_作図用」に書いて作図する。保存は「保存する」ボタンで別に）
                try:
                    spec = save_custom(dict(c, name=TMP_SPEC))
                except ValueError as e:
                    return self.send(400, {"error": str(e)})
            threading.Thread(target=run_job, args=(fn, {k: data.get(f"img{i}") for i, k in ((1, "1F"), (2, "2F"), (3, "3F"))}, spec), daemon=True).start()
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
