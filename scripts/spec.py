"""仕様ファイル（会社・シリーズごとの設定）を、基準の default.json に重ねて読む。

仕様ファイル 05_設定/仕様/<名前>.json は、基準と違う項目だけを書く：
  {"_extends": "../default.json", "placement": {"library": [...]}, "offsets_mm": {...}}
- 辞書は中身どうしを重ねる（書いた項目だけ置き換わる）。リストと数値は丸ごと置き換える。
- キーが "_" で始まるものは注釈（"_未確定" に、わかっていない項目を書いておく）。
- 今の仕様の名前は 05_設定/現在の仕様.txt（各スクリプトが current() で読む）。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ROOT / "05_設定"


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if k == "_extends":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_cfg(path):
    path = Path(path)
    cfg = json.load(open(path, encoding="utf-8"))
    base = cfg.get("_extends")
    if base:
        return _merge(load_cfg((path.parent / base).resolve()), cfg)
    return cfg


def current_name():
    f = SETTINGS / "現在の仕様.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""


def current_path():
    name = current_name()
    return SETTINGS / "仕様" / f"{name}.json" if name else SETTINGS / "default.json"


def current():
    return load_cfg(current_path())
