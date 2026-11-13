"""環境設定「参考図面の .JWF」を、作る .jww すべてに入れる（user 11/13）。

参考図面の .JWF（05_設定）で作られた参考図面の .jww の見出し（色・線幅・線種・文字・レイヤグループ名と縮尺）を
05_設定/参考_環境.json に写しておき、図面を作る時に apply_env(d) で入れる。
 - 色1＝水色(0,192,192)、色6＝青(0,0,255) など、JWFの色もこの見出しに入っている
 - 図面ごとの縮尺・グループ0とレイヤの名前は、作る側（make_plan など）が後から決める

作り直し：python jww_env.py  （参考図面の .jww から 参考_環境.json を作る）
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_JSON = ROOT / "05_設定" / "参考_環境.json"
KEYS = ("palette", "pens", "line_types", "text_presets", "layer_groups")


def build(src=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ezjww
    src = src or ROOT / "04_参考" / "05 参考図面　平面図1階.jww"
    h = ezjww.read_document(str(src))["header"]
    ENV_JSON.write_text(json.dumps({k: h[k] for k in KEYS}, ensure_ascii=False), encoding="utf-8")
    return ENV_JSON


def apply_env(d):
    """ezjww.new() の図面 d に環境設定を入れる（なければ何もしない）。"""
    if not ENV_JSON.exists():
        return d
    env = json.loads(ENV_JSON.read_text(encoding="utf-8"))
    for k in KEYS:
        if k in env:
            d.options[k] = copy.deepcopy(env[k])
    return d


if __name__ == "__main__":
    print(build())
