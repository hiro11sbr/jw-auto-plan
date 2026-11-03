"""Add plan pieces from one maker sheet to parts_spec.json (one entry per piece in a y band).

usage: add_parts_from_sheet.py <parts_dir> <sheet.jww (relative to parts_dir)> <y0> <y1>
       <maker> <series> <kind> <op> <note> [max_width_mm] [skip_x ...]

Finds the drawn pieces whose bbox lies within y0..y1 (paper mm on the sheet), names each by the
nearest W= text in the same row, and appends them. Entries already in the spec (same file and
bbox) are skipped, so running it twice does not duplicate parts.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_parts  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
parts_dir, rel = Path(sys.argv[1]), sys.argv[2]
y0, y1 = float(sys.argv[3]), float(sys.argv[4])
maker, series, kind, op, note = sys.argv[5:10]
max_w = float(sys.argv[10]) if len(sys.argv) > 10 else 1e9
skip_x = [float(v) for v in sys.argv[11:]]

spec_path = parts_dir / "parts_spec.json"
spec = json.load(open(spec_path, encoding="utf-8"))
have = {(p["file"], tuple(p["bbox"])) for p in spec["parts"] if "bbox" in p}  # ready parts have none
gs, texts = cluster_parts.groups(str(parts_dir / rel))
row = [(g, n) for g, n in gs if y0 <= g[1] and g[3] <= y1 and n >= 4]
for g, n in sorted(row, key=lambda t: t[0][0]):
    if any(abs(g[0] - x) < 0.5 for x in skip_x):
        continue
    cand = [t for t in texts if abs(t[2] - g[1]) < 13]
    if not cand:
        continue
    w = min(cand, key=lambda t: abs(t[1] - g[0]))[0]
    if w > max_w:
        continue
    pad = 0.3  # take in the whole piece (frame lines end on the bbox)
    bbox = [round(g[0] - pad, 2), round(g[1] - pad, 2), round(g[2] + pad, 2), round(g[3] + pad, 2)]
    key = (rel.replace("\\", "/"), tuple(bbox))
    if key in have:
        continue
    spec["parts"].append({"maker": maker, "series": series, "kind": kind, "op": op, "width": w,
                          "note": note, "file": key[0], "bbox": bbox})
    have.add(key)
    print(f"added {kind} W{w} bbox {bbox} ({n} pieces)")
json.dump(spec, open(spec_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
