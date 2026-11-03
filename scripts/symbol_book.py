"""記号見本帳: drawing conventions other than fittings (garage shutter, 玄関収納, ...), cut out
of the sample plans with their name and how they are handled.

usage: symbol_book.py <記号見本帳.json> <out.png>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jp_text  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
CELL = 380
COLS = 4


def main():
    spec_path, out = sys.argv[1:3]
    root = Path(spec_path).resolve().parent.parent
    items = json.load(open(spec_path, encoding="utf-8"))["items"]
    cells = []
    for n, it in enumerate(items, 1):
        img = cv2.imdecode(np.fromfile(str(root / it["image"]), np.uint8), cv2.IMREAD_COLOR)
        x0, y0, x1, y1 = it["px"]
        crop = img[y0:y1, x0:x1]
        s = min((CELL - 70) / crop.shape[0], (CELL - 10) / crop.shape[1])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        cell = np.full((CELL, CELL, 3), 255, np.uint8)
        cell[64:64 + crop.shape[0], 4:4 + crop.shape[1]] = crop
        cell = jp_text.put(cell, [(f"{n} {it['name']}（{it['from']}）", (4, 4), (0, 0, 0), 15),
                                  (it["how"], (4, 26), (120, 60, 0), 12)])
        cv2.rectangle(cell, (0, 0), (CELL - 1, CELL - 1), (200, 200, 200), 1)
        cells.append(cell)
    rows = (len(cells) + COLS - 1) // COLS
    sheet = np.full((rows * CELL, COLS * CELL, 3), 255, np.uint8)
    for k, c in enumerate(cells):
        r, cc = divmod(k, COLS)
        sheet[r * CELL:(r + 1) * CELL, cc * CELL:(cc + 1) * CELL] = c
    cv2.imencode(".png", sheet)[1].tofile(out)
    print(out, len(cells))


if __name__ == "__main__":
    main()
