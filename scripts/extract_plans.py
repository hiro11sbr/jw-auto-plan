"""Extract embedded raster images from the presentation PDF (page, size, placement)."""
import sys
from pathlib import Path

import pymupdf as fitz

sys.stdout.reconfigure(encoding="utf-8")
pdf, outdir = Path(sys.argv[1]), Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
doc = fitz.open(pdf)
for pno, page in enumerate(doc, 1):
    for img in page.get_images(full=True):
        xref, w, h = img[0], img[2], img[3]
        # small images (logos, title strips) are skipped by area: a narrow house's plan can be
        # under 1000 px wide (プラン5 2F: 931 x 2077)
        if w * h < 1_000_000 or min(w, h) < 600:
            continue
        rects = page.get_image_rects(xref)
        r = rects[0] if rects else None
        dpi = w / (r.width / 72) if r else 0
        name = f"p{pno}_x{xref}_{w}x{h}.png"
        fitz.Pixmap(doc, xref).save(outdir / name)
        print(name, "rect(pt)=", r, f"dpi={dpi:.1f}")
