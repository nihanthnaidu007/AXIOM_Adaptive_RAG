#!/usr/bin/env python3
"""Generate the Wave 4 PDF fixtures (one-time, deterministic content).

Run from this directory:  python generate_fixtures.py

- scanned_invoice.pdf  — image-only PDF (a "scan"): rendered text in a raster
  image, no text layer. pdfplumber extracts "" from it; Docling OCR reads it.
- quarterly_report_table.pdf — digital PDF with a text layer (baseline14
  Helvetica, no embedding) containing prose plus a row-oriented table.

Licensing: Pillow (HPND/MIT-CMU) and the Python standard library only —
no AGPL PDF tooling touches fixture generation.
"""

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

SCAN_LINES = [
    "ACME SUPPLIES - INVOICE #2026-0417",
    "Date: 14 March 2026",
    "Bill to: Northwind Traders Ltd",
    "",
    "Item: Steel brackets (x240)      $4,180.00",
    "Item: Hex bolts grade 8.8 (x1200)  $960.00",
    "Item: Galvanised washers (x800)    $214.00",
    "",
    "Subtotal:                         $5,354.00",
    "Tax (8.5%):                         $455.09",
    "Total due:                        $5,809.09",
]

TABLE_LINES = [
    "Quarterly Revenue Report - FY2026",
    "",
    "Quarter   Revenue       Gross Margin   New Logos",
    "Q1 2026   $1,204,000    41.2%          14",
    "Q2 2026   $1,387,500    42.8%          18",
    "Q3 2026   $1,502,300    43.1%          21",
    "Q4 2026   $1,644,900    43.9%          25",
    "",
    "Revenue grew every quarter of FY2026 while gross margin expanded",
    "roughly 270 basis points, driven mostly by support-tier mix rather",
    "than list-price changes. New logo momentum was strongest in EMEA.",
]

FILLER = (
    "The figures above are unaudited management accounts prepared on an "
    "accrual basis and rounded to the nearest hundred dollars. Prior-year "
    "comparatives have been restated for the divested retail line. "
)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_digital_pdf(path: str) -> None:
    """Single-page digital PDF: text-layer lines via Tj/T* operators."""
    lines = TABLE_LINES + [""] + [FILLER] * 80  # pad toward tens of KB
    stream_lines = ["BT", "/F1 11 Tf", "13 TL", "72 720 Td"]
    for i, line in enumerate(lines):
        if i:
            stream_lines.append("T*")
        stream_lines.append(f"({_pdf_escape(line)}) Tj")
    stream_lines.append("ET")
    stream = "\n".join(stream_lines).encode("latin-1", "replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj_num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (obj_num, body)
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, xref_pos)
    )
    with open(path, "wb") as fh:
        fh.write(bytes(out))


def build_scanned_pdf(path: str) -> None:
    """Image-only PDF: readable raster text, no text layer for pdfplumber."""
    scale = 2
    width, line_height = 1240, 44
    img = Image.new("L", (width, 200 + line_height * len(SCAN_LINES)), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    y = 60
    for line in SCAN_LINES:
        draw.text((80, y), line, fill=20, font=font)
        y += line_height
    # Light speckle noise so the page reads as a real scan to the eye.
    for x, y2 in [(1000, 120), (300, 300), (700, 520), (200, 640), (900, 760)]:
        draw.ellipse([x, y2, x + 6, y2 + 6], fill=210)
    img = img.resize((width // scale, img.height // scale)).convert("L")
    img.save(path, "PDF", resolution=200.0)


if __name__ == "__main__":
    scan_path = os.path.join(HERE, "scanned_invoice.pdf")
    table_path = os.path.join(HERE, "quarterly_report_table.pdf")
    build_scanned_pdf(scan_path)
    build_digital_pdf(table_path)
    for p in (scan_path, table_path):
        print(p, os.path.getsize(p), "bytes")
