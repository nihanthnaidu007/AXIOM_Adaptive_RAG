"""Shared document extraction seam (Wave 4).

The HTTP ingest endpoint and the connectors must not each grow their own
parse tree: digital PDFs go through pdfplumber, scanned PDFs go through the
OCR adapter when OCR is enabled, and text formats decode as UTF-8. Every
caller receives the same ParseOutcome with page-coverage stats, so an
"empty but successful" ingest can never masquerade as a good one (the
scanned-page silent drop this wave removes).
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List

from axiom.ingest import ocr as ocr_module
from axiom.ingest.loader import DocumentChunker


class ScannedPdfNotSupportedError(ValueError):
    """A scanned PDF was ingested with OCR disabled — fail visible, never silent."""


@dataclass
class ParseOutcome:
    """Result of parsing one document's bytes.

    parse_confidence is the fraction of pages that yielded any text
    (1.0 when the document has no page structure). Callers surface it —
    a parse below 1.0 means pages were dropped somewhere in parsing.
    """

    pages: List[Dict] = field(default_factory=list)
    ocr_used: bool = False
    scanned: bool = False
    parse_confidence: float = 1.0


def _coverage(pages: List[Dict]) -> float:
    if not pages:
        return 0.0
    with_text = sum(1 for p in pages if (p.get("text") or "").strip())
    return with_text / len(pages)


def _parse_pdf_bytes(content: bytes, *, ocr_enabled: bool) -> ParseOutcome:
    """Digital-PDF first (pdfplumber); scanned PDFs route to the OCR adapter.

    Scanned detection: every page came back without a text layer. With OCR
    enabled (and the extra installed) the document re-parses through Docling;
    without OCR the error is explicit and operator-actionable.
    """
    chunker = DocumentChunker()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        pages = chunker.load_pdf(tmp_path)
        scanned = bool(pages) and all(not (p.get("text") or "").strip() for p in pages)
        if not scanned:
            return ParseOutcome(
                pages=pages,
                ocr_used=False,
                scanned=False,
                parse_confidence=_coverage(pages),
            )

        if not ocr_enabled:
            raise ScannedPdfNotSupportedError(
                "Scanned PDF detected (no extractable text layer) but OCR is "
                "disabled. Set OCR_ENABLED=true and install the OCR extra "
                "(pip install 'axiom-adaptive-rag[ocr]') to index scanned "
                "documents."
            )
        ocr_pages = ocr_module.parse_pdf_ocr(tmp_path)
        return ParseOutcome(
            pages=ocr_pages,
            ocr_used=True,
            scanned=True,
            parse_confidence=_coverage(ocr_pages),
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _parse_text_bytes(content: bytes) -> ParseOutcome:
    """txt/md path: UTF-8 decode (lenient, matching the historical endpoint)."""
    text = content.decode("utf-8", errors="ignore")
    pages = DocumentChunker().load_text(text, source="")
    return ParseOutcome(pages=pages, ocr_used=False, scanned=False, parse_confidence=_coverage(pages))


def parse_document(content: bytes, ext: str, *, ocr_enabled: bool = False) -> ParseOutcome:
    """Parse document bytes by extension. The one parse path for uploads AND connectors.

    Raises:
        ScannedPdfNotSupportedError: scanned PDF with OCR disabled.
        axiom.ingest.ocr.OcrExtraMissingError: OCR enabled but the extra is
            absent (message carries the install hint).
    """
    if ext == ".pdf":
        return _parse_pdf_bytes(content, ocr_enabled=ocr_enabled)
    return _parse_text_bytes(content)
