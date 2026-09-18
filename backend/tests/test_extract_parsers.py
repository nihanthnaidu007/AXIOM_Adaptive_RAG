"""Wave 4 parser-seam tests: parse_document on fixtures, OCR degradation.

Covers the D1 contract:
- text formats decode with full confidence
- the digital tabular fixture parses with a text layer (pdfplumber default)
- the scanned fixture raises ScannedPdfNotSupportedError when OCR is disabled
  (fail visible — the historical silent `extract_text() or ""` drop is gone)
- with OCR enabled but the extra absent, OcrExtraMissingError carries the
  install hint (mirrors loader.py:54-57 pdfplumber guidance)
- with OCR enabled and available, the OCR adapter path is taken (stubbed —
  real Docling conversion is exercised by the CI ocr-extra-smoke job and the
  MinIO integration job keeps the wire path honest)
"""

from pathlib import Path

import pytest

from axiom.ingest.extract import (
    ParseOutcome,
    ScannedPdfNotSupportedError,
    parse_document,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pdf"

TEXT_MARKDOWN = """# Quarterly Report

Revenue grew 12% quarter over quarter, driven by enterprise expansion.
Gross margin held steady at 71% for the full period.
"""


def test_text_document_parses_with_full_confidence():
    outcome = parse_document(TEXT_MARKDOWN.encode("utf-8"), ".md")

    assert isinstance(outcome, ParseOutcome)
    assert outcome.pages
    assert outcome.ocr_used is False
    assert outcome.scanned is False
    assert outcome.parse_confidence == 1.0
    assert "Revenue grew" in "\n".join(p["text"] for p in outcome.pages)


def test_digital_pdf_fixture_parses_with_text_layer():
    """The digital fixture must parse via pdfplumber with every page covered."""
    outcome = parse_document(FIXTURES.joinpath("quarterly_report_table.pdf").read_bytes(), ".pdf")

    assert outcome.ocr_used is False
    assert outcome.scanned is False
    assert outcome.parse_confidence == 1.0
    full_text = "\n".join(p["text"] for p in outcome.pages)
    assert "Revenue" in full_text


def test_scanned_pdf_with_ocr_disabled_fails_visibly():
    with pytest.raises(ScannedPdfNotSupportedError) as excinfo:
        parse_document(FIXTURES.joinpath("scanned_invoice.pdf").read_bytes(), ".pdf")

    # Operator-actionable: the error names the two fixes.
    assert "OCR_ENABLED" in str(excinfo.value)
    assert "pip install" in str(excinfo.value)


def test_scanned_pdf_with_ocr_enabled_but_extra_absent_names_install_hint(monkeypatch):
    from axiom.ingest import ocr as ocr_module

    monkeypatch.setattr(ocr_module, "is_ocr_available", lambda: False)
    with pytest.raises(ocr_module.OcrExtraMissingError) as excinfo:
        parse_document(
            FIXTURES.joinpath("scanned_invoice.pdf").read_bytes(), ".pdf", ocr_enabled=True
        )

    assert "[ocr]" in str(excinfo.value)
    assert "pip install" in str(excinfo.value)


def test_scanned_pdf_with_ocr_available_routes_to_ocr_adapter(monkeypatch):
    """With the extra present, the scanned document re-parses through Docling."""
    from axiom.ingest import ocr as ocr_module

    ocr_pages = [
        {"page_num": 1, "text": "INVOICE #4471", "blocks": [{"kind": "prose", "text": "INVOICE #4471"}]},
        {"page_num": 2, "text": "TOTAL DUE $1,204.00"},
    ]

    def _parse(file_path):
        return [dict(p) for p in ocr_pages]

    monkeypatch.setattr(ocr_module, "parse_pdf_ocr", _parse)

    outcome = parse_document(
        FIXTURES.joinpath("scanned_invoice.pdf").read_bytes(), ".pdf", ocr_enabled=True
    )

    assert outcome.ocr_used is True
    assert outcome.scanned is True
    assert outcome.parse_confidence == 1.0
    assert "INVOICE" in "\n".join(p["text"] for p in outcome.pages)
    assert "TOTAL DUE" in "\n".join(p["text"] for p in outcome.pages)


def test_parse_confidence_reflects_partial_page_coverage(monkeypatch):
    """A page structure where one page has text and one is empty reports 0.5 —
    never masquerades as a fully successful parse."""
    import axiom.ingest.extract as extract_module

    class _FakeChunker:
        def load_pdf(self, file_path):
            return [
                {"page_num": 1, "text": "real content"},
                {"page_num": 2, "text": ""},
            ]

    monkeypatch.setattr(extract_module, "DocumentChunker", _FakeChunker)
    outcome = parse_document(b"%PDF-1.4 fake", ".pdf")

    assert outcome.parse_confidence == pytest.approx(0.5)
    assert outcome.scanned is False  # some pages had text -> not routed to OCR


def test_unknown_extension_is_decoded_as_text():
    outcome = parse_document("plain content".encode("utf-8"), ".unknown")

    assert outcome.parse_confidence == 1.0
    assert outcome.pages
