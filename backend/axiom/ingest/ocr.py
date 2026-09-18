"""Optional Docling-backed OCR parser (Wave 4, D1).

Docling is an OPTIONAL extra: ``pip install 'axiom-adaptive-rag[ocr]'``.
Everything in this module imports docling lazily so the rest of the codebase
— and every CI job that does not opt into OCR — never pays for the
dependency. When OCR is requested but the extra is absent, the failure is
VISIBLE with an install hint (the in-tree precedent is pdfplumber's own
ImportError with hint in loader.py), never a silent empty-text result.

License posture: Docling is MIT and fits this MIT repository. AGPL-licensed
PDF tooling (pymupdf/pymupdf4llm) must not appear here or in any fixture
generator.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

OCR_INSTALL_HINT = (
    "OCR requires the Docling extra. Install it with: "
    "pip install 'axiom-adaptive-rag[ocr]'"
)


class OcrExtraMissingError(ImportError):
    """OCR was requested but the optional Docling extra is not installed."""


class OcrConversionError(RuntimeError):
    """Docling failed to convert a document (fail-visible parse failure)."""


_ocr_available: bool | None = None


def is_ocr_available() -> bool:
    """True when the docling extra imports cleanly (cached probe)."""
    global _ocr_available
    if _ocr_available is None:
        try:
            import docling  # noqa: F401
            _ocr_available = True
        except Exception as exc:  # ImportError and any broken extra install
            logger.info("Docling OCR extra not available: %s", exc)
            _ocr_available = False
    return _ocr_available


def reset_ocr_availability() -> None:
    """Drop the cached availability probe (tests install/uninstall the extra)."""
    global _ocr_available
    _ocr_available = None


def _require_docling() -> Any:
    """Import docling or raise the fail-visible install-hint error."""
    try:
        import docling.document_converter  # noqa: F401
        return docling
    except Exception as exc:
        raise OcrExtraMissingError(
            f"OCR was requested but Docling is not installed. {OCR_INSTALL_HINT}"
        ) from exc


def parse_pdf_ocr(file_path: str) -> List[Dict]:
    """OCR a PDF with Docling and return page dicts with structured blocks.

    Returns the same page contract as ``DocumentChunker.load_pdf`` plus a
    ``blocks`` list per page: table blocks carry their markdown serialization
    (kept atomic downstream by the chunker), prose blocks carry plain text.

    Raises:
        OcrExtraMissingError: docling not installed (message carries the
            install hint).
        OcrConversionError: the conversion itself failed.
    """
    docling = _require_docling()
    try:
        converter = docling.document_converter.DocumentConverter()
        result = converter.convert(file_path)
        return _document_to_pages(result.document)
    except OcrExtraMissingError:
        raise
    except Exception as exc:
        raise OcrConversionError(
            f"OCR conversion failed for a document (page extraction error: "
            f"{type(exc).__name__})."
        ) from exc


def _document_to_pages(document: Any) -> List[Dict]:
    """Group a DoclingDocument's items into {page_num, text, blocks} page dicts.

    Text items render as prose blocks; table items render as markdown table
    blocks (their ``export_to_markdown`` serialization), which the chunker
    keeps atomic. Pictures contribute nothing textual.
    """
    from docling_core.types.doc.items import TableItem
    from docling_core.types.doc.labels import DocItemLabel

    blocks_by_page: Dict[int, List[Dict]] = {}

    for item, _level in document.iterate_items():
        prov = getattr(item, "prov", None) or []
        page_no = int(prov[0].page_no) if prov else 0

        if isinstance(item, TableItem):
            markdown = item.export_to_markdown(document)
            if markdown and markdown.strip():
                blocks_by_page.setdefault(page_no, []).append(
                    {"kind": "table", "text": markdown}
                )
            continue

        text = getattr(item, "text", None)
        if not text or not str(text).strip():
            continue
        # Captions/page-headers stay prose; the table itself is the atomic unit.
        if getattr(item, "label", None) == DocItemLabel.PICTURE:
            continue
        blocks_by_page.setdefault(page_no, []).append(
            {"kind": "prose", "text": str(text)}
        )

    pages: List[Dict] = []
    for page_num in sorted(blocks_by_page):
        blocks = blocks_by_page[page_num]
        pages.append(
            {
                "page_num": page_num,
                "text": "\n".join(b["text"] for b in blocks),
                "blocks": blocks,
            }
        )
    return pages
