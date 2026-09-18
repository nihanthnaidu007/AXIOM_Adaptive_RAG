"""AXIOM Document Loader - PDF and Text Chunking (page-aware, table-atomic).

Wave 4 changes, both spec-mandated:

1. Pages are first-class through chunking. A chunk records the page span it
   covers (``page_start``/``page_end``) instead of silently flattening
   ``page_num`` away — page-level citations are the OCR wave's traceability
   contract.
2. Table blocks stay markdown-atomic. They bypass the sentence splitter and
   are exempt from the ``min_chunk_size`` discard: a short table is real
   content, not a fragment to shred or drop.
"""

import hashlib
from typing import Dict, List

import nltk
import tiktoken

_tiktoken_enc = tiktoken.get_encoding("cl100k_base")

# Block kinds. "prose" rides the sentence-splitting sliding window; "table"
# blocks are emitted as standalone atomic chunks. Pages without explicit
# blocks are treated as a single prose block (pdfplumber / plain text).
PROSE_BLOCK = "prose"
TABLE_BLOCK = "table"


def _page_segments(page: Dict) -> List[Dict]:
    """Flatten one page dict into ordered {kind, text, page_num} segments.

    Pages may carry an explicit ``blocks`` list (structured parsers such as
    the Docling OCR adapter mark tables there); without one the page's whole
    text is a single prose segment.
    """
    page_num = page.get("page_num", 0)
    blocks = page.get("blocks")
    if blocks:
        return [
            {
                "kind": str(b.get("kind", PROSE_BLOCK)),
                "text": str(b.get("text", "")),
                "page_num": page_num,
            }
            for b in blocks
        ]
    return [{"kind": PROSE_BLOCK, "text": str(page.get("text", "")), "page_num": page_num}]


class DocumentChunker:
    """
    Document chunker using pdfplumber for PDFs and sliding window for text.
    Preserves paragraph and sentence boundaries; keeps table blocks atomic.
    """

    def __init__(
        self,
        chunk_size: int = 512,       # tokens (tiktoken cl100k_base)
        chunk_overlap: int = 64,      # overlap tokens between adjacent chunks
        min_chunk_size: int = 100     # discard prose chunks shorter than this (tokens)
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            min_chunk_size: Minimum prose chunk size to keep (tokens).
                Table blocks are exempt — a small table must survive chunking.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        try:
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
            self._use_nltk = True
        except Exception:
            self._use_nltk = False

    def load_pdf(self, file_path: str) -> List[Dict]:
        """
        Load PDF via pdfplumber (text-layer extraction, the digital-PDF default).

        Every page comes back — including text-empty ones — so callers can
        detect scanned documents and fail visibly instead of silently
        indexing a fraction of the file. Structured table recovery is the
        OCR adapter's job (axiom.ingest.ocr); this path emits prose only.

        Args:
            file_path: Path to PDF file

        Returns:
            List of {page_num, text} dicts; text may be "" for scanned pages
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required for PDF loading. Install with: pip install pdfplumber")

        pages = []

        with pdfplumber.open(file_path) as pdf:
            # Detect repeated headers/footers
            first_lines = []
            last_lines = []

            for page in pdf.pages:
                text = page.extract_text() or ""
                lines = text.split('\n')
                if lines:
                    first_lines.append(lines[0] if lines else "")
                    last_lines.append(lines[-1] if lines else "")

            # Find repeated headers/footers (appear on >50% of pages)
            header_threshold = len(pdf.pages) * 0.5
            repeated_headers = set()
            repeated_footers = set()

            for line in set(first_lines):
                if first_lines.count(line) > header_threshold:
                    repeated_headers.add(line)

            for line in set(last_lines):
                if last_lines.count(line) > header_threshold:
                    repeated_footers.add(line)

            # Extract text from each page, removing headers/footers
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                lines = text.split('\n')

                # Remove repeated headers/footers
                if lines and lines[0] in repeated_headers:
                    lines = lines[1:]
                if lines and lines[-1] in repeated_footers:
                    lines = lines[:-1]

                cleaned_text = '\n'.join(lines).strip()
                # Empty text is preserved: a scanned page must be visible to
                # the caller, never silently absorbed into a shorter page list.
                pages.append({
                    "page_num": page_num,
                    "text": cleaned_text
                })

        return pages

    def load_text(self, text: str, source: str) -> List[Dict]:
        """
        Load raw text string directly.

        Args:
            text: Raw text content
            source: Source identifier

        Returns:
            List of {page_num, text} dicts (single page)
        """
        return [{"page_num": 1, "text": text.strip()}]

    def chunk(self, pages: List[Dict], source: str = "unknown", origin_type: str = "upload") -> List[Dict]:
        """
        Sliding window chunking with overlap, respecting sentence boundaries.

        Table blocks (from structured parsers) bypass the sentence splitter
        and the min-chunk discard; each becomes exactly one atomic chunk.

        Args:
            pages: List of {page_num, text} dicts, optionally with a
                ``blocks`` list of {kind, text} entries
            source: Source identifier for the document
            origin_type: Provenance label carried on every chunk
                ("upload" | "s3" | "crawl")

        Returns:
            List of {chunk_id, source, content, chunk_index, token_count,
            page_start, page_end, origin_type} dicts
        """
        # Bind every chunk id to the exact document version: re-uploading
        # modified content yields fresh ids instead of colliding with stale
        # rows via ON CONFLICT DO NOTHING.
        full_text = "\n\n".join(page["text"] for page in pages)
        content_hash = hashlib.sha256(full_text.encode()).hexdigest()[:12]

        chunks: List[Dict] = []
        chunk_index = 0

        # Open prose window: list of (sentence, page_num), running token count.
        window: List[tuple] = []
        window_tokens = 0

        def flush_window() -> None:
            """Emit the open prose window (subject to the min-size discard)."""
            nonlocal chunk_index, window, window_tokens
            if not window:
                return
            chunk_text = ' '.join(text for text, _ in window)
            chunk_token_count = len(_tiktoken_enc.encode(chunk_text))
            if chunk_token_count >= self.min_chunk_size:
                page_start = min(p for _, p in window)
                page_end = max(p for _, p in window)
                chunk_id = self._generate_chunk_id(source, content_hash, chunk_index)
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": source,
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                    "token_count": chunk_token_count,
                    "page_start": page_start,
                    "page_end": page_end,
                    "origin_type": origin_type,
                })
                chunk_index += 1
            window = []
            window_tokens = 0

        def carry_overlap() -> None:
            """Restart the window with trailing sentences up to chunk_overlap."""
            nonlocal window, window_tokens
            overlap: List[tuple] = []
            overlap_tokens = 0
            for text, page_num in reversed(window):
                sent_tokens = len(_tiktoken_enc.encode(text))
                if overlap_tokens + sent_tokens <= self.chunk_overlap:
                    overlap.insert(0, (text, page_num))
                    overlap_tokens += sent_tokens
                else:
                    break
            window = overlap
            window_tokens = overlap_tokens

        for page in pages:
            for segment in _page_segments(page):
                if segment["kind"] == TABLE_BLOCK:
                    # Tables are atomic: flush the prose window (the window's
                    # own min-size rule still applies to it), then emit the
                    # table whole — never sentence-split, never discarded,
                    # never merged with surrounding prose.
                    flush_window()
                    table_text = segment["text"].strip()
                    if table_text:
                        table_tokens = len(_tiktoken_enc.encode(table_text))
                        chunk_id = self._generate_chunk_id(source, content_hash, chunk_index)
                        chunks.append({
                            "chunk_id": chunk_id,
                            "source": source,
                            "content": table_text,
                            "chunk_index": chunk_index,
                            "token_count": table_tokens,
                            "page_start": segment["page_num"],
                            "page_end": segment["page_num"],
                            "origin_type": origin_type,
                        })
                        chunk_index += 1
                    continue

                for sentence in self._split_into_sentences(segment["text"]):
                    sentence_tokens = len(_tiktoken_enc.encode(sentence))

                    # If adding this sentence exceeds chunk size, save current chunk
                    if window_tokens + sentence_tokens > self.chunk_size and window:
                        flush_window()
                        carry_overlap()

                    window.append((sentence, segment["page_num"]))
                    window_tokens += sentence_tokens

        # Don't forget the last chunk
        flush_window()

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK sent_tokenize."""
        if self._use_nltk:
            sentences = nltk.sent_tokenize(text)
        else:
            sentences = text.split('. ')
        return [s.strip() for s in sentences if s.strip()]

    def _generate_chunk_id(self, source: str, content_hash: str, chunk_index: int) -> str:
        """Generate a unique chunk ID from source, file content hash, and index."""
        content = f"{source}:{content_hash}:{chunk_index}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()
        return hash_digest[:12]
