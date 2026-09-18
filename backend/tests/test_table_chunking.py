"""Wave 4 table-atomicity tests (D2).

A markdown-atomic table chunk must:
- bypass the sentence splitter (multi-row tables stay whole)
- be exempt from the min_chunk_size=100 discard (short tables survive)
- stay standalone — never merged with surrounding prose, no overlap bleed
- carry the page span of the page it came from

The chunker consumes blocks from structured parsers ({"kind": "table"}) —
exactly what the OCR adapter emits for markdown tables.
"""


from axiom.ingest.loader import TABLE_BLOCK, DocumentChunker


def _page(num, *blocks):
    return {"page_num": num, "text": "\n".join(b["text"] for b in blocks), "blocks": list(blocks)}


TABLE_TEXT = (
    "| Quarter | Revenue | Margin |\n"
    "|---------|---------|--------|\n"
    "| Q1 | $1.2M | 71% |\n"
    "| Q2 | $1.4M | 72% |\n"
    "| Q3 | $1.1M | 69% |\n"
    "| Q4 | $1.9M | 74% |"
)


def test_table_block_becomes_exactly_one_chunk():
    chunker = DocumentChunker()
    pages = [_page(1, {"kind": TABLE_BLOCK, "text": TABLE_TEXT})]

    chunks = chunker.chunk(pages, source="s3://bucket/t.csv")

    assert len(chunks) == 1
    body = chunks[0]["content"]
    assert body.startswith("| Quarter |")
    # Every row survived: the splitter never tore the table apart.
    for row in ("Q1", "Q2", "Q3", "Q4"):
        assert row in body


def test_short_table_survives_min_chunk_discard():
    """A ~20-token table would be discarded if it went through the prose path."""
    chunker = DocumentChunker()
    tiny_table = "| id | name |\n|----|------|\n| 1 | acme |"
    pages = [_page(1, {"kind": TABLE_BLOCK, "text": tiny_table})]

    chunks = chunker.chunk(pages, source="t")

    assert len(chunks) == 1, "table exemption from min_chunk_size is the D2 contract"


def test_short_prose_is_still_discarded():
    """The exemption is table-only — the prose discard keeps its teeth."""
    chunker = DocumentChunker()
    pages = [{"page_num": 1, "text": "Too short."}]

    chunks = chunker.chunk(pages, source="t")

    assert chunks == []


def test_table_never_merges_with_prose_or_overlaps():
    chunker = DocumentChunker()
    # Long enough to clear min_chunk_size=100 on its own (measured ~150
    # tokens): the separation assertions must hold for chunks that both
    # survive the discard.
    prose = (
        "The quarterly results were reviewed by the board in detail during the "
        "September meeting, with every divisional lead presenting variance notes. "
        "Several directors raised questions about margin pressure in the third "
        "quarter and the outlook for the following year, citing procurement "
        "delays and the weaker seasonal demand that shaped the first half. "
        "Management committed to a revised forecast cycle in November and pledged "
        "to bring a refreshed pricing analysis before the committee, alongside a "
        "review of the logistics contracts that inflated the cost of goods sold "
        "throughout the summer months. The audit committee also asked for a "
        "dedicated deep dive into working capital movements and requested that "
        "the treasury team model the impact of the new credit terms on cash "
        "conversion ahead of the year-end close."
    )
    pages = [_page(1, {"kind": "prose", "text": prose}, {"kind": TABLE_BLOCK, "text": TABLE_TEXT})]

    chunks = chunker.chunk(pages, source="t")

    assert len(chunks) == 2
    assert TABLE_TEXT in chunks[1]["content"]
    assert "board" not in chunks[1]["content"], "table must not absorb prose"
    assert "| Quarter |" not in chunks[0]["content"], "prose must not absorb the table"
    # No overlap bleed: the prose chunk does not repeat the table's first row.
    assert chunks[0]["content"].count("board") == 1


def test_table_carries_page_span():
    chunker = DocumentChunker()
    pages = [_page(3, {"kind": TABLE_BLOCK, "text": TABLE_TEXT})]

    chunks = chunker.chunk(pages, source="t")

    assert chunks[0]["page_start"] == 3
    assert chunks[0]["page_end"] == 3


def test_table_larger_than_chunk_size_stays_whole():
    """Oversized tables are never split by the window logic."""
    chunker = DocumentChunker()
    big_table = "| id | value |\n|----|-------|\n" + "\n".join(
        f"| {i} | {'x' * 40} |" for i in range(80)
    )
    pages = [_page(1, {"kind": TABLE_BLOCK, "text": big_table})]

    chunks = chunker.chunk(pages, source="t")

    assert len(chunks) == 1
    assert chunks[0]["content"].count("|") == big_table.count("|")


def test_two_tables_stay_separate():
    chunker = DocumentChunker()
    t1 = "| a | b |\n|---|---|\n| 1 | 2 |"
    t2 = "| c | d |\n|---|---|\n| 3 | 4 |"
    pages = [_page(1, {"kind": TABLE_BLOCK, "text": t1}, {"kind": TABLE_BLOCK, "text": t2})]

    chunks = chunker.chunk(pages, source="t")

    assert len(chunks) == 2
    assert chunks[0]["content"] == t1
    assert chunks[1]["content"] == t2


def test_empty_table_block_is_dropped_without_emitting_chunk():
    chunker = DocumentChunker()
    pages = [_page(1, {"kind": TABLE_BLOCK, "text": "   "})]

    assert chunker.chunk(pages, source="t") == []


def test_chunk_metadata_shape_for_tables():
    chunker = DocumentChunker()
    pages = [_page(2, {"kind": TABLE_BLOCK, "text": TABLE_TEXT})]

    (chunk,) = chunker.chunk(pages, source="doc.md", origin_type="s3")

    assert chunk["source"] == "doc.md"
    assert chunk["chunk_index"] == 0
    assert chunk["origin_type"] == "s3"
    assert chunk["token_count"] > 0
    assert chunk["chunk_id"]
