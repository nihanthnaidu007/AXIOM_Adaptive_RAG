import pytest

from axiom.retrieval.bm25_index import BM25Index


def test_search_returns_empty_on_empty_index():
    idx = BM25Index()
    assert idx.search("any query", top_k=5) == []


@pytest.mark.asyncio
async def test_add_and_search_finds_relevant_chunk():
    idx = BM25Index()
    await idx.add_chunks([
        {
            "chunk_id": "c1",
            "content": "neural networks and deep learning",
            "source": "test.pdf",
            "chunk_index": 0,
            "token_count": 5,
        }
    ])
    result = idx.search("deep learning", top_k=1)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "c1"


@pytest.mark.asyncio
async def test_deduplication_prevents_double_index():
    idx = BM25Index()
    chunk = {
        "chunk_id": "dup",
        "content": "the same chunk content",
        "source": "test.pdf",
        "chunk_index": 0,
        "token_count": 4,
    }
    await idx.add_chunks([chunk])
    await idx.add_chunks([chunk])
    assert len(idx._documents) == 1


@pytest.mark.asyncio
async def test_search_top_k_respected():
    idx = BM25Index()
    chunks = [
        {
            "chunk_id": f"c{i}",
            "content": f"document number {i} about query topic",
            "source": "test.pdf",
            "chunk_index": i,
            "token_count": 6,
        }
        for i in range(5)
    ]
    await idx.add_chunks(chunks)
    result = idx.search("query", top_k=3)
    assert len(result) <= 3
