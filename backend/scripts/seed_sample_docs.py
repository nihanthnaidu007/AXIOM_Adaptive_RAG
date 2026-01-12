"""Seed AXIOM indexes with sample RAG / ML / NLP chunks for development."""

import asyncio
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axiom.retrieval.vector_store import vector_store
from axiom.retrieval.bm25_index import bm25_index
from axiom.ingest.indexer import DualIndexer

SAMPLE_CHUNKS = [
    {
        "source": "rag_overview.pdf",
        "content": (
            "Retrieval-Augmented Generation (RAG) combines a retriever module with a "
            "generative language model. The retriever fetches relevant passages from a "
            "document store, which the generator conditions on to produce grounded answers. "
            "This two-stage pipeline reduces hallucination by anchoring generation in evidence."
        ),
    },
    {
        "source": "vector_databases.pdf",
        "content": (
            "Vector databases such as pgvector, Pinecone, and Weaviate store high-dimensional "
            "embeddings and support approximate nearest-neighbor (ANN) search. pgvector adds "
            "a vector column type and cosine / inner-product distance operators to PostgreSQL, "
            "enabling semantic search without a separate database."
        ),
    },
    {
        "source": "llm_fundamentals.pdf",
        "content": (
            "Large Language Models (LLMs) like GPT-4 and Claude are autoregressive transformers "
            "trained on internet-scale text. They predict the next token given a context window, "
            "achieving state-of-the-art performance on question answering, summarization, and "
            "code generation tasks."
        ),
    },
    {
        "source": "embeddings_guide.pdf",
        "content": (
            "Text embeddings are dense vector representations that capture semantic meaning. "
            "Models like OpenAI text-embedding-3-small produce 1536-dimensional vectors where "
            "cosine similarity correlates with semantic relatedness, enabling similarity search "
            "across large document collections."
        ),
    },
    {
        "source": "transformer_architecture.pdf",
        "content": (
            "The Transformer architecture introduced in 'Attention Is All You Need' (2017) "
            "replaces recurrence with multi-head self-attention. Each layer computes scaled "
            "dot-product attention over queries, keys, and values, enabling parallel processing "
            "of sequence elements and capturing long-range dependencies."
        ),
    },
    {
        "source": "ragas_evaluation.pdf",
        "content": (
            "RAGAS is a framework for evaluating RAG pipelines. It measures faithfulness "
            "(is the answer supported by context?), answer relevancy (does it address the "
            "question?), and context precision/recall. These metrics help detect hallucination "
            "and ensure generation quality."
        ),
    },
    {
        "source": "bm25_algorithm.pdf",
        "content": (
            "BM25 (Okapi BM25) is a probabilistic ranking function based on term frequency "
            "and inverse document frequency. The BM25 scoring formula is: "
            "score(D,Q) = sum( IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl)) ) "
            "where k1 typically equals 1.2 and b equals 0.75."
        ),
    },
    {
        "source": "semantic_search.pdf",
        "content": (
            "Semantic search goes beyond keyword matching by understanding query intent. "
            "It encodes both queries and documents into a shared embedding space, then ranks "
            "results by vector similarity. This enables retrieval of conceptually related "
            "content even when surface-level terms differ."
        ),
    },
    {
        "source": "hallucination_detection.pdf",
        "content": (
            "Hallucination in LLMs occurs when the model generates factual-sounding but "
            "unsupported claims. Detection techniques include entailment checking against "
            "source documents, self-consistency sampling, and faithfulness scoring. RAG "
            "systems reduce hallucination by grounding generation in retrieved evidence."
        ),
    },
    {
        "source": "cross_encoder_reranking.pdf",
        "content": (
            "Cross-encoder rerankers like ms-marco-MiniLM-L-6-v2 jointly encode query-passage "
            "pairs to produce a relevance score. Unlike bi-encoders that embed query and passage "
            "independently, cross-encoders capture fine-grained interactions but are more "
            "expensive — hence used as a second-stage reranker on a smaller candidate set."
        ),
    },
    {
        "source": "hybrid_retrieval.pdf",
        "content": (
            "Hybrid retrieval combines sparse (BM25) and dense (vector) search strategies. "
            "BM25 excels on exact keyword matches while vector search captures semantic meaning. "
            "Reciprocal Rank Fusion (RRF) merges the two ranked lists using the formula "
            "RRF(d) = sum(1/(k+rank)) with k=60, producing a single ranking."
        ),
    },
    {
        "source": "chunking_strategies.pdf",
        "content": (
            "Document chunking strategies include fixed-size (e.g. 512 tokens), sentence-based, "
            "and recursive splitting. Overlap between chunks preserves cross-boundary context. "
            "The optimal chunk size depends on the embedding model's context window and the "
            "downstream retrieval granularity requirements."
        ),
    },
    {
        "source": "prompt_engineering.pdf",
        "content": (
            "Prompt engineering for RAG systems involves structuring the context window with "
            "retrieved passages, a system instruction, and the user query. Techniques like "
            "chain-of-thought prompting and few-shot examples improve answer quality and "
            "reduce hallucination in generation."
        ),
    },
    {
        "source": "transformer_2024.pdf",
        "content": (
            "The latest transformer architecture improvements in 2024 include mixture-of-experts "
            "(MoE) routing, grouped query attention (GQA), and sliding window attention for "
            "extended context lengths. These innovations reduce inference cost while maintaining "
            "or improving model quality on standard benchmarks."
        ),
    },
    {
        "source": "evaluation_metrics.pdf",
        "content": (
            "Information retrieval evaluation metrics include Mean Reciprocal Rank (MRR), "
            "Normalized Discounted Cumulative Gain (nDCG), and Recall@k. For RAG pipelines, "
            "end-to-end metrics like RAGAS faithfulness and answer relevancy provide a more "
            "holistic assessment of the full retrieve-then-generate pipeline."
        ),
    },
    {
        "source": "pgvector_setup.pdf",
        "content": (
            "Setting up pgvector involves installing the extension in PostgreSQL, creating "
            "a table with a vector column, and building an IVFFlat or HNSW index. IVFFlat "
            "with 100 lists is suitable for up to ~1M vectors. For larger datasets, HNSW "
            "provides better recall at the cost of higher memory usage."
        ),
    },
]


def prepare_chunks() -> list[dict]:
    chunks = []
    for i, raw in enumerate(SAMPLE_CHUNKS):
        content = raw["content"]
        chunk_id = hashlib.sha256(f"{raw['source']}:{i}".encode()).hexdigest()[:12]
        chunks.append({
            "chunk_id": chunk_id,
            "source": raw["source"],
            "content": content,
            "chunk_index": i,
            "token_count": len(content.split()),
        })
    return chunks


async def main():
    chunks = prepare_chunks()
    print(f"Prepared {len(chunks)} sample chunks")

    connected = await vector_store.connect()
    if not connected:
        print("WARNING: pgvector not connected — seeding BM25 only")

    indexer = DualIndexer()
    result = await indexer.index_chunks(chunks)

    print(f"Seeded {result['chunk_count']} chunks")
    print(f"BM25: {result['bm25']}")
    print(f"Vector: {result['vector']}")
    if "rows_inserted" in result:
        print(f"pgvector rows inserted: {result['rows_inserted']}")
    if "vector_error" in result:
        print(f"Vector error: {result['vector_error']}")


if __name__ == "__main__":
    asyncio.run(main())
