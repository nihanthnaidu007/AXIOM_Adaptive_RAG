"""AXIOM Evaluation Benchmark — 30-query dataset across 6 categories."""

from typing import List, Dict

BENCHMARK_QUERIES: List[Dict] = [
    # ── FACTUAL (strategy: bm25) ──────────────────────────────────────────
    {
        "query": "What is the BM25 Okapi term frequency formula",
        "expected_strategy": "bm25",
        "category": "FACTUAL",
        "ground_truth_keywords": ["term frequency", "inverse document frequency", "saturation", "k1", "BM25"],
        "should_trigger_correction": False,
    },
    {
        "query": "Define cosine similarity in vector space retrieval",
        "expected_strategy": "bm25",
        "category": "FACTUAL",
        "ground_truth_keywords": ["cosine", "dot product", "magnitude", "similarity", "vectors"],
        "should_trigger_correction": False,
    },
    {
        "query": "What does IVFFlat stand for in pgvector",
        "expected_strategy": "bm25",
        "category": "FACTUAL",
        "ground_truth_keywords": ["inverted file", "flat", "index", "pgvector", "approximate"],
        "should_trigger_correction": False,
    },
    {
        "query": "What is Reciprocal Rank Fusion",
        "expected_strategy": "bm25",
        "category": "FACTUAL",
        "ground_truth_keywords": ["reciprocal", "rank", "fusion", "score", "hybrid"],
        "should_trigger_correction": False,
    },
    {
        "query": "What is the default k value used in RRF scoring",
        "expected_strategy": "bm25",
        "category": "FACTUAL",
        "ground_truth_keywords": ["k", "60", "constant", "RRF", "rank"],
        "should_trigger_correction": False,
    },
    # ── ABSTRACT (strategy: vector) ───────────────────────────────────────
    {
        "query": "Explain why dense embeddings capture semantic meaning better than sparse representations",
        "expected_strategy": "vector",
        "category": "ABSTRACT",
        "ground_truth_keywords": ["dense", "semantic", "embedding", "sparse", "representation"],
        "should_trigger_correction": False,
    },
    {
        "query": "How does attention mechanism relate to information retrieval",
        "expected_strategy": "vector",
        "category": "ABSTRACT",
        "ground_truth_keywords": ["attention", "relevance", "query", "key", "value"],
        "should_trigger_correction": False,
    },
    {
        "query": "Why does hybrid retrieval outperform single-strategy search",
        "expected_strategy": "vector",
        "category": "ABSTRACT",
        "ground_truth_keywords": ["hybrid", "complementary", "semantic", "lexical", "precision"],
        "should_trigger_correction": False,
    },
    {
        "query": "What is the conceptual difference between bi-encoder and cross-encoder models",
        "expected_strategy": "vector",
        "category": "ABSTRACT",
        "ground_truth_keywords": ["bi-encoder", "cross-encoder", "independent", "joint", "reranking"],
        "should_trigger_correction": False,
    },
    {
        "query": "How do vector databases enable semantic search at scale",
        "expected_strategy": "vector",
        "category": "ABSTRACT",
        "ground_truth_keywords": ["vector", "index", "approximate", "nearest neighbor", "scale"],
        "should_trigger_correction": False,
    },
    # ── TIME_SENSITIVE (strategy: hybrid) ─────────────────────────────────
    {
        "query": "What are the latest improvements in transformer architecture",
        "expected_strategy": "hybrid",
        "category": "TIME_SENSITIVE",
        "ground_truth_keywords": ["transformer", "architecture", "improvement", "attention", "efficiency"],
        "should_trigger_correction": False,
    },
    {
        "query": "Current best practices for RAG pipeline evaluation",
        "expected_strategy": "hybrid",
        "category": "TIME_SENSITIVE",
        "ground_truth_keywords": ["RAG", "evaluation", "faithfulness", "relevancy", "metrics"],
        "should_trigger_correction": False,
    },
    {
        "query": "Recent advances in embedding model performance",
        "expected_strategy": "hybrid",
        "category": "TIME_SENSITIVE",
        "ground_truth_keywords": ["embedding", "model", "performance", "benchmark", "improvement"],
        "should_trigger_correction": False,
    },
    {
        "query": "Latest techniques for reducing hallucination in LLMs",
        "expected_strategy": "hybrid",
        "category": "TIME_SENSITIVE",
        "ground_truth_keywords": ["hallucination", "reduction", "grounding", "faithfulness", "LLM"],
        "should_trigger_correction": False,
    },
    {
        "query": "What are modern approaches to document chunking strategies",
        "expected_strategy": "hybrid",
        "category": "TIME_SENSITIVE",
        "ground_truth_keywords": ["chunking", "document", "strategy", "overlap", "semantic"],
        "should_trigger_correction": False,
    },
    # ── MULTI_HOP (strategy: hybrid) ─────────────────────────────────────
    {
        "query": "How does BM25 scoring work and why does it complement semantic search in hybrid retrieval",
        "expected_strategy": "hybrid",
        "category": "MULTI_HOP",
        "ground_truth_keywords": ["BM25", "semantic", "hybrid", "lexical", "complementary"],
        "should_trigger_correction": False,
    },
    {
        "query": "What is cross-encoder reranking and how does it differ from the initial retrieval step",
        "expected_strategy": "hybrid",
        "category": "MULTI_HOP",
        "ground_truth_keywords": ["cross-encoder", "reranking", "retrieval", "relevance", "score"],
        "should_trigger_correction": False,
    },
    {
        "query": "How do embeddings work and what role do they play in the RAGAS evaluation framework",
        "expected_strategy": "hybrid",
        "category": "MULTI_HOP",
        "ground_truth_keywords": ["embedding", "RAGAS", "evaluation", "vector", "faithfulness"],
        "should_trigger_correction": False,
    },
    {
        "query": "What is hallucination in LLMs and how does faithfulness scoring detect it",
        "expected_strategy": "hybrid",
        "category": "MULTI_HOP",
        "ground_truth_keywords": ["hallucination", "faithfulness", "scoring", "grounded", "LLM"],
        "should_trigger_correction": False,
    },
    {
        "query": "How does pgvector store embeddings and what index type does it use for similarity search",
        "expected_strategy": "hybrid",
        "category": "MULTI_HOP",
        "ground_truth_keywords": ["pgvector", "embedding", "index", "IVFFlat", "similarity"],
        "should_trigger_correction": False,
    },
    # ── STRESS_CORRECTION (should_trigger_correction: True) ───────────────
    {
        "query": "Tell me everything about the system",
        "expected_strategy": "hybrid",
        "category": "STRESS_CORRECTION",
        "ground_truth_keywords": ["system", "retrieval", "pipeline"],
        "should_trigger_correction": True,
    },
    {
        "query": "How good is the retrieval",
        "expected_strategy": "hybrid",
        "category": "STRESS_CORRECTION",
        "ground_truth_keywords": ["retrieval", "quality", "evaluation"],
        "should_trigger_correction": True,
    },
    {
        "query": "What makes this better than other approaches",
        "expected_strategy": "hybrid",
        "category": "STRESS_CORRECTION",
        "ground_truth_keywords": ["approach", "advantage", "comparison"],
        "should_trigger_correction": True,
    },
    {
        "query": "Summarize all the important things",
        "expected_strategy": "hybrid",
        "category": "STRESS_CORRECTION",
        "ground_truth_keywords": ["summary", "important", "key"],
        "should_trigger_correction": True,
    },
    {
        "query": "What should I know about this",
        "expected_strategy": "hybrid",
        "category": "STRESS_CORRECTION",
        "ground_truth_keywords": ["know", "overview", "information"],
        "should_trigger_correction": True,
    },
    # ── EDGE_CASES ────────────────────────────────────────────────────────
    {
        "query": "RAG",
        "expected_strategy": "bm25",
        "category": "EDGE_CASES",
        "ground_truth_keywords": ["retrieval", "augmented", "generation"],
        "should_trigger_correction": False,
    },
    {
        "query": "What is the meaning of life in the context of AI retrieval systems",
        "expected_strategy": "vector",
        "category": "EDGE_CASES",
        "ground_truth_keywords": ["retrieval", "AI", "purpose"],
        "should_trigger_correction": True,
    },
    {
        "query": "Compare BM25 vs vector search vs hybrid retrieval vs cross-encoder reranking vs semantic cache",
        "expected_strategy": "hybrid",
        "category": "EDGE_CASES",
        "ground_truth_keywords": ["BM25", "vector", "hybrid", "reranking", "cache"],
        "should_trigger_correction": False,
    },
    {
        "query": "If I have a question that requires information not in the knowledge base what happens",
        "expected_strategy": "hybrid",
        "category": "EDGE_CASES",
        "ground_truth_keywords": ["knowledge base", "missing", "retrieval", "fallback"],
        "should_trigger_correction": True,
    },
    {
        "query": "What is AXIOM",
        "expected_strategy": "bm25",
        "category": "EDGE_CASES",
        "ground_truth_keywords": ["AXIOM", "adaptive", "RAG", "pipeline"],
        "should_trigger_correction": False,
    },
]


