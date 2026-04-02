# AXIOM: Adaptive Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

##Adaptive RAG pipeline with a self-correcting hallucination detection loop.

### Query Input & Document Upload Panel
![Query Input & Document Upload Panel](Images/Image_1.png)

### Pipeline Execution - Retrieval Signal & RAGAS Evaluation
![Pipeline Execution - Retrieval Signal & RAGAS Evaluation](Images/Image_2.png)

### Hallucination Correction Loop - Query Rewrite Iterations
![Hallucination Correction Loop - Query Rewrite Iterations](Images/Image_3.png)

### Final Answer Panel with Confidence Band
![Final Answer Panel with Confidence Band](Images/Image_4.png)

Submit a query. AXIOM classifies it, routes it to the right retrieval strategy, generates an answer, evaluates it for faithfulness against the retrieved context, and rewrites the query if the answer fails. This correction loop runs up to three times. If the answer passes, it is cached. If it fails all three attempts, the system surfaces the best available answer with a confidence rating.

AXIOM is built on a LangGraph cyclic graph with 12 nodes. The hallucination gate runs on every answer. Retrieval uses BM25, pgvector, or RRF hybrid fusion depending on query type. A cross-encoder reranker scores all candidates before generation. Everything is observable through LangSmith.

---

## What It Does

- 🔍 **Three Retrieval Strategies:** BM25 keyword search, pgvector semantic search, and RRF hybrid fusion. Query type determines which strategy fires.
- ♻️ **Self-Correcting Hallucination Loop:** Every generated answer is evaluated for faithfulness against retrieved chunks by an Ollama critic. If faithfulness < 0.75, the query is rewritten and the pipeline runs again, up to three iterations.
- 📊 **RAGAS Evaluation:** Three-dimensional scoring covering faithfulness (answer grounded in context), answer relevancy (answer addresses the question), and context groundedness (context contains the answer). Scored by llama3.2 running locally via Ollama.
- ⚡ **Redis Semantic Cache:** Answers that pass the hallucination gate are cached with their embedding. Identical or semantically similar future queries hit the cache directly. Cache hits are 30-50x faster than full pipeline runs.
- 🔀 **Multi-Hop Decomposition:** Complex queries are broken into sub-queries, each resolved independently, then synthesized into a single answer.
- 🔁 **Cross-Encoder Reranking:** Retrieved chunks are reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2` before generation. The top 5 most relevant chunks reach the LLM.
- 📡 **LangSmith Tracing:** Full trace tree per query covering every node, LLM call, retrieval step, and evaluation score.
- 📄 **Document Ingestion:** Upload PDFs, TXT, or Markdown files. Chunks are indexed to both BM25 and pgvector simultaneously using tiktoken token counting and NLTK sentence splitting.
- 🔒 **Rate Limiting and Validation:** 30 requests per minute per IP. Empty queries and queries over 2000 characters are rejected before any LLM call is made.

---

## Architecture

![AXIOM Architecture Diagram](axiom_architecture_diagram.svg)

---

## Pipeline Flow

```
classify_query -> check_cache -> route_retrieval -> [bm25 | vector | hybrid]
    -> decompose_query -> rerank_chunks -> generate_answer -> evaluate_answer
         ^                                                         |
         +-------------- rewrite_query <- (faith < 0.75) ---------+
                                                                   |
                                                    finalize_answer -> END
```

**Routing logic:**

```
1.  Query arrives -> classify_query assigns type: FACTUAL, ABSTRACT, TIME_SENSITIVE, MULTI_HOP
2.  check_cache -> if semantic similarity > 0.85 with a cached query, return immediately
3.  route_retrieval -> FACTUAL routes to BM25, ABSTRACT to vector, TIME_SENSITIVE/MULTI_HOP to hybrid
4.  decompose_query -> for MULTI_HOP: split into sub-queries, run each through BM25, merge results
5.  rerank_chunks -> cross-encoder scores all retrieved chunks, top 5 pass to generation
6.  generate_answer -> Claude Sonnet generates answer from top 5 chunks
7.  evaluate_answer -> Ollama llama3.2 scores faithfulness, relevancy, groundedness
8.  if faithfulness < 0.75 AND correction_attempts < max_correction_attempts -> rewrite_query -> loop
9.  if faithfulness >= 0.75 OR correction_attempts exhausted -> finalize_answer
10. finalize_answer -> write to Redis cache if gate_passed=True -> return response
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph 0.2+, cyclic StateGraph, MemorySaver checkpointing |
| LLM | Claude Sonnet via Anthropic API |
| Retrieval (Keyword) | BM25 via rank_bm25, in-memory, hydrated from PostgreSQL on startup |
| Retrieval (Semantic) | pgvector with OpenAI text-embedding-3-small |
| Retrieval (Hybrid) | Reciprocal Rank Fusion (k=60) merging BM25 and vector rankings |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers |
| Evaluation | RAGAS metrics via Ollama llama3.2 running locally |
| Cache | Redis, cosine similarity semantic cache, 7-day TTL |
| Database | PostgreSQL with pgvector extension (Docker) |
| Backend API | FastAPI with async background tasks, rate limiting via slowapi |
| Frontend | React 18, Tailwind CSS |
| Observability | LangSmith, full trace tree, latency per node |
| Document Parsing | pdfplumber for PDFs, tiktoken for token counting, NLTK for sentence splitting |

---

## Prerequisites

- **Python 3.11+** - check with `python3 --version`
- **Node.js 18+** - check with `node --version`
- **Docker** - for PostgreSQL + pgvector and Redis
- **Ollama** - for local RAGAS evaluation. Install from [ollama.ai](https://ollama.ai), then pull the model:
  ```bash
  ollama pull llama3.2
  ```
- **Anthropic API key** - get one at [console.anthropic.com](https://console.anthropic.com)
- **OpenAI API key** - used for document embeddings at [platform.openai.com](https://platform.openai.com)
- **LangSmith API key** - optional, free tier at [smith.langchain.com](https://smith.langchain.com)

---

## Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/nihanthnaidu007/Axiom-Adaptive-RAG-main.git
cd Axiom-Adaptive-RAG-main
```

### Step 2: Start infrastructure

```bash
docker-compose up -d
```

This starts PostgreSQL with the pgvector extension on port 5432 and Redis on port 6379.

### Step 3: Set up the backend

```bash
cd backend
pip install -r requirements.txt
```

### Step 4: Create your environment file

```bash
cp .env.example .env
```

Open `backend/.env` and fill in your keys:

```
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith tracing (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=axiom-rag

# Infrastructure (defaults work with docker-compose)
POSTGRES_URL=postgresql+asyncpg://axiom:axiom_secret@localhost:5432/axiom_rag
REDIS_URL=redis://localhost:6379
OLLAMA_URL=http://localhost:11434
```

### Step 5: Start the backend

```bash
cd backend
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

You should see:

```
LangSmith tracing ENABLED, project: axiom-rag
pgvector connected, chunk_embeddings table ready
BM25 hydrated from pgvector, 0 chunks loaded
Redis semantic cache connected
Ollama critic connected, real RAGAS evaluation enabled
INFO: Uvicorn running on http://127.0.0.1:8000
```

### Step 6: Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000).

### Step 7: Upload documents

Use the upload panel in the UI to upload PDF files. The pipeline needs documents in the index before queries will return meaningful results. After uploading, the status bar shows the chunk count.

Seven reference documents are used in development and benchmarking:
- `Attention Is All You Need.pdf`
- `bm25_okapi.pdf`
- `dense_passage_retrieval.pdf`
- `pgvector_readme.pdf`
- `rag_lewis_2020.pdf`
- `ragas_evaluation.pdf`
- `sentence_bert.pdf`

---

## How to Use

### Running a query

1. Type a question in the query input at the top of the dashboard
2. The strategy auto-detection badge shows which retrieval path will be used
3. Click **RUN**

### What you will see

**Pipeline Strip** - Shows each of the 12 nodes firing in sequence. Nodes light up as they complete. The correction loop counter shows how many rewrites have occurred.

**Retrieval Signal panel** - Shows the top 5 reranked chunks with source filename, rerank score, and position delta showing how much the reranker moved each chunk up or down.

**Evaluation Signal panel** - Shows the three RAGAS scores and their history across correction iterations. The score history shows how faithfulness changed across rewrites.

**Correction Record** - If the hallucination gate fired, each iteration shows the rewrite reasoning and the new query that was attempted.

**Answer panel** - The final answer with confidence band: VERIFIED (>=85%), PROBABLE (>=65%), UNCERTAIN (>=45%), UNRELIABLE (<45%). Sources are listed with rerank scores.

---

## Confidence Bands

```
VERIFIED    -> faithfulness >= 0.85, passed gate on first or second attempt
PROBABLE    -> faithfulness >= 0.65
UNCERTAIN   -> faithfulness >= 0.45, likely needed correction
UNRELIABLE  -> faithfulness < 0.45, exhausted all corrections

Composite = faithfulness x 0.5 + relevancy x 0.3 + groundedness x 0.2
Correction penalty = composite - (0.10 x correction_attempts)
```

---

## Document Ingestion

AXIOM accepts PDF, TXT, and Markdown files through the upload panel.

**Chunking:** Documents are split into overlapping chunks using tiktoken for accurate token counting and NLTK sentence tokenizer for clean sentence boundaries. Chunk size and overlap are configurable in the environment.

**Dual indexing:** Every chunk is indexed to both BM25 (in-memory, rebuilt on startup) and pgvector (persistent). Both indexes are available immediately after upload.

**Supported formats:**
- PDF - pdfplumber extracts text page by page. Pages under 50 characters (blank or image-only) are skipped.
- TXT and Markdown - read directly, split by the same chunker.

---

## Retrieval Strategies

### BM25
Used for FACTUAL queries with specific terminology. BM25Okapi scores chunks against tokenized query terms. Returns top 20 by score, passes to reranker.

### Vector
Used for ABSTRACT queries requiring semantic matching. OpenAI `text-embedding-3-small` embeds the query. pgvector returns top 20 by cosine similarity.

### Hybrid (RRF)
Used for TIME_SENSITIVE and MULTI_HOP queries. BM25 and vector run in parallel via `asyncio.gather`. Results are merged using Reciprocal Rank Fusion with k=60. Combined ranking passed to reranker.

### Multi-hop decomposition
For complex queries, AXIOM breaks the query into sub-queries using Claude. Each sub-query runs through BM25 independently. Results are merged and synthesized into a single context before generation.

---

## Semantic Cache

Answers that pass the hallucination gate (faithfulness >= 0.75) are stored in Redis with their query embedding.

**Two-tier lookup:**
- Tier 1 (exact): normalized query string hashed and looked up directly. O(1).
- Tier 2 (semantic): cosine similarity computed over the 200 most recent cache entries. Returns if similarity > 0.85.

Cache entries expire after 7 days. Cache hits skip the entire retrieval and generation pipeline. Typical cache hit latency is 1-3 seconds vs 30-90 seconds for a full run.

---

## Evaluation

AXIOM uses three RAGAS metrics scored by Ollama llama3.2 running locally:

**Faithfulness** - Are the claims in the answer supported by the retrieved context? This is the primary hallucination gate metric. Threshold: 0.75.

**Answer Relevancy** - Does the answer actually address the question that was asked?

**Context Groundedness** - Does the retrieved context contain the information needed to answer the question?

When Ollama is unavailable, the system continues but marks all scores as `evaluation_mode: "mock"` and `evaluation_passed: false`. Mock mode queries will always run to the correction limit. The API response surfaces this clearly.

---

## LangSmith Tracing

Every query produces a full trace at [smith.langchain.com](https://smith.langchain.com).

The trace shows:
- `classify_query` - classification result and reasoning
- `check_cache` - cache hit/miss and similarity score if near-hit
- `route_retrieval` - strategy selected and why
- `retrieve_bm25` / `retrieve_vector` / `retrieve_hybrid` - chunk count, top score, latency
- `rerank_chunks` - pre/post rerank positions, reranker_mode (real or fallback)
- `generate_answer` - prompt tokens, completion tokens, latency
- `evaluate_answer` - all three RAGAS scores, evaluation_mode
- `rewrite_query` - rewrite reasoning, new query (appears once per correction iteration)
- `finalize_answer` - gate_passed, confidence band, cache write result

The LangSmith trace URL is surfaced in the status bar of the UI for every completed query.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health: service status, index counts, stub_mode |
| `POST` | `/api/query` | Run a query through the full pipeline |
| `GET` | `/api/stats` | Cache stats, session count, doc counts |
| `POST` | `/api/ingest` | Upload a document for indexing |
| `POST` | `/api/eval/run` | Start the 30-query benchmark suite |
| `GET` | `/api/eval/status/{job_id}` | Poll eval job progress |

### Example: Run a query

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the BM25 Okapi term frequency formula", "session_id": "demo-001"}'
```

Response:

```json
{
  "session_id": "demo-001",
  "answer": "BM25 scores documents using ...",
  "retrieval_strategy": "bm25",
  "evaluation_mode": "real",
  "evaluation_passed": true,
  "gate_passed": true,
  "correction_attempts": 0,
  "served_from_cache": false,
  "confidence_band": "VERIFIED",
  "total_latency_ms": 42310,
  "ragas_scores": {
    "faithfulness": 0.90,
    "answer_relevancy": 0.88,
    "context_groundedness": 0.85,
    "composite": 0.886,
    "evaluation_mode": "real",
    "scorer_model": "ollama/llama3.2"
  },
  "reranked_chunks": ["..."],
  "langsmith_trace_url": "https://smith.langchain.com/..."
}
```

### Example: Health check

```bash
curl "http://localhost:8000/api/health"
```

```json
{
  "status": "ok",
  "stub_mode": false,
  "index_status": {
    "bm25": "ready",
    "bm25_doc_count": 161,
    "vector": "ready",
    "vector_doc_count": 161,
    "reranker": "loaded"
  },
  "services": {
    "postgres": "connected",
    "redis": "connected",
    "ollama": "connected"
  },
  "langsmith": "enabled",
  "checkpointing": "enabled (MemorySaver)"
}
```

---

## Benchmark Results

Run the evaluation suite:

```bash
cd backend
curl -X POST http://localhost:8000/api/eval/run | python -m json.tool
# Poll with returned job_id
curl http://localhost:8000/api/eval/status/{job_id}
```

| Metric | Value |
|---|---|
| Completion Rate | 86.7% (26/30) |
| Strategy Classification Accuracy | 60.0% |
| Avg Faithfulness Score | 0.6000 |
| Avg Answer Relevancy | 0.6077 |
| Avg Context Groundedness | 0.6223 |
| Avg Composite RAGAS Score | 0.6068 |
| Correction Rate | 53.3% |
| Correction Success Rate | 100.0% |
| Cache Hit Rate (after warmup) | 0.0% |
| Avg Query Latency | 77,472 ms |
| P95 Query Latency | 120,022 ms |
| Keyword Hit Rate | 58.8% |
| Scorer Model | ollama/llama3.2 |

*Last run: 2026-03-29, real Ollama scoring, all pipeline fixes applied.*

**Category breakdown:**

| Category | Completed | Strategy Accuracy | Notes |
|---|---|---|---|
| FACTUAL | 5/5 | 100% | BM25 routing correct on all 5 |
| ABSTRACT | 5/5 | 100% | Vector routing correct on all 5 |
| TIME_SENSITIVE | 4/5 | 60% | 1 timeout at 120s gate |
| MULTI_HOP | 3/5 | 60% | 2 timeouts, multi-step reasoning under full correction loop |
| STRESS_CORRECTION | 5/5 | 0% (by design) | Vague queries resolve via vector, not expected hybrid label |
| EDGE_CASES | 4/5 | 40% | 1 timeout |

Correction success rate 100%: every query that entered the correction loop recovered. The 4 timeouts occur on multi-step reasoning queries under full Ollama evaluation latency at the 120-second per-query gate.

---

## Project Structure

```
Axiom-Adaptive-RAG-main/
├── docker-compose.yml                     <- PostgreSQL (pgvector) + Redis
├── README.md
├── .env.example                           <- Copy to backend/.env
├── backend/
│   ├── server.py                          <- FastAPI app, rate limiting, session management, persistence
│   ├── requirements.txt
│   ├── axiom/
│   │   ├── config.py                      <- All tunable parameters, thresholds, top_k values, timeouts
│   │   ├── graph/
│   │   │   ├── state.py                   <- AxiomState TypedDict, all pipeline fields
│   │   │   ├── graph.py                   <- StateGraph with cyclic edges, MemorySaver checkpointing
│   │   │   └── nodes/
│   │   │       ├── classify_query.py      <- Claude Sonnet, FACTUAL/ABSTRACT/TIME_SENSITIVE/MULTI_HOP
│   │   │       ├── check_cache.py         <- Redis two-tier lookup (exact + cosine similarity)
│   │   │       ├── route_retrieval.py     <- Pure routing, no LLM
│   │   │       ├── retrieve_bm25.py       <- rank_bm25.BM25Okapi
│   │   │       ├── retrieve_vector.py     <- pgvector async query
│   │   │       ├── retrieve_hybrid.py     <- asyncio.gather BM25+vector, RRF merge
│   │   │       ├── decompose_query.py     <- Multi-hop sub-query runner
│   │   │       ├── rerank_chunks.py       <- CrossEncoder ms-marco-MiniLM-L-6-v2
│   │   │       ├── generate_answer.py     <- Claude Sonnet with retry logic
│   │   │       ├── evaluate_answer.py     <- RAGAS via Ollama critic
│   │   │       ├── rewrite_query.py       <- Claude Sonnet query rewriter
│   │   │       └── finalize_answer.py     <- Gate logic, cache write, confidence band
│   │   ├── retrieval/
│   │   │   ├── bm25_index.py              <- BM25Index singleton, hydrated from pgvector on startup
│   │   │   ├── vector_store.py            <- Async SQLAlchemy + pgvector
│   │   │   ├── embeddings.py              <- OpenAI text-embedding-3-small singleton
│   │   │   ├── hybrid_fusion.py           <- Reciprocal Rank Fusion k=60
│   │   │   └── reranker.py                <- CrossEncoderReranker with fallback
│   │   ├── evaluation/
│   │   │   ├── ragas_scorer.py            <- Three-metric Ollama scorer
│   │   │   ├── critic_llm.py              <- Ollama HTTP client with TTL connection cache
│   │   │   └── thresholds.py              <- Confidence band definitions
│   │   ├── cache/
│   │   │   └── semantic_cache.py          <- Redis semantic cache, two-tier lookup, sorted index
│   │   ├── ingest/
│   │   │   ├── loader.py                  <- pdfplumber, tiktoken chunker, NLTK sentence splitter
│   │   │   └── indexer.py                 <- Dual BM25 + pgvector writer
│   │   ├── observability/
│   │   │   └── langsmith.py               <- LangSmith RunnableConfig
│   │   └── eval_suite/
│   │       ├── benchmark.py               <- 30 queries across 6 categories
│   │       └── runner.py                  <- Benchmark runner, aggregate metrics, background job
├── frontend/
│   └── src/
│       ├── App.js                         <- Main dashboard, API wiring
│       ├── index.css                      <- MERIDIAN design system
│       └── components/axiom/
│           ├── QueryInput.js
│           ├── PipelineStrip.js           <- 12-node pipeline visualization
│           ├── SignalPanel.js             <- Retrieved chunks with rerank scores
│           ├── EvaluationPanel.js         <- RAGAS scores, score history
│           ├── CorrectionRecord.js        <- Per-iteration rewrite reasoning
│           ├── AnswerPanel.js             <- Final answer, confidence band, sources
│           ├── StatusBar.js               <- Docs count, cache entries, LangSmith URL
│           ├── UploadPanel.js             <- Document upload and index status
│           └── HexBackground.js           <- Animated topology background
```

---

## Troubleshooting

**Backend won't start - `KeyError: ANTHROPIC_API_KEY`**

Make sure `backend/.env` exists. Copy from the example:
```bash
cp .env.example backend/.env
```
Then fill in your API keys.

**`pgvector connection failed`**

Docker containers are not running. Start them:
```bash
docker-compose up -d
docker-compose ps
```
Both `postgres` and `redis` should show status `running`.

**`BM25 hydrated - 0 chunks loaded`**

No documents have been indexed yet. Upload PDFs through the UI upload panel. The status bar updates with chunk count after each upload.

**Queries return `INSUFFICIENT_CONTEXT`**

The knowledge base does not contain relevant documents for the query. Upload documents related to your question domain.

**`evaluation_mode: "mock"` in every response**

Ollama is not running or the `llama3.2` model is not pulled. Fix:
```bash
ollama serve
ollama pull llama3.2
```

**Frontend shows `stub_mode: true` in health**

One of three conditions: Ollama is not connected, the reranker model failed to load, or the BM25 index is empty. Check each service individually using the health endpoint.

**Correction loop runs 3 times on every query**

Ollama is down and mock evaluation mode is active. Mock mode forces `evaluation_passed: false` on every attempt. Start Ollama to get real scoring.

**Rate limit hit - HTTP 429**

The query endpoint allows 30 requests per minute per IP. If running the eval suite, it batches queries with thread IDs. A single IP hitting the endpoint manually should not reach this limit under normal use.

---

## Configuration

All tunable parameters live in the config module. Key values:

| Parameter | Default | Description |
|---|---|---|
| `faithfulness_threshold` | 0.75 | Minimum faithfulness to pass hallucination gate |
| `relevancy_threshold` | 0.60 | Minimum relevancy score |
| `groundedness_threshold` | 0.60 | Minimum groundedness score |
| `max_correction_attempts` | 3 | Maximum rewrite iterations before finalizing |
| `bm25_top_k` | 20 | Chunks returned by BM25 before reranking |
| `vector_top_k` | 20 | Chunks returned by vector search before reranking |
| `rerank_top_k` | 5 | Chunks that pass reranker to reach generation |
| `cache_similarity_threshold` | 0.85 | Cosine similarity required for cache hit |
| `cache_ttl_seconds` | 604800 | Cache entry lifetime (7 days) |

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude Sonnet |
| `OPENAI_API_KEY` | Yes | OpenAI API key for text-embedding-3-small |
| `POSTGRES_URL` | No | PostgreSQL connection string (default: docker-compose value) |
| `REDIS_URL` | No | Redis connection string (default: `redis://localhost:6379`) |
| `OLLAMA_URL` | No | Ollama base URL (default: `http://localhost:11434`) |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | LangSmith project name (default: `axiom-rag`) |

---

## Key Design Decisions

**Why a cyclic graph instead of a linear chain?**

The correction loop requires the graph to return to an earlier node after evaluation. A linear chain cannot express this without re-running from the start and wasting the retrieval work already done. The cyclic LangGraph routes back to `rewrite_query` and then to retrieval without touching the nodes that already ran.

**Why BM25 and vector search together?**

BM25 is strong on exact terminology: technical names, formulas, specific phrases. Vector search is strong on semantic meaning: paraphrases, concepts, intent. Neither alone is sufficient. The hybrid path runs both in parallel and merges via RRF so neither ranking dominates.

**Why a cross-encoder for reranking instead of just using vector scores?**

Bi-encoder vector scores (used in retrieval) are computed independently for the query and each document. Cross-encoders see the query and document together, which produces more accurate relevance judgments. The tradeoff is speed: cross-encoders are too slow to run over thousands of documents but fast enough for the top 20 retrieved candidates.

**Why Ollama for evaluation instead of the same LLM used for generation?**

Using Claude to evaluate Claude's own outputs introduces bias. A separate local model (llama3.2 via Ollama) acts as an independent critic. This also keeps evaluation costs near zero regardless of query volume.

**Why Redis for caching instead of a vector database?**

The cache is small (bounded to recent queries) and needs sub-second lookup. Redis with a cosine similarity scan over the 200 most recent entries is fast enough and avoids adding another infrastructure dependency. At larger scale this would be replaced with a proper ANN index.

---

## License

MIT License - Nihanth Naidu Kalisetti, 2026
