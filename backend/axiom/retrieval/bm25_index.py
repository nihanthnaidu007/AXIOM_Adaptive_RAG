"""AXIOM BM25 Index - Real BM25Okapi implementation using rank_bm25."""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 index using rank_bm25 library for real keyword scoring."""

    def __init__(self):
        self._bm25 = None
        self._documents: List[Dict] = []
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, split on non-alphanumeric, drop tokens under 2 chars."""
        tokens = re.split(r'[^a-z0-9]+', text.lower())
        return [t for t in tokens if len(t) >= 2]

    def build(self, chunks: List[Dict]) -> None:
        """Build index from scratch with the given chunks."""
        from rank_bm25 import BM25Okapi
        self._documents = list(chunks)
        self._tokenized_corpus = [self._tokenize(c.get("content", "")) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built with %d documents", len(self._documents))

    # Keep build_index as an alias for backward compat with any callers
    def build_index(self, chunks: List[Dict]) -> None:
        self.build(chunks)

    def add_chunks(self, chunks: List[Dict]) -> None:
        """Extend documents and rebuild index from scratch."""
        from rank_bm25 import BM25Okapi
        self._documents.extend(chunks)
        self._tokenized_corpus.extend(
            self._tokenize(c.get("content", "")) for c in chunks
        )
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index rebuilt — %d total documents", len(self._documents))

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Score all documents against query, return top_k sorted descending."""
        if self._bm25 is None or not self._documents:
            return []
        try:
            tokenized_query = self._tokenize(query)
            scores = self._bm25.get_scores(tokenized_query)
            scored = sorted(
                zip(scores, self._documents),
                key=lambda x: x[0],
                reverse=True,
            )
            results = []
            for score, doc in scored[:top_k]:
                d = dict(doc)
                d["bm25_score"] = float(score)
                results.append(d)
            return results
        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc)
            return []

    def count(self) -> int:
        return len(self._documents)

    def is_empty(self) -> bool:
        return len(self._documents) == 0

    # Backward compat alias
    def get_document_count(self) -> int:
        return len(self._documents)


bm25_index = BM25Index()
