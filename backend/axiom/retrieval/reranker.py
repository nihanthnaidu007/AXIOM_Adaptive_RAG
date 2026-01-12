"""AXIOM Cross-Encoder Reranker - Real cross-encoder scoring with fallback."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self):
        self._model = None
        self._loaded = False

    def load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._loaded = True
            logger.info("Cross-encoder reranker loaded successfully")
        except Exception as e:
            logger.warning("Cross-encoder failed to load: %s", e)
            self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def rerank(self, query: str, chunks: list, top_k: int = 5) -> list:
        if not self._loaded:
            self.load()

        if not self._loaded:
            # Fallback: return top_k with descending position-based scores, log clearly
            logger.warning("Reranker not loaded — using position-based fallback scores")
            result = []
            for i, chunk in enumerate(chunks[:top_k]):
                chunk.rerank_score = round(0.95 - i * 0.08, 3)
                chunk.pre_rerank_position = i
                chunk.post_rerank_position = i
                result.append(chunk)
            return result

        # Build (query, passage) pairs
        pairs = []
        for chunk in chunks:
            if hasattr(chunk, 'content'):
                content = chunk.content
            elif isinstance(chunk, dict):
                content = chunk.get('content', '')
            else:
                content = str(chunk)
            pairs.append((query, content))

        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.warning("Cross-encoder predict failed: %s", e)
            for i, chunk in enumerate(chunks[:top_k]):
                chunk.rerank_score = round(0.95 - i * 0.08, 3)
                chunk.pre_rerank_position = i
                chunk.post_rerank_position = i
            return chunks[:top_k]

        # Sort by score descending, assign positions
        scored_chunks = list(zip([float(s) for s in scores], range(len(chunks)), chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for new_pos, (score, orig_pos, chunk) in enumerate(scored_chunks[:top_k]):
            chunk.rerank_score = score
            chunk.pre_rerank_position = orig_pos
            chunk.post_rerank_position = new_pos
            reranked.append(chunk)

        return reranked


reranker = CrossEncoderReranker()


def get_reranker() -> CrossEncoderReranker:
    """Return the module-level singleton (backward compat for server.py)."""
    return reranker
