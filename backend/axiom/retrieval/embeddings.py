"""AXIOM Embeddings - OpenAI text-embedding-3-small."""

import hashlib
import logging
from typing import List

from openai import AsyncOpenAI

from axiom.config import get_config

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """OpenAI embeddings client — single shared AsyncOpenAI instance."""

    def __init__(self):
        cfg = get_config()
        self._client = AsyncOpenAI(api_key=cfg.openai_api_key)
        self.model = cfg.embedding_model
        self.dimensions = cfg.embedding_dimensions
        self._cache: dict[str, List[float]] = {}

    async def embed_text(self, text: str) -> List[float]:
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        response = await self._client.embeddings.create(
            model=self.model, input=text, dimensions=self.dimensions
        )
        embedding = response.data[0].embedding
        self._cache[cache_key] = embedding
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self.model, input=texts, dimensions=self.dimensions
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings = [item.embedding for item in sorted_data]
        for text, emb in zip(texts, embeddings):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            self._cache[cache_key] = emb
        return embeddings


embeddings_client = EmbeddingsClient()


async def embed_text(text: str) -> List[float]:
    return await embeddings_client.embed_text(text)


async def embed_batch(texts: List[str]) -> List[List[float]]:
    return await embeddings_client.embed_batch(texts)
