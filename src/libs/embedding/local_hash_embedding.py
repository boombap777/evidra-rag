"""Deterministic, dependency-free embeddings for offline tests and smoke runs.

This provider is intentionally not a semantic-quality model.  It turns hashed
tokens into a unit vector so the ingestion/retrieval contracts can be exercised
without credentials, a model server, or network access.  Production and
benchmark profiles continue to use Qwen3-Embedding through llama.cpp.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING, Any

from src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from src.core.settings import Settings

_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


class LocalHashEmbedding(BaseEmbedding):
    """Create stable token-hash vectors for fully offline contract testing."""

    def __init__(
        self,
        settings: Settings,
        *,
        dimensions: int | None = None,
        **_: Any,
    ) -> None:
        configured = dimensions or getattr(settings.embedding, "dimensions", 64)
        if not isinstance(configured, int) or configured < 2:
            raise ValueError("LocalHashEmbedding dimensions must be an integer >= 2")
        self._dimensions = configured

    def embed(
        self,
        texts: list[str],
        trace: Any | None = None,
        **_: Any,
    ) -> list[list[float]]:
        self.validate_texts(texts)
        vectors = [self._embed_one(text) for text in texts]
        if trace is not None:
            trace.metadata.setdefault("embedding_provider", "local_hash")
        return vectors

    def get_dimension(self) -> int:
        return self._dimensions

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = _TOKEN_RE.findall(text.casefold()) or [text.casefold()]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]
