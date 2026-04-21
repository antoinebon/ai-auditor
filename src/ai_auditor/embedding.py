"""Sentence-transformer embeddings.

Wraps the ``sentence-transformers`` model in a thin class so tests can swap
it for a fake. The model is loaded lazily on first use — importing this
module is cheap, but calling ``encode`` triggers the download (a few MB for
``all-MiniLM-L6-v2``) and the torch import chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder:
    """Lazy wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        if not texts:
            return []
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [vec.tolist() for vec in vectors]
