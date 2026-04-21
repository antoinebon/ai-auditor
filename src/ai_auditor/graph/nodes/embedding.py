"""Embedding node — computes vectors for policy chunks and upserts them."""

from __future__ import annotations

from collections.abc import Callable

from ai_auditor.embedding import Embedder
from ai_auditor.graph.state import MainState
from ai_auditor.models import EmbeddedChunk
from ai_auditor.vector_store import VectorStore


def make_embed_chunks_node(
    embedder: Embedder, store: VectorStore
) -> Callable[[MainState], dict[str, list[EmbeddedChunk]]]:
    """Closure factory that captures the embedder and the vector store."""

    def embed_chunks(state: MainState) -> dict[str, list[EmbeddedChunk]]:
        chunks = state["chunks"]
        if not chunks:
            return {}
        vectors = embedder.encode([c.text for c in chunks])
        embedded = [
            EmbeddedChunk(chunk=chunk, embedding=vec)
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        store.upsert_chunks(embedded)
        # The store holds the data from here on; no need to carry it in state.
        return {}

    return embed_chunks
