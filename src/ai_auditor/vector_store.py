"""Vector store wrapper.

Uses ChromaDB's ``EphemeralClient`` (in-memory, no persistence) so one
analyse run starts with a fresh collection and disposes of it on exit.
Swapping to ``PersistentClient(path=...)`` is a one-line change at the
``_make_client`` seam if we ever want the policy index to survive runs.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from ai_auditor.models import EmbeddedChunk


@dataclass(frozen=True)
class QueryHit:
    """One retrieval result: the chunk id with a similarity score and metadata.

    ``similarity`` is in [0, 1] where higher is better (we convert Chroma's
    cosine *distance* to a similarity score with ``1 - distance``).
    """

    chunk_id: str
    similarity: float
    metadata: dict[str, Any]
    document: str


class VectorStore:
    """Thin ChromaDB collection wrapper scoped to a single analyse run."""

    def __init__(self, collection_name: str = "policy_chunks") -> None:
        self._client: ClientAPI = _make_client()
        # Ephemeral client starts empty, but be defensive against reuse
        # (chroma raises a bespoke NotFoundError; we don't care which exception).
        with contextlib.suppress(Exception):
            self._client.delete_collection(collection_name)
        self._collection: Collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, embedded: list[EmbeddedChunk]) -> None:
        """Insert policy chunks with their precomputed embeddings."""
        if not embedded:
            return
        # Chroma's stubs declare a tight union for `embeddings`; list[list[float]]
        # is accepted at runtime but mypy can't see it is a subtype.
        self._collection.upsert(
            ids=[e.chunk.id for e in embedded],
            embeddings=[e.embedding for e in embedded],  # type: ignore[arg-type]
            documents=[e.chunk.text for e in embedded],
            metadatas=[
                {
                    "section_id": e.chunk.section_id,
                    "section_heading": e.chunk.section_heading,
                    "page_start": e.chunk.page_start,
                    "page_end": e.chunk.page_end,
                }
                for e in embedded
            ],
        )

    def query(self, query_embeddings: list[list[float]], top_k: int = 5) -> list[list[QueryHit]]:
        """Run ``query_embeddings`` against the collection.

        Returns one list of hits per query embedding, already sorted by
        similarity (best first) and capped at ``top_k`` per query.
        """
        if not query_embeddings:
            return []
        raw = self._collection.query(
            query_embeddings=query_embeddings,  # type: ignore[arg-type]
            n_results=top_k,
        )
        results: list[list[QueryHit]] = []
        ids_by_query = raw.get("ids") or []
        dists_by_query = raw.get("distances") or []
        metas_by_query = raw.get("metadatas") or []
        docs_by_query = raw.get("documents") or []
        for ids, dists, metas, docs in zip(
            ids_by_query, dists_by_query, metas_by_query, docs_by_query, strict=True
        ):
            hits = [
                QueryHit(
                    chunk_id=cid,
                    similarity=float(1.0 - dist),
                    metadata=dict(meta or {}),
                    document=doc or "",
                )
                for cid, dist, meta, doc in zip(ids, dists, metas, docs, strict=True)
            ]
            results.append(hits)
        return results


def _make_client() -> ClientAPI:
    """Single seam for persistence mode; flip here to go to disk."""
    return chromadb.EphemeralClient()
