"""Tests for the Chroma-backed vector store wrapper."""

from __future__ import annotations

from ai_auditor.models import EmbeddedChunk, PolicyChunk
from ai_auditor.vector_store import VectorStore


def _chunk(chunk_id: str, text: str, section_id: str = "s_00") -> PolicyChunk:
    return PolicyChunk(
        id=chunk_id,
        section_id=section_id,
        section_heading="test",
        text=text,
        page_start=1,
        page_end=1,
    )


def test_upsert_then_query_returns_nearest() -> None:
    store = VectorStore(collection_name="t_nearest")
    # Three simple 3-D vectors pointing along different axes.
    chunks = [
        EmbeddedChunk(chunk=_chunk("c_0001", "x-axis"), embedding=[1.0, 0.0, 0.0]),
        EmbeddedChunk(chunk=_chunk("c_0002", "y-axis"), embedding=[0.0, 1.0, 0.0]),
        EmbeddedChunk(chunk=_chunk("c_0003", "z-axis"), embedding=[0.0, 0.0, 1.0]),
    ]
    store.upsert_chunks(chunks)

    results = store.query([[1.0, 0.0, 0.0]], top_k=3)
    assert len(results) == 1
    hits = results[0]
    assert hits[0].chunk_id == "c_0001"
    # Similarity = 1 - cosine_distance. For identical unit vectors that's 1.0.
    assert hits[0].similarity > 0.99


def test_query_returns_metadata_and_document() -> None:
    store = VectorStore(collection_name="t_meta")
    chunks = [
        EmbeddedChunk(
            chunk=_chunk("c_0001", "policy statement about access"),
            embedding=[1.0, 0.0],
        )
    ]
    store.upsert_chunks(chunks)
    hits = store.query([[1.0, 0.0]], top_k=1)[0]
    assert hits[0].document == "policy statement about access"
    assert hits[0].metadata["section_id"] == "s_00"
    assert hits[0].metadata["section_heading"] == "test"


def test_upsert_empty_is_noop() -> None:
    store = VectorStore(collection_name="t_empty")
    store.upsert_chunks([])  # should not raise
    assert store.query([[0.0, 1.0]], top_k=3) == [[]]


def test_query_empty_returns_empty() -> None:
    store = VectorStore(collection_name="t_q_empty")
    assert store.query([], top_k=3) == []


def test_multiple_queries_return_results_per_query() -> None:
    store = VectorStore(collection_name="t_multi")
    chunks = [
        EmbeddedChunk(chunk=_chunk("c_0001", "x"), embedding=[1.0, 0.0]),
        EmbeddedChunk(chunk=_chunk("c_0002", "y"), embedding=[0.0, 1.0]),
    ]
    store.upsert_chunks(chunks)
    results = store.query([[1.0, 0.0], [0.0, 1.0]], top_k=1)
    assert len(results) == 2
    assert results[0][0].chunk_id == "c_0001"
    assert results[1][0].chunk_id == "c_0002"
