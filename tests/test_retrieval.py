"""Tests for the multi-query retrieval node.

The real ``Embedder`` loads a sentence-transformers model; we don't want
that in unit tests, so we use a ``FakeEmbedder`` that maps predefined query
strings to hand-authored vectors. The vector store itself is real — it's
fast and cheap to spin up in-memory.
"""

from __future__ import annotations

from ai_auditor.graph.nodes.retrieval import retrieve_for_control
from ai_auditor.models import Control, EmbeddedChunk, PolicyChunk
from ai_auditor.vector_store import VectorStore


class FakeEmbedder:
    """Deterministic embedder: returns pre-registered vectors by text."""

    def __init__(self, by_text: dict[str, list[float]]) -> None:
        self._by_text = by_text

    def encode(self, texts: list[str]) -> list[list[float]]:
        try:
            return [self._by_text[t] for t in texts]
        except KeyError as exc:  # pragma: no cover — test misconfiguration
            raise AssertionError(f"FakeEmbedder missing vector for: {exc.args[0]}") from exc


def _chunk(cid: str, text: str) -> PolicyChunk:
    return PolicyChunk(
        id=cid,
        section_id="s_00",
        section_heading="h",
        text=text,
        page_start=1,
        page_end=1,
    )


def test_multi_query_unions_hits_by_chunk_id() -> None:
    store = VectorStore(collection_name="t_union")
    # Three chunks: A matches query1 strongly, B matches query2 strongly,
    # C is mediocre for both. Multi-query retrieval should surface A and B.
    store.upsert_chunks(
        [
            EmbeddedChunk(chunk=_chunk("c_A", "text A"), embedding=[1.0, 0.0, 0.0]),
            EmbeddedChunk(chunk=_chunk("c_B", "text B"), embedding=[0.0, 1.0, 0.0]),
            EmbeddedChunk(chunk=_chunk("c_C", "text C"), embedding=[0.5, 0.5, 0.1]),
        ]
    )

    control = Control(
        id="A.5.24",
        title="Incident management",
        theme="Organizational",
        description="...",
        queries=["incident response plan", "security event reporting"],
    )
    embedder = FakeEmbedder(
        {
            "incident response plan": [1.0, 0.0, 0.0],
            "security event reporting": [0.0, 1.0, 0.0],
        }
    )

    hits = retrieve_for_control(control, embedder, store, per_query_k=3, final_k=5)  # type: ignore[arg-type]
    ids = [h.chunk_id for h in hits]
    # A and B should both be surfaced by the two different queries.
    assert "c_A" in ids
    assert "c_B" in ids
    # No duplicate ids even though multiple queries ran.
    assert len(set(ids)) == len(ids)


def test_dedup_keeps_highest_similarity() -> None:
    store = VectorStore(collection_name="t_dedup")
    store.upsert_chunks(
        [EmbeddedChunk(chunk=_chunk("c_A", "A"), embedding=[1.0, 0.0])],
    )
    control = Control(
        id="A.1",
        title="t",
        theme="Organizational",
        description="d",
        queries=["strong", "weak"],
    )
    embedder = FakeEmbedder(
        {
            "strong": [1.0, 0.0],  # exact match
            "weak": [0.5, 0.5],  # partial match
        }
    )
    hits = retrieve_for_control(control, embedder, store, per_query_k=3, final_k=5)  # type: ignore[arg-type]
    assert len(hits) == 1
    # similarity should reflect the strong query, not the weak one.
    assert hits[0].similarity > 0.95


def test_fallback_query_when_no_queries_present() -> None:
    store = VectorStore(collection_name="t_fallback")
    store.upsert_chunks(
        [EmbeddedChunk(chunk=_chunk("c_A", "A"), embedding=[1.0, 0.0])],
    )
    control = Control(
        id="A.1",
        title="Access control",
        theme="Organizational",
        description="Rules for controlling access.",
        queries=[],
    )
    fallback_text = f"{control.title} — {control.description}"
    embedder = FakeEmbedder({fallback_text: [1.0, 0.0]})
    hits = retrieve_for_control(control, embedder, store, per_query_k=3, final_k=5)  # type: ignore[arg-type]
    assert hits and hits[0].chunk_id == "c_A"


def test_final_k_caps_output_size() -> None:
    store = VectorStore(collection_name="t_cap")
    store.upsert_chunks(
        [
            EmbeddedChunk(chunk=_chunk(f"c_{i}", str(i)), embedding=[1.0 - i * 0.01, 0.0])
            for i in range(8)
        ]
    )
    control = Control(
        id="A.1",
        title="t",
        theme="Organizational",
        description="d",
        queries=["q"],
    )
    embedder = FakeEmbedder({"q": [1.0, 0.0]})
    hits = retrieve_for_control(control, embedder, store, per_query_k=8, final_k=3)  # type: ignore[arg-type]
    assert len(hits) == 3
