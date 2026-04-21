"""Deterministic multi-query retrieval for one control.

For each control we embed a set of pre-generated query phrasings, query the
vector store once per phrasing, and union the hits by chunk id keeping the
best similarity score. This mitigates the vocabulary gap between a
compliance framework's wording and the target policy's wording — a
single-embedding query often misses hits that use synonyms.

When a control has no pre-generated queries yet, we fall back to a single
query built from its title + description. The behaviour and output shape
are otherwise identical.
"""

from __future__ import annotations

from ai_auditor.embedding import Embedder
from ai_auditor.models import Control
from ai_auditor.vector_store import QueryHit, VectorStore


def retrieve_for_control(
    control: Control,
    embedder: Embedder,
    store: VectorStore,
    per_query_k: int = 5,
    final_k: int = 10,
) -> list[QueryHit]:
    """Return the top ``final_k`` policy chunks for ``control``.

    ``per_query_k`` controls how many hits we pull from each query phrasing
    before unioning. ``final_k`` caps the deduplicated output.
    """
    queries = _control_queries(control)
    query_embeddings = embedder.encode(queries)
    per_query_hits = store.query(query_embeddings, top_k=per_query_k)

    best: dict[str, QueryHit] = {}
    for hits in per_query_hits:
        for hit in hits:
            existing = best.get(hit.chunk_id)
            if existing is None or hit.similarity > existing.similarity:
                best[hit.chunk_id] = hit

    ordered = sorted(best.values(), key=lambda h: h.similarity, reverse=True)
    return ordered[:final_k]


def _control_queries(control: Control) -> list[str]:
    """Effective query list for a control.

    If pre-generated queries are present in the YAML we trust them.
    Otherwise we synthesise a single query from the control's title and
    description — retrieval quality is worse without multi-query, but the
    pipeline still produces a valid output.
    """
    if control.queries:
        return list(control.queries)
    return [f"{control.title} — {control.description}"]
