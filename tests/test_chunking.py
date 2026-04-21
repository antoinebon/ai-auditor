"""Tests for the chunking node."""

from __future__ import annotations

from itertools import pairwise

import pytest

from ai_auditor.graph.nodes.chunking import DEFAULT_TARGET_WORDS, chunk_document
from ai_auditor.models import ParsedDocument, Section


def _doc(sections: list[Section]) -> ParsedDocument:
    return ParsedDocument(
        path="/tmp/test.pdf",  # type: ignore[arg-type]  # pydantic coerces str->Path
        title="test",
        sections=sections,
        page_count=max((s.page_end for s in sections), default=1),
    )


def test_small_section_becomes_single_chunk() -> None:
    doc = _doc(
        [
            Section(
                id="s_00",
                heading="Access Control",
                level=1,
                page_start=1,
                page_end=1,
                text="Short body text covering access control briefly.",
            )
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.section_heading == "Access Control"
    assert "access control" in chunk.text.lower()
    assert chunk.page_start == 1 and chunk.page_end == 1


def test_large_section_splits_with_overlap() -> None:
    # Build a long section: many distinct sentences, total >> target_words.
    sentences = [
        f"This is sentence number {i} describing a point about our policy." for i in range(60)
    ]
    long_text = " ".join(sentences)
    doc = _doc(
        [
            Section(
                id="s_00",
                heading="Big Section",
                level=1,
                page_start=1,
                page_end=3,
                text=long_text,
            )
        ]
    )
    chunks = chunk_document(doc, target_words=80, overlap_words=15)
    assert len(chunks) >= 2

    # Overlap: consecutive chunks share at least one sentence.
    for prev, nxt in pairwise(chunks):
        prev_tail = " ".join(prev.text.split()[-20:])
        nxt_head = " ".join(nxt.text.split()[:20])
        # Compute a rough overlap by checking whether any 5-word shingle
        # from the tail appears at the head of the next chunk.
        tail_tokens = prev_tail.split()
        shingles = {" ".join(tail_tokens[i : i + 5]) for i in range(max(0, len(tail_tokens) - 5))}
        assert any(s in nxt_head for s in shingles), "Expected word-level overlap between chunks"


def test_empty_section_produces_no_chunks() -> None:
    doc = _doc(
        [
            Section(
                id="s_00",
                heading="Empty",
                level=1,
                page_start=1,
                page_end=1,
                text="",
            )
        ]
    )
    assert chunk_document(doc) == []


def test_overlap_must_be_smaller_than_target() -> None:
    doc = _doc(
        [
            Section(
                id="s_00",
                heading="x",
                level=1,
                page_start=1,
                page_end=1,
                text="abc",
            )
        ]
    )
    with pytest.raises(ValueError):
        chunk_document(doc, target_words=10, overlap_words=10)


def test_chunk_ids_are_unique_across_sections() -> None:
    sections = [
        Section(
            id=f"s_{i:02d}",
            heading=f"Section {i}",
            level=2,
            page_start=1,
            page_end=1,
            text=f"Short text for section {i}.",
        )
        for i in range(5)
    ]
    chunks = chunk_document(_doc(sections))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    # Spot-check: default target is big enough that each small section maps 1:1.
    assert DEFAULT_TARGET_WORDS > 100
