"""Chunking node — section text to bounded, overlapping ``PolicyChunk``s.

We count tokens with a naive word-based approximation (1 token ~= 0.75 words
for English; we use words directly with a conservative target). This keeps
us free of a tokenizer dependency and is accurate enough for bounding
chunks that will be fed to an embedding model and an LLM.
"""

from __future__ import annotations

import re

from ai_auditor.models import ParsedDocument, PolicyChunk, Section

DEFAULT_TARGET_WORDS = 220  # ~ 280-300 tokens for English text
DEFAULT_OVERLAP_WORDS = 40
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def chunk_document(
    document: ParsedDocument,
    target_words: int = DEFAULT_TARGET_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[PolicyChunk]:
    """Return the document's sections broken into bounded chunks.

    Sections smaller than ``target_words`` become a single chunk. Larger
    sections are split on sentence boundaries into overlapping windows.
    """
    if overlap_words >= target_words:
        raise ValueError("overlap_words must be smaller than target_words")

    chunks: list[PolicyChunk] = []
    counter = 0
    for section in document.sections:
        for text in _split_section(section, target_words, overlap_words):
            counter += 1
            chunks.append(
                PolicyChunk(
                    id=f"c_{counter:04d}",
                    section_id=section.id,
                    section_heading=section.heading,
                    text=text,
                    page_start=section.page_start,
                    page_end=section.page_end,
                )
            )
    return chunks


def _split_section(section: Section, target_words: int, overlap_words: int) -> list[str]:
    words = section.text.split()
    if not words:
        return []
    if len(words) <= target_words:
        return [section.text.strip()]

    # Sentence-aware splitting: accumulate sentences until we hit the target
    # word budget, then emit a chunk and roll back by ``overlap_words`` worth
    # of the last sentences for the next window.
    sentences = SENTENCE_SPLIT.split(section.text.strip())
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sent_words = len(sentence.split())
        if buffer_words + sent_words > target_words and buffer:
            chunks.append(" ".join(buffer).strip())
            buffer, buffer_words = _overlap_tail(buffer, overlap_words)
        buffer.append(sentence)
        buffer_words += sent_words
    if buffer:
        chunks.append(" ".join(buffer).strip())
    return chunks


def _overlap_tail(buffer: list[str], overlap_words: int) -> tuple[list[str], int]:
    """Return a tail of ``buffer`` holding ~``overlap_words`` words."""
    tail: list[str] = []
    tail_words = 0
    for sentence in reversed(buffer):
        tail.insert(0, sentence)
        tail_words += len(sentence.split())
        if tail_words >= overlap_words:
            break
    return tail, tail_words
