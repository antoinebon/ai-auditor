"""PDF parsing node.

Extracts section-structured text from a PDF using ``pymupdf``. Section
boundaries are detected heuristically from font size (headings are typically
rendered in a larger font than body text). When no detectable heading
structure exists, the whole document becomes a single synthetic section — the
downstream chunker will still break it into bounded chunks.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

import pymupdf

from ai_auditor.models import ParsedDocument, Section

# A heading is typically both larger than body text and short. We treat
# spans >= body_median * HEADING_SIZE_RATIO and <= MAX_HEADING_CHARS as a
# candidate heading. Numeric prefixes (1., 1.1, A., I., etc.) promote a span
# to heading regardless of size.
HEADING_SIZE_RATIO = 1.15
MAX_HEADING_CHARS = 140
NUMERIC_HEADING_RE = re.compile(
    r"""^(?:
        \d+(?:\.\d+)*\.?           # 1, 1.1, 1.1.1, 1., 1.1.
      | [A-Z](?:\.\d+)+\.?         # A.1, A.1.1
      | [IVXLCM]+\.                # Roman numerals: I., II., ...
      | (?:Section|Chapter|Part)\s+\d+
    )\s+\S""",
    re.VERBOSE | re.IGNORECASE,
)


def parse_pdf(path: Path) -> ParsedDocument:
    """Parse ``path`` into a ``ParsedDocument`` with heading-aware sections."""
    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
        page_spans: list[list[dict[str, Any]]] = [_page_text_spans(page) for page in doc]
        page_count = doc.page_count
        title_meta = (doc.metadata or {}).get("title") or None

    body_median = _median_body_font_size(page_spans)
    sections = _build_sections(page_spans, body_median)
    if not sections:
        # Fallback: single synthetic section covering everything.
        all_text = "\n\n".join(span["text"] for page in page_spans for span in page if span["text"])
        if all_text.strip():
            sections = [
                Section(
                    id="s_00",
                    heading=title_meta or path.stem,
                    level=1,
                    page_start=1,
                    page_end=page_count or 1,
                    text=all_text.strip(),
                )
            ]

    inferred_title = title_meta or (sections[0].heading if sections else path.stem)
    return ParsedDocument(
        path=path,
        title=inferred_title,
        sections=sections,
        page_count=page_count,
    )


def _page_text_spans(page: pymupdf.Page) -> list[dict[str, Any]]:
    """Flatten pymupdf's block/line/span tree into one span list per page.

    Each returned dict has ``text``, ``size``, ``flags``, ``page`` (1-indexed).
    """
    out: list[dict[str, Any]] = []
    page_num = page.number + 1
    raw = page.get_text("dict")  # type: ignore[no-untyped-call]
    for block in raw.get("blocks", []):
        if block.get("type", 0) != 0:
            continue  # skip images etc.
        for line in block.get("lines", []):
            # Reassemble line text from spans; track the largest span size so
            # heading detection isn't fooled by a short mixed-size line.
            parts: list[str] = []
            max_size = 0.0
            flags = 0
            for span in line.get("spans", []):
                text = span.get("text", "")
                parts.append(text)
                size = float(span.get("size", 0.0))
                if size > max_size:
                    max_size = size
                flags |= int(span.get("flags", 0))
            joined = "".join(parts).strip()
            if not joined:
                continue
            out.append(
                {
                    "text": joined,
                    "size": max_size,
                    "flags": flags,
                    "page": page_num,
                }
            )
    return out


def _median_body_font_size(pages: list[list[dict[str, Any]]]) -> float:
    sizes = [s["size"] for page in pages for s in page if s["size"] > 0]
    if not sizes:
        return 0.0
    return float(statistics.median(sizes))


def _is_heading(span: dict[str, Any], body_median: float) -> bool:
    text = span["text"]
    if len(text) > MAX_HEADING_CHARS:
        return False
    if NUMERIC_HEADING_RE.match(text):
        return True
    if body_median <= 0:
        return False
    return bool(span["size"] >= body_median * HEADING_SIZE_RATIO)


def _heading_level(size: float, body_median: float) -> int:
    """Map a heading's font-size ratio to a heading level (1=biggest)."""
    if body_median <= 0:
        return 2
    ratio = size / body_median
    if ratio >= 1.6:
        return 1
    if ratio >= 1.35:
        return 2
    if ratio >= 1.15:
        return 3
    return 4


def _build_sections(pages: list[list[dict[str, Any]]], body_median: float) -> list[Section]:
    sections: list[Section] = []
    current_heading: str | None = None
    current_level = 1
    current_page_start = 1
    current_body: list[str] = []
    current_last_page = 1

    def flush() -> None:
        nonlocal current_body, current_heading
        if current_heading is None and not current_body:
            return
        text = "\n".join(current_body).strip()
        if not text and current_heading is None:
            return
        sections.append(
            Section(
                id=f"s_{len(sections):02d}",
                heading=current_heading or "Untitled section",
                level=current_level,
                page_start=current_page_start,
                page_end=current_last_page,
                text=text,
            )
        )
        current_body = []

    for page in pages:
        for span in page:
            if _is_heading(span, body_median):
                flush()
                current_heading = span["text"]
                current_level = _heading_level(span["size"], body_median)
                current_page_start = span["page"]
                current_last_page = span["page"]
            else:
                current_body.append(span["text"])
                current_last_page = span["page"]

    flush()
    return sections
