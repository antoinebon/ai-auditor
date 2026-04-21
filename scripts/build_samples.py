"""Build the synthetic sample policy PDFs from text sources.

Only the synthetic samples are generated here; the real, third-party
PDFs (e.g. ``northwestern_infosec_policy.pdf``) live in the repo as
downloaded artefacts with attribution in ``data/examples/README.md``.

Keeping the synthetic sources as plain text in
``data/examples/sources/*.txt`` makes them easy to diff and review; the
PDFs are regenerated on demand with reportlab. Run from the repo root::

    uv run python scripts/build_samples.py
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")
SOURCES = Path("data/examples/sources")
TARGETS = Path("data/examples")


def _strip_metadata_header(text: str) -> str:
    """Drop any attribution / licence header above the first ``---`` line.

    Source ``.txt`` files may prefix the document body with repo-facing
    metadata (provenance URL, licence notice) as ``#``-prefixed comment
    lines, terminated by a ``---`` separator. Everything above the
    separator is for the repo reader, not for the analyser — stripping it
    keeps attribution out of the embedded chunks and the retrieval index.
    """
    separator = "\n---\n"
    if separator in text:
        return text.split(separator, 1)[1].lstrip()
    # Plain files without a separator are passed through unchanged.
    return text


def build_pdf(src: Path, dst: Path) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PolicyTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "PolicyHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "PolicyBody",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=15,
        spaceAfter=6,
    )

    text = _strip_metadata_header(src.read_text(encoding="utf-8"))
    blocks: list[tuple[str, str]] = []
    title: str | None = None
    buffer: list[str] = []

    def flush(current_heading: str | None) -> None:
        if current_heading is None and not buffer:
            return
        body = " ".join(b.strip() for b in buffer if b.strip())
        blocks.append((current_heading or "", body))
        buffer.clear()

    current_heading: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if buffer:
                flush(current_heading)
                current_heading = None
            continue
        if title is None:
            title = line.strip()
            continue
        if HEADING_RE.match(line) or (line.isupper() and len(line) < 60):
            flush(current_heading)
            current_heading = line.strip()
        else:
            buffer.append(line)
    flush(current_heading)

    flowables: list = []
    if title:
        flowables.append(Paragraph(title, title_style))
        flowables.append(Spacer(1, 6))
    for heading, body in blocks:
        if heading:
            flowables.append(Paragraph(heading, heading_style))
        if body:
            flowables.append(Paragraph(body, body_style))

    dst.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(dst), pagesize=LETTER).build(flowables)


def main() -> None:
    mapping = {
        "minimal_policy.txt": "minimal_policy.pdf",
        "sans_acceptable_use.txt": "sans_acceptable_use.pdf",
    }
    for src_name, dst_name in mapping.items():
        src = SOURCES / src_name
        dst = TARGETS / dst_name
        build_pdf(src, dst)
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
