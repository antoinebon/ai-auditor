"""Tests for the PDF parsing node."""

from __future__ import annotations

from pathlib import Path

from ai_auditor.graph.nodes.parsing import parse_pdf


def test_parse_pdf_extracts_expected_sections(sample_policy_pdf: Path) -> None:
    parsed = parse_pdf(sample_policy_pdf)

    # Body text is intact.
    concatenated = " ".join(s.text for s in parsed.sections)
    for needle in ["role-based access", "Security Incident Response Team", "awareness training"]:
        assert needle in concatenated, f"Expected '{needle}' in parsed text"

    # Three numbered sections should be detected (the title counts as a
    # leading heading; the three numbered sections follow it).
    numbered_headings = [
        s.heading for s in parsed.sections if s.heading.startswith(("1.", "2.", "3."))
    ]
    assert numbered_headings == ["1. Access Control", "2. Incident Response", "3. Training"]


def test_parse_pdf_fills_page_range(sample_policy_pdf: Path) -> None:
    parsed = parse_pdf(sample_policy_pdf)
    assert parsed.page_count >= 1
    for section in parsed.sections:
        assert 1 <= section.page_start <= parsed.page_count
        assert section.page_end >= section.page_start


def test_parse_pdf_assigns_unique_section_ids(sample_policy_pdf: Path) -> None:
    parsed = parse_pdf(sample_policy_pdf)
    ids = [s.id for s in parsed.sections]
    assert len(ids) == len(set(ids))
