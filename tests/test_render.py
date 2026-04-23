"""Tests for the Markdown/JSON report writers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ai_auditor.models import (
    ControlAssessment,
    EvidenceSpan,
    Report,
    ReportStats,
    SectionRef,
)
from ai_auditor.render import render_markdown, write_outputs


def _report(tmp_path: Path) -> Report:
    return Report(
        document_path=tmp_path / "policy.pdf",
        document_title="Example Policy",
        analyzed_at=datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC),
        model="qwen2.5:7b-instruct",
        agentic=False,
        assessments=[
            ControlAssessment(
                control_id="A.5.15",
                coverage="covered",
                evidence=[
                    EvidenceSpan(
                        section_id="s_01",
                        relevance_note="Sets a review cadence.",
                    ),
                ],
                reasoning="Cadence is explicit.",
                confidence="high",
            ),
            ControlAssessment(
                control_id="A.5.24",
                coverage="not_covered",
                evidence=[],
                reasoning="Incident-response plan not described.",
                confidence="medium",
            ),
        ],
        summary="Two controls analysed with one gap.",
        stats=ReportStats(
            total_controls=2,
            covered=1,
            partial=0,
            not_covered=1,
            by_theme={
                "Organizational": {
                    "covered": 1,
                    "partial": 0,
                    "not_covered": 1,
                    "total": 2,
                }
            },
        ),
        sections=[
            SectionRef(id="s_01", heading="Access Control", page_start=3, page_end=4),
            SectionRef(id="s_02", heading="Incident Response", page_start=5, page_end=6),
        ],
    )


def test_render_markdown_contains_expected_sections(tmp_path: Path) -> None:
    md = render_markdown(_report(tmp_path))
    assert "# Gap analysis — Example Policy" in md
    assert "## Executive summary" in md
    assert "Two controls analysed" in md
    assert "## Coverage overview" in md
    assert "### By theme" in md
    assert "## Per-control findings" in md
    # Assessment rendering uses icons + the control id.
    assert "A.5.15" in md and "A.5.24" in md
    assert "✅" in md
    assert "❌" in md
    # Evidence shows the section id, resolved heading, and pages.
    assert "`s_01`" in md
    assert "Access Control" in md
    assert "p.3-4" in md
    assert "Sets a review cadence." in md


def test_write_outputs_writes_json_and_md(tmp_path: Path) -> None:
    report = _report(tmp_path)
    out = tmp_path / "out"
    json_path, md_path = write_outputs(report, out)

    assert json_path.exists()
    assert md_path.exists()
    payload = json_path.read_text(encoding="utf-8")
    # JSON is valid pydantic round-trip material.
    rebuilt = Report.model_validate_json(payload)
    assert rebuilt.stats.total_controls == 2
    assert "A.5.15" in md_path.read_text(encoding="utf-8")
