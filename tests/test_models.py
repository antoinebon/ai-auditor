"""Smoke tests for the domain models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_auditor.models import (
    Control,
    ControlAssessment,
    EvidenceSpan,
    PolicyChunk,
    Report,
    ReportStats,
    Section,
)


def test_control_requires_valid_theme() -> None:
    with pytest.raises(ValidationError):
        Control(
            id="A.5.1",
            title="Policies for information security",
            theme="Made-Up-Theme",  # type: ignore[arg-type]
            description="...",
        )


def test_control_is_frozen() -> None:
    c = Control(
        id="A.5.1",
        title="Policies for information security",
        theme="Organizational",
        description="...",
    )
    with pytest.raises(ValidationError):
        c.id = "A.5.2"  # type: ignore[misc]


def test_control_assessment_round_trip() -> None:
    ca = ControlAssessment(
        control_id="A.5.15",
        coverage="partial",
        evidence=[
            EvidenceSpan(chunk_id="c_0001", quote="Access is reviewed.", relevance_note="..."),
        ],
        reasoning="Partial because review cadence is unspecified.",
        confidence="medium",
    )
    dumped = ca.model_dump()
    rebuilt = ControlAssessment.model_validate(dumped)
    assert rebuilt == ca


def test_report_serializes_to_json() -> None:
    report = Report(
        document_path=Path("/tmp/example.pdf"),
        document_title="Example Policy",
        analyzed_at=datetime(2026, 4, 21, 12, 0, 0),
        model="qwen2.5:7b-instruct",
        agentic=False,
        assessments=[],
        summary="No content.",
        stats=ReportStats(total_controls=0, covered=0, partial=0, not_covered=0),
    )
    payload = report.model_dump_json()
    assert "qwen2.5:7b-instruct" in payload


def test_section_rejects_invalid_level() -> None:
    with pytest.raises(ValidationError):
        Section(id="s1", heading="h", level=0, page_start=1, page_end=1, text="t")


def test_policy_chunk_minimum_fields() -> None:
    pc = PolicyChunk(
        id="c_0001",
        section_id="s_01",
        section_heading="Scope",
        text="All employees...",
        page_start=1,
        page_end=1,
    )
    assert pc.id == "c_0001"
