"""Domain models — the contract between pipeline stages.

These are Pydantic models (not TypedDicts) because they carry validation for
LLM-produced structured outputs and are the stable schema written to the
report JSON. LangGraph state objects (which hold these) are defined in
``graph/state.py`` and use ``TypedDict`` to match LangGraph idioms.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Coverage = Literal["covered", "partial", "not_covered"]
Confidence = Literal["low", "medium", "high"]
Theme = Literal["Organizational", "People", "Physical", "Technological"]


class Control(BaseModel):
    """One ISO 27001:2022 Annex A control."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Control identifier, e.g. 'A.5.15'.")
    title: str
    theme: Theme
    description: str = Field(description="Paraphrased description suitable for semantic retrieval.")
    queries: list[str] = Field(
        default_factory=list,
        description="Pre-generated multi-query reformulations for retrieval.",
    )


class Section(BaseModel):
    """A heading-delimited section of the parsed document."""

    id: str
    heading: str
    level: int = Field(ge=1, le=6)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str


class ParsedDocument(BaseModel):
    """Output of the PDF parser."""

    path: Path
    title: str | None = None
    sections: list[Section]
    page_count: int = Field(ge=0)


class PolicyChunk(BaseModel):
    """A retrievable chunk of policy text, bounded to a section."""

    id: str
    section_id: str
    section_heading: str
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class EmbeddedChunk(BaseModel):
    """A PolicyChunk paired with its embedding vector."""

    chunk: PolicyChunk
    embedding: list[float]


class EvidenceSpan(BaseModel):
    """A single evidence citation supporting an assessment.

    Cites a policy section by id; the rendered report looks up the
    section's heading and page range from ``Report.sections``. The
    ``relevance_note`` is the LLM's one-sentence justification for
    this particular citation — the overall verdict reasoning lives
    on ``ControlAssessment.reasoning``.
    """

    section_id: str
    relevance_note: str = Field(description="Why this section supports the verdict.")


class SectionRef(BaseModel):
    """Report-side view of a parsed ``Section`` — no text, just metadata.

    Shipped on ``Report.sections`` so renderers can resolve
    ``EvidenceSpan.section_id`` to a human-readable heading + page range
    without needing to re-parse the source PDF.
    """

    id: str
    heading: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class ControlAssessment(BaseModel):
    """Per-control judgment produced by the assessment node."""

    control_id: str
    coverage: Coverage
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    reasoning: str
    confidence: Confidence


class ReportStats(BaseModel):
    """Aggregated coverage stats across all assessments."""

    total_controls: int = Field(ge=0)
    covered: int = Field(ge=0)
    partial: int = Field(ge=0)
    not_covered: int = Field(ge=0)
    by_theme: dict[str, dict[str, int]] = Field(default_factory=dict)


class Report(BaseModel):
    """End-to-end gap analysis report."""

    document_path: Path
    document_title: str | None = None
    analyzed_at: datetime
    model: str = Field(description="Ollama model identifier used for LLM calls.")
    agentic: bool = Field(default=False, description="Whether the agentic retrieval path was used.")
    assessments: list[ControlAssessment]
    summary: str = Field(description="LLM-generated executive summary paragraph.")
    stats: ReportStats
    sections: list[SectionRef] = Field(
        default_factory=list,
        description="Section registry — evidence spans look up heading + page range here.",
    )
