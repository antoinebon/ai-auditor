"""Report synthesis node.

Aggregation is deterministic (counting coverage classes by theme etc.); the
executive summary is the one place in the pipeline where we ask the LLM to
produce prose. Summary generation is optional — ``summary_llm=None`` yields
a plain deterministic paragraph so tests don't need an LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from ai_auditor.graph.state import MainState
from ai_auditor.llm import content_text
from ai_auditor.models import ControlAssessment, Report, ReportStats


def make_synthesize_node(
    model_name: str,
    *,
    agentic: bool,
    summary_llm: ChatOllama | None = None,
) -> Callable[[MainState], dict[str, Report]]:
    """Closure factory for the report-synthesis node."""

    def synthesize(state: MainState) -> dict[str, Report]:
        assessments = state.get("assessments", [])
        stats = _compute_stats(assessments)
        parsed = state.get("parsed")
        summary = _compose_summary(assessments, stats, summary_llm)
        report = Report(
            document_path=state["document_path"],
            document_title=parsed.title if parsed else None,
            analyzed_at=datetime.now(UTC),
            model=model_name,
            agentic=agentic,
            assessments=assessments,
            summary=summary,
            stats=stats,
        )
        return {"report": report}

    return synthesize


def _compute_stats(assessments: list[ControlAssessment]) -> ReportStats:
    total = len(assessments)
    covered = sum(1 for a in assessments if a.coverage == "covered")
    partial = sum(1 for a in assessments if a.coverage == "partial")
    not_covered = sum(1 for a in assessments if a.coverage == "not_covered")

    by_theme: dict[str, dict[str, int]] = {}
    for a in assessments:
        theme = _theme_from_control_id(a.control_id)
        by_theme.setdefault(
            theme, {"covered": 0, "partial": 0, "not_covered": 0, "total": 0}
        )
        by_theme[theme][a.coverage] += 1
        by_theme[theme]["total"] += 1

    return ReportStats(
        total_controls=total,
        covered=covered,
        partial=partial,
        not_covered=not_covered,
        by_theme=by_theme,
    )


def _theme_from_control_id(cid: str) -> str:
    # Annex A identifiers map A.5 → Organizational, A.6 → People,
    # A.7 → Physical, A.8 → Technological.
    lookup = {
        "5": "Organizational",
        "6": "People",
        "7": "Physical",
        "8": "Technological",
    }
    try:
        theme_num = cid.split(".")[1]
    except IndexError:
        return "Unknown"
    return lookup.get(theme_num, "Unknown")


def _compose_summary(
    assessments: list[ControlAssessment],
    stats: ReportStats,
    llm: ChatOllama | None,
) -> str:
    if llm is None:
        return _deterministic_summary(stats)
    user = _summary_user_prompt(assessments, stats)
    system = (
        "You are drafting a one-paragraph executive summary for a compliance "
        "gap-analysis report. Summarise the coverage distribution and the "
        "most notable gaps in 3-5 sentences. Be factual and concise. No "
        "bullet points, no markdown headings, no preamble."
    )
    message = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return content_text(message).strip()


def _deterministic_summary(stats: ReportStats) -> str:
    return (
        f"Analysed {stats.total_controls} ISO 27001:2022 Annex A controls: "
        f"{stats.covered} covered, {stats.partial} partially covered, and "
        f"{stats.not_covered} not covered."
    )


def _summary_user_prompt(assessments: list[ControlAssessment], stats: ReportStats) -> str:
    lines = [
        "Coverage summary:",
        f"- total controls analysed: {stats.total_controls}",
        f"- covered: {stats.covered}",
        f"- partial: {stats.partial}",
        f"- not_covered: {stats.not_covered}",
        "",
        "Notable gaps (controls marked not_covered or partial):",
    ]
    gaps = [a for a in assessments if a.coverage != "covered"]
    for a in gaps[:12]:
        lines.append(f"- {a.control_id} ({a.coverage}): {a.reasoning}")
    if len(gaps) > 12:
        lines.append(f"… and {len(gaps) - 12} more.")
    return "\n".join(lines)
