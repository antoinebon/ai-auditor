"""Report rendering — ``Report`` to human-readable Markdown and to disk."""

from __future__ import annotations

from pathlib import Path

from ai_auditor.models import ControlAssessment, Report, SectionRef

COVERAGE_ICON = {
    "covered": "✅",
    "partial": "🟡",
    "not_covered": "❌",
}


def render_markdown(report: Report) -> str:
    """Render ``report`` as a self-contained Markdown document."""
    s = report.stats
    lines: list[str] = []
    title = report.document_title or report.document_path.name
    lines.append(f"# Gap analysis — {title}")
    lines.append("")
    lines.append(f"- Document: `{report.document_path}`")
    lines.append(f"- Analysed at: {report.analyzed_at.isoformat()}")
    lines.append(f"- LLM: `{report.model}`  (agentic retrieval: `{report.agentic}`)")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(report.summary.strip())
    lines.append("")
    lines.append("## Coverage overview")
    lines.append("")
    lines.append(f"- Controls analysed: **{s.total_controls}**")
    lines.append(f"- ✅ covered: **{s.covered}**")
    lines.append(f"- 🟡 partial: **{s.partial}**")
    lines.append(f"- ❌ not covered: **{s.not_covered}**")
    lines.append("")
    if s.by_theme:
        lines.append("### By theme")
        lines.append("")
        lines.append("| Theme | Total | Covered | Partial | Not covered |")
        lines.append("| --- | ---:| ---:| ---:| ---:|")
        for theme, counts in sorted(s.by_theme.items()):
            lines.append(
                f"| {theme} "
                f"| {counts.get('total', 0)} "
                f"| {counts.get('covered', 0)} "
                f"| {counts.get('partial', 0)} "
                f"| {counts.get('not_covered', 0)} |"
            )
        lines.append("")
    lines.append("## Per-control findings")
    lines.append("")
    sections_by_id = {s.id: s for s in report.sections}
    for a in report.assessments:
        lines.extend(_render_assessment(a, sections_by_id))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(report: Report, out_dir: Path) -> tuple[Path, Path]:
    """Write ``report`` to ``<out_dir>/report.json`` + ``report.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _render_assessment(a: ControlAssessment, sections_by_id: dict[str, SectionRef]) -> list[str]:
    icon = COVERAGE_ICON.get(a.coverage, "•")
    lines = [
        f"### {icon} {a.control_id} — {a.coverage} ({a.confidence})",
        "",
        a.reasoning.strip(),
    ]
    if a.evidence:
        lines.append("")
        lines.append("**Evidence:**")
        for span in a.evidence:
            ref = sections_by_id.get(span.section_id)
            if ref is not None:
                lines.append(
                    f"- `{span.section_id}` — {ref.heading} (p.{ref.page_start}-{ref.page_end})"
                )
            else:
                lines.append(f"- `{span.section_id}`")
            lines.append(f"  _{span.relevance_note.strip()}_")
    return lines
