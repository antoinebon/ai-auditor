"""Assessment node — per-control coverage judgment with structured output.

Takes a ``Control`` plus the retrieved evidence hits, builds a prompt that
renders every hit inline with its ``chunk_id``, and asks the LLM to return a
``ControlAssessment``. We post-filter evidence spans so any fabricated
chunk_ids are dropped, and we downgrade coverage to ``not_covered`` if the
model claims coverage without any surviving citation.
"""

from __future__ import annotations

import logging
from importlib import resources

from langchain_ollama import ChatOllama

from ai_auditor.llm import call_json
from ai_auditor.models import Control, ControlAssessment, EvidenceSpan
from ai_auditor.vector_store import QueryHit

logger = logging.getLogger(__name__)

_ASSESSMENT_PROMPT = (
    resources.files("ai_auditor.prompts").joinpath("assessment.md").read_text(encoding="utf-8")
)


def assess_control(
    control: Control,
    evidence: list[QueryHit],
    llm: ChatOllama,
) -> ControlAssessment:
    """Produce a ``ControlAssessment`` for ``control`` given ``evidence`` hits."""
    user_prompt = _render_user_prompt(control, evidence)
    raw = call_json(llm, system=_ASSESSMENT_PROMPT, user=user_prompt, schema=ControlAssessment)
    return _validate_and_repair(raw, control, evidence)


def _render_user_prompt(control: Control, evidence: list[QueryHit]) -> str:
    header = (
        f"Control: {control.id} — {control.title}\n"
        f"Theme: {control.theme}\n"
        f"Description:\n{control.description}\n\n"
    )
    if not evidence:
        body = (
            "No evidence excerpts were retrieved from the policy document "
            "for this control. Your coverage judgment should be "
            "`not_covered` with an empty evidence list."
        )
    else:
        lines = ["Evidence excerpts retrieved from the policy document:\n"]
        for hit in evidence:
            section = hit.metadata.get("section_heading", "<unknown section>")
            page_start = hit.metadata.get("page_start", "?")
            page_end = hit.metadata.get("page_end", "?")
            lines.append(
                f"### chunk_id={hit.chunk_id} "
                f"(section: {section}, p.{page_start}-{page_end}, "
                f"similarity={hit.similarity:.2f})\n"
                f"{hit.document.strip()}\n"
            )
        body = "\n".join(lines)
    footer = (
        "\nReturn a single JSON object matching the schema in the system "
        "prompt. Cite only chunk_ids that appear in the evidence section above."
    )
    return header + body + footer


def _validate_and_repair(
    assessment: ControlAssessment,
    control: Control,
    evidence: list[QueryHit],
) -> ControlAssessment:
    """Strip fabricated evidence and coerce impossible combinations."""
    valid_ids = {hit.chunk_id for hit in evidence}
    kept: list[EvidenceSpan] = []
    dropped: list[str] = []
    for span in assessment.evidence:
        if span.chunk_id in valid_ids:
            kept.append(span)
        else:
            dropped.append(span.chunk_id)
    if dropped:
        logger.warning(
            "Dropping fabricated chunk_ids from assessment for %s: %s",
            control.id,
            dropped,
        )

    # Normalise control_id — a local model occasionally echoes the wrong id.
    normalised_control_id = control.id

    coverage = assessment.coverage
    reasoning = assessment.reasoning
    confidence = assessment.confidence
    if coverage in {"covered", "partial"} and not kept:
        # Claim of coverage with zero surviving citations is not defensible.
        logger.warning(
            "Coverage=%s with no valid citations for %s; coercing to not_covered",
            coverage,
            control.id,
        )
        coverage = "not_covered"
        confidence = "low"
        reasoning = (
            f"{reasoning}\n\n[post-validation] Original verdict was "
            f"'{assessment.coverage}' but no cited evidence survived chunk_id "
            "validation."
        )

    return ControlAssessment(
        control_id=normalised_control_id,
        coverage=coverage,
        evidence=kept if coverage != "not_covered" else [],
        reasoning=reasoning,
        confidence=confidence,
    )
