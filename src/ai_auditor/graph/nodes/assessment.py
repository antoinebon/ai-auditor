"""Assessment node — per-control coverage judgment with structured output.

Takes a ``Control`` plus retrieved evidence hits, builds a prompt that
renders every hit inline with its ``chunk_id``, and asks the LLM to return
a ``ControlAssessment``. We post-filter evidence spans so any fabricated
chunk_ids are dropped, and we downgrade coverage to ``not_covered`` if the
model claims coverage without any surviving citation.

Also hosts ``finalize_assessment`` — the shared post-validation helper that
the agentic retrieval path re-uses — and ``make_assess_one_control_node``,
the factory that composes retrieval + assessment into a single LangGraph
node for the deterministic path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib import resources

from langchain_core.language_models import BaseChatModel

from ai_auditor.embedding import Embedder
from ai_auditor.graph.nodes.retrieval import retrieve_for_control
from ai_auditor.graph.state import PerControlState
from ai_auditor.llm import call_json
from ai_auditor.models import (
    Confidence,
    Control,
    ControlAssessment,
    Coverage,
    EvidenceSpan,
)
from ai_auditor.vector_store import QueryHit, VectorStore

logger = logging.getLogger(__name__)

_ASSESSMENT_PROMPT = (
    resources.files("ai_auditor.prompts").joinpath("assessment.md").read_text(encoding="utf-8")
)


def assess_control(
    control: Control,
    evidence: list[QueryHit],
    llm: BaseChatModel,
) -> ControlAssessment:
    """Produce a ``ControlAssessment`` for ``control`` given ``evidence`` hits."""
    user_prompt = _render_user_prompt(control, evidence)
    raw = call_json(llm, system=_ASSESSMENT_PROMPT, user=user_prompt, schema=ControlAssessment)
    return finalize_assessment(
        control_id=control.id,
        coverage=raw.coverage,
        confidence=raw.confidence,
        reasoning=raw.reasoning,
        evidence=raw.evidence,
        valid_chunk_ids={hit.chunk_id for hit in evidence},
    )


def make_assess_one_control_node(
    embedder: Embedder,
    store: VectorStore,
    llm: BaseChatModel,
) -> Callable[[PerControlState], dict[str, list[ControlAssessment]]]:
    """Closure factory for the deterministic per-control fan-out node.

    Returned callable takes a ``PerControlState`` payload and returns a
    single-element list under ``assessments`` so LangGraph's
    ``operator.add`` reducer on ``MainState.assessments`` merges parallel
    branches automatically.
    """

    def assess_one_control(sub_state: PerControlState) -> dict[str, list[ControlAssessment]]:
        control = sub_state["control"]
        hits = retrieve_for_control(control, embedder, store)
        return {"assessments": [assess_control(control, hits, llm)]}

    return assess_one_control


def finalize_assessment(
    *,
    control_id: str,
    coverage: Coverage,
    confidence: Confidence,
    reasoning: str,
    evidence: list[EvidenceSpan],
    valid_chunk_ids: set[str],
) -> ControlAssessment:
    """Post-validate an LLM-produced (coverage, evidence, …) tuple.

    Two protections apply in order:

    1. Every evidence span whose ``chunk_id`` isn't in ``valid_chunk_ids``
       is dropped — fabricated citations are not defensible.
    2. If the resulting coverage is ``covered`` or ``partial`` but no
       citations survived, coverage is coerced to ``not_covered`` and
       confidence to ``low``, with a note appended to the reasoning so the
       downgrade is auditable.

    Both the deterministic assessment path and the agentic retrieval path
    call this helper.
    """
    kept: list[EvidenceSpan] = []
    dropped: list[str] = []
    for span in evidence:
        if span.chunk_id in valid_chunk_ids:
            kept.append(span)
        else:
            dropped.append(span.chunk_id)
    if dropped:
        logger.warning(
            "Dropping fabricated chunk_ids from assessment for %s: %s",
            control_id,
            dropped,
        )

    if coverage in {"covered", "partial"} and not kept:
        logger.warning(
            "Coverage=%s with no valid citations for %s; coercing to not_covered",
            coverage,
            control_id,
        )
        return ControlAssessment(
            control_id=control_id,
            coverage="not_covered",
            evidence=[],
            reasoning=(
                f"{reasoning}\n\n[post-validation] Original verdict was "
                f"'{coverage}' but no cited evidence survived chunk_id "
                "validation."
            ),
            confidence="low",
        )

    return ControlAssessment(
        control_id=control_id,
        coverage=coverage,
        evidence=kept if coverage != "not_covered" else [],
        reasoning=reasoning,
        confidence=confidence,
    )


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
