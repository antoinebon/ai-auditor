"""Bounded ReAct retrieval agent (per-control).

Hand-rolled agent loop using ``BaseChatModel.bind_tools`` + LangChain's tool
message convention. Four tools:

- ``list_sections``   — show the document's table of contents
- ``search_policy``   — semantic search over policy chunks
- ``read_section``    — read a full section verbatim
- ``finalize``        — record the coverage judgment and terminate

The loop is capped at ``max_iterations`` LLM calls. Tracing is handled by
MLflow's LangChain autolog plus an ``@mlflow.trace`` parent span on
``run_retrieval_agent`` — see ``ai_auditor.tracing``. There is no
bespoke on-disk trace writer; inspect traces via ``mlflow ui``.

If the agent hits the iteration cap without calling ``finalize``, or the
model returns text without any tool call, we emit a low-confidence
``not_covered`` assessment with whatever evidence was touched. We never
silently drop into "I don't know" — the caller always gets a
``ControlAssessment``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from importlib import resources
from typing import Any

import mlflow
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_auditor.embedding import Embedder
from ai_auditor.graph.nodes.assessment import finalize_assessment
from ai_auditor.models import (
    Confidence,
    Control,
    ControlAssessment,
    Coverage,
    EvidenceSpan,
    ParsedDocument,
)
from ai_auditor.vector_store import QueryHit, VectorStore

logger = logging.getLogger(__name__)

_AGENT_PROMPT = (
    resources.files("ai_auditor.prompts").joinpath("retrieval_agent.md").read_text(encoding="utf-8")
)

MAX_ITERATIONS_DEFAULT = 6
MIN_SEARCHES_BEFORE_NOT_COVERED = 2


# ---------------------------------------------------------------------------
# Tool argument schemas — LangChain's StructuredTool uses these for the
# function-calling schema exposed to the model.
# ---------------------------------------------------------------------------


class SearchPolicyArgs(BaseModel):
    query: str = Field(description="Natural-language query, 5-12 words, policy-style wording.")
    top_k: int = Field(default=5, ge=1, le=10)


class ReadSectionArgs(BaseModel):
    section_id: str = Field(description="Section id returned by list_sections.")


class ListSectionsArgs(BaseModel):
    pass  # no arguments


class EvidenceArg(BaseModel):
    section_id: str = Field(
        description="Section id you received from search_policy or read_section."
    )
    relevance_note: str = Field(
        description="One sentence explaining why this section supports the verdict."
    )


class FinalizeArgs(BaseModel):
    coverage: str = Field(description="One of: covered, partial, not_covered.")
    evidence: list[EvidenceArg] = Field(
        default_factory=list,
        description=(
            "Cited evidence. Each item pairs a section_id you actually received "
            "with a one-sentence relevance note. Do not invent section_ids."
        ),
    )
    reasoning: str = Field(description="2-4 sentences justifying the coverage judgment.")
    confidence: str = Field(description="One of: low, medium, high.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@mlflow.trace(name="retrieval_agent")
def run_retrieval_agent(
    control: Control,
    document: ParsedDocument,
    embedder: Embedder,
    store: VectorStore,
    llm: BaseChatModel,
    *,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
) -> ControlAssessment:
    """Run the agent loop for ``control`` and return a ``ControlAssessment``.

    The ``@mlflow.trace`` decorator wraps each per-control invocation in a
    parent span tagged with the control id; the underlying LLM and tool
    calls show up as nested spans via ``mlflow.langchain.autolog``. When
    MLflow tracing is disabled the decorator is a no-op.
    """
    mlflow.update_current_trace(tags={"control_id": control.id, "control_title": control.title})
    state = _AgentState(control=control, document=document)
    tools = _build_tools(state, embedder, store)
    llm_with_tools = llm.bind_tools([t for _, t in tools])
    tool_map = dict(tools)

    messages: list[Any] = [
        SystemMessage(content=_AGENT_PROMPT),
        HumanMessage(content=_control_prompt(control)),
    ]

    for _iteration in range(max_iterations):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # Model answered without using tools — nothing more to dispatch.
            break

        finalized = False
        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            call_id = call.get("id", f"call_{_iteration}")

            if name == "finalize":
                state.pending_finalize = FinalizeArgs.model_validate(args)
                finalized = True
                break

            tool = tool_map.get(name)
            result = f"unknown tool: {name}" if tool is None else tool.invoke(args)
            messages.append(ToolMessage(content=result, tool_call_id=call_id))

        if finalized:
            break

    return _build_assessment(state, max_iterations_reached=(state.pending_finalize is None))


# ---------------------------------------------------------------------------
# Internal state + tool construction
# ---------------------------------------------------------------------------


class _AgentState:
    """Mutable per-run state the tools read and write."""

    def __init__(self, control: Control, document: ParsedDocument) -> None:
        self.control = control
        self.document = document
        self.seen_hits: dict[str, QueryHit] = {}
        self.seen_sections: set[str] = set()
        self.search_count = 0
        self.pending_finalize: FinalizeArgs | None = None


def _build_tools(
    state: _AgentState, embedder: Embedder, store: VectorStore
) -> list[tuple[str, StructuredTool]]:
    """Return ``(name, StructuredTool)`` pairs bound to the agent's state."""

    def search_policy(query: str, top_k: int = 5) -> str:
        state.search_count += 1
        [hits] = store.query(embedder.encode([query]), top_k=top_k)
        for h in hits:
            state.seen_hits[h.chunk_id] = h
            sid = h.metadata.get("section_id")
            if isinstance(sid, str):
                state.seen_sections.add(sid)
        payload = [
            {
                "section_id": h.metadata.get("section_id", ""),
                "similarity": round(h.similarity, 3),
                "section_heading": h.metadata.get("section_heading", ""),
                "page_start": h.metadata.get("page_start"),
                "page_end": h.metadata.get("page_end"),
                "text_preview": h.document[:500],
            }
            for h in hits
        ]
        return json.dumps({"hits": payload}, ensure_ascii=False)

    def read_section(section_id: str) -> str:
        for section in state.document.sections:
            if section.id == section_id:
                state.seen_sections.add(section.id)
                return json.dumps(
                    {
                        "section_id": section.id,
                        "heading": section.heading,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                        "text": section.text,
                    },
                    ensure_ascii=False,
                )
        return json.dumps({"error": f"no such section_id: {section_id}"})

    def list_sections() -> str:
        payload = [
            {
                "section_id": s.id,
                "heading": s.heading,
                "level": s.level,
                "pages": [s.page_start, s.page_end],
            }
            for s in state.document.sections
        ]
        return json.dumps({"sections": payload}, ensure_ascii=False)

    def finalize(
        coverage: str,
        evidence: list[dict[str, str]] | None = None,
        reasoning: str = "",
        confidence: str = "low",
    ) -> str:
        # Handled by the outer loop (state.pending_finalize); this is only
        # called when LangChain materialises the tool — not the path we use.
        state.pending_finalize = FinalizeArgs(
            coverage=coverage,
            evidence=[EvidenceArg.model_validate(e) for e in (evidence or [])],
            reasoning=reasoning,
            confidence=confidence,
        )
        return "ok"

    return [
        (
            "list_sections",
            StructuredTool.from_function(
                func=list_sections,
                name="list_sections",
                description="List section IDs, headings, and page ranges of the policy document.",
                args_schema=ListSectionsArgs,
            ),
        ),
        (
            "search_policy",
            StructuredTool.from_function(
                func=search_policy,
                name="search_policy",
                description="Semantic search over policy chunks. Returns chunk ids + text previews.",
                args_schema=SearchPolicyArgs,
            ),
        ),
        (
            "read_section",
            StructuredTool.from_function(
                func=read_section,
                name="read_section",
                description="Read one section's full text given its section_id.",
                args_schema=ReadSectionArgs,
            ),
        ),
        (
            "finalize",
            StructuredTool.from_function(
                func=finalize,
                name="finalize",
                description="Record the coverage judgment and stop the session.",
                args_schema=FinalizeArgs,
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Assessment construction + validation
# ---------------------------------------------------------------------------


def _build_assessment(state: _AgentState, *, max_iterations_reached: bool) -> ControlAssessment:
    if state.pending_finalize is None:
        logger.warning(
            "Agent for %s hit the iteration cap without calling finalize",
            state.control.id,
        )
        return _fallback_assessment(state, reason="no_finalize")

    final = state.pending_finalize
    coverage = _coerce_coverage(final.coverage)
    confidence = _coerce_confidence(final.confidence)
    reasoning = final.reasoning.strip() or "(no reasoning provided)"

    # Enforce min-searches-before-not_covered rule (agent-specific; not
    # relevant to the deterministic path, which has a fixed query budget).
    if coverage == "not_covered" and state.search_count < MIN_SEARCHES_BEFORE_NOT_COVERED:
        logger.warning(
            "Agent for %s finalised not_covered after only %d searches; downgrading confidence",
            state.control.id,
            state.search_count,
        )
        confidence = "low"

    # Build EvidenceSpans straight from the agent's structured finalize
    # args and delegate to the shared finalize_assessment for section-id
    # validation + coverage coercion. Sections touched via read_section
    # are citable because they're in state.seen_sections.
    evidence_spans: list[EvidenceSpan] = [
        EvidenceSpan(section_id=e.section_id, relevance_note=e.relevance_note)
        for e in final.evidence
    ]

    assessment = finalize_assessment(
        control_id=state.control.id,
        coverage=coverage,
        confidence=confidence,
        reasoning=reasoning,
        evidence=evidence_spans,
        valid_section_ids=state.seen_sections,
    )

    if max_iterations_reached:
        assessment = assessment.model_copy(
            update={
                "reasoning": (
                    f"{assessment.reasoning}\n\n"
                    "[meta] Agent hit iteration cap before terminating cleanly."
                )
            }
        )

    return assessment


def _fallback_assessment(state: _AgentState, *, reason: str) -> ControlAssessment:
    return ControlAssessment(
        control_id=state.control.id,
        coverage="not_covered",
        evidence=[],
        reasoning=(
            f"Agent terminated without a structured finalize ({reason}). "
            f"Searches performed: {state.search_count}. "
            "Coverage set to not_covered with low confidence — flag for human review."
        ),
        confidence="low",
    )


def _coerce_coverage(raw: str) -> Coverage:
    value = (raw or "").strip().lower()
    if value == "covered":
        return "covered"
    if value == "partial":
        return "partial"
    if value == "not_covered":
        return "not_covered"
    logger.warning("Unknown coverage value from agent: %r; treating as not_covered", raw)
    return "not_covered"


def _coerce_confidence(raw: str) -> Confidence:
    value = (raw or "").strip().lower()
    if value == "high":
        return "high"
    if value == "medium":
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _control_prompt(control: Control) -> str:
    return (
        f"Control to assess: {control.id} — {control.title}\n"
        f"Theme: {control.theme}\n"
        f"Description:\n{control.description}\n\n"
        "Investigate the policy document using your tools, then call `finalize`."
    )


# ---------------------------------------------------------------------------
# Factory wiring into the graph — used by build.py under --agentic
# ---------------------------------------------------------------------------


def make_agentic_assess_node(
    embedder: Embedder,
    store: VectorStore,
    llm: BaseChatModel,
    *,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
) -> Callable[[dict[str, Any]], dict[str, list[ControlAssessment]]]:
    """Build a per-control node that runs the agentic retrieval path.

    The returned node receives a ``PerControlState``-shaped dict (control
    + parsed document); ``run_retrieval_agent`` traces itself via MLflow
    autolog + ``@mlflow.trace``.
    """

    def node(sub_state: dict[str, Any]) -> dict[str, list[ControlAssessment]]:
        control = sub_state["control"]
        document = sub_state["parsed"]
        assessment = run_retrieval_agent(
            control,
            document,
            embedder,
            store,
            llm,
            max_iterations=max_iterations,
        )
        return {"assessments": [assessment]}

    return node
