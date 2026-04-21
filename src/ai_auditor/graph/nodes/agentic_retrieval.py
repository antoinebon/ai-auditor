"""Bounded ReAct retrieval agent (per-control).

Hand-rolled agent loop using ``ChatOllama.bind_tools`` + LangChain's tool
message convention. Four tools:

- ``list_sections``   — show the document's table of contents
- ``search_policy``   — semantic search over policy chunks
- ``read_section``    — read a full section verbatim
- ``finalize``        — record the coverage judgment and terminate

The loop is capped at ``max_iterations`` LLM calls. Every step is appended
to an optional JSONL trace so each judgment produces a full reasoning
record — the "defensible audit trail" we'd expect in a compliance tool.

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
from pathlib import Path
from typing import Any, TextIO

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from ai_auditor.embedding import Embedder
from ai_auditor.models import Control, ControlAssessment, EvidenceSpan, ParsedDocument
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


class FinalizeArgs(BaseModel):
    coverage: str = Field(description="One of: covered, partial, not_covered.")
    evidence_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk ids you actually received. Must not be invented.",
    )
    reasoning: str = Field(description="2-4 sentences justifying the coverage judgment.")
    confidence: str = Field(description="One of: low, medium, high.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_retrieval_agent(
    control: Control,
    document: ParsedDocument,
    embedder: Embedder,
    store: VectorStore,
    llm: ChatOllama,
    *,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
    trace_writer: TextIO | None = None,
) -> ControlAssessment:
    """Run the agent loop for ``control`` and return a ``ControlAssessment``."""
    state = _AgentState(control=control, document=document)
    tools = _build_tools(state, embedder, store)
    llm_with_tools = llm.bind_tools([t for _, t in tools])
    tool_map = dict(tools)

    messages: list[Any] = [
        SystemMessage(content=_AGENT_PROMPT),
        HumanMessage(content=_control_prompt(control)),
    ]

    for iteration in range(max_iterations):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)
        _trace(trace_writer, control.id, iteration, "model_response", _summarise_response(response))

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # Model answered without using tools — nudge once, then bail.
            _trace(trace_writer, control.id, iteration, "no_tool_calls", {})
            break

        finalized = False
        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            call_id = call.get("id", f"call_{iteration}")

            if name == "finalize":
                state.pending_finalize = FinalizeArgs.model_validate(args)
                _trace(trace_writer, control.id, iteration, "finalize", args)
                finalized = True
                break

            tool = tool_map.get(name)
            result = f"unknown tool: {name}" if tool is None else tool.invoke(args)
            _trace(
                trace_writer,
                control.id,
                iteration,
                f"tool:{name}",
                {"args": args, "result_preview": _preview(result)},
            )
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
        payload = [
            {
                "chunk_id": h.chunk_id,
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
        evidence_chunk_ids: list[str] | None = None,
        reasoning: str = "",
        confidence: str = "low",
    ) -> str:
        # Handled by the outer loop (state.pending_finalize); this is only
        # called when LangChain materialises the tool — not the path we use.
        state.pending_finalize = FinalizeArgs(
            coverage=coverage,
            evidence_chunk_ids=list(evidence_chunk_ids or []),
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

    # Drop fabricated chunk ids — only cite things the agent actually saw.
    kept: list[EvidenceSpan] = []
    dropped: list[str] = []
    for cid in final.evidence_chunk_ids:
        hit = state.seen_hits.get(cid)
        if hit is None:
            dropped.append(cid)
            continue
        kept.append(
            EvidenceSpan(
                chunk_id=cid,
                quote=hit.document[:300].strip(),
                relevance_note=hit.metadata.get("section_heading", ""),
            )
        )
    if dropped:
        logger.warning(
            "Dropping fabricated chunk_ids from agent finalize for %s: %s",
            state.control.id,
            dropped,
        )

    # Enforce min-searches-before-not_covered rule.
    if coverage == "not_covered" and state.search_count < MIN_SEARCHES_BEFORE_NOT_COVERED:
        logger.warning(
            "Agent for %s finalised not_covered after only %d searches; downgrading confidence",
            state.control.id,
            state.search_count,
        )
        confidence = "low"

    # Coverage/citation consistency: claimed coverage with no surviving evidence → coerce.
    reasoning = final.reasoning.strip() or "(no reasoning provided)"
    if coverage in {"covered", "partial"} and not kept:
        logger.warning(
            "Agent finalised %s for %s without any valid citations; coercing to not_covered",
            coverage,
            state.control.id,
        )
        coverage = "not_covered"
        confidence = "low"
        reasoning = (
            f"{reasoning}\n\n[post-validation] Agent claimed '{final.coverage}' with no "
            "surviving citation after chunk_id validation."
        )

    if max_iterations_reached:
        reasoning = f"{reasoning}\n\n[meta] Agent hit iteration cap before terminating cleanly."

    return ControlAssessment(
        control_id=state.control.id,
        coverage=coverage,
        evidence=kept if coverage != "not_covered" else [],
        reasoning=reasoning,
        confidence=confidence,
    )


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


def _coerce_coverage(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value in {"covered", "partial", "not_covered"}:
        return value
    logger.warning("Unknown coverage value from agent: %r; treating as not_covered", raw)
    return "not_covered"


def _coerce_confidence(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value in {"low", "medium", "high"}:
        return value
    return "low"


# ---------------------------------------------------------------------------
# Prompt construction + trace helpers
# ---------------------------------------------------------------------------


def _control_prompt(control: Control) -> str:
    return (
        f"Control to assess: {control.id} — {control.title}\n"
        f"Theme: {control.theme}\n"
        f"Description:\n{control.description}\n\n"
        "Investigate the policy document using your tools, then call `finalize`."
    )


def _summarise_response(msg: AIMessage) -> dict[str, Any]:
    tool_calls = getattr(msg, "tool_calls", None) or []
    return {
        "tool_calls": [{"name": tc["name"], "args": tc.get("args", {})} for tc in tool_calls],
        "text_preview": _preview(
            msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        ),
    }


def _preview(text: str, limit: int = 240) -> str:
    s = (text or "").strip()
    return s if len(s) <= limit else s[:limit] + "…"


def _trace(
    writer: TextIO | None,
    control_id: str,
    iteration: int,
    action: str,
    payload: Any,
) -> None:
    if writer is None:
        return
    record = {
        "control_id": control_id,
        "iteration": iteration,
        "action": action,
        "payload": payload,
    }
    writer.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    writer.flush()


# ---------------------------------------------------------------------------
# Factory wiring into the graph — used by build.py under --agentic
# ---------------------------------------------------------------------------


def make_agentic_assess_node(
    embedder: Embedder,
    store: VectorStore,
    llm: ChatOllama,
    *,
    audit_trail_path: Path | None = None,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
) -> Callable[[dict[str, Any]], dict[str, list[ControlAssessment]]]:
    """Build a per-control node that runs the agentic retrieval path.

    The returned node receives a ``PerControlState``-shaped dict plus the
    parsed document via closure (it needs the document to expose
    ``list_sections`` / ``read_section``). The caller therefore has to
    thread ``parsed`` into the agent — we do it by making the full
    ``MainState`` visible via a shared reference injected at build time
    through ``document_ref``.
    """
    # Opened lazily so tests that never take the agentic path don't touch
    # the filesystem.
    writer: TextIO | None = None
    if audit_trail_path is not None:
        audit_trail_path.parent.mkdir(parents=True, exist_ok=True)
        writer = audit_trail_path.open("a", encoding="utf-8")

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
            trace_writer=writer,
        )
        return {"assessments": [assessment]}

    return node
