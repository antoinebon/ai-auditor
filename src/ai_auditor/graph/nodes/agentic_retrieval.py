"""LangGraph subgraph for agentic retrieval (per-control).

Replaces a former hand-rolled ReAct loop with a three-node subgraph:

- ``agent``     — tool-calling LLM turn; appends the model's response
                  to ``MessagesState.messages``.
- ``tools``     — standard ``ToolNode`` dispatch of ``list_sections``,
                  ``search_policy``, ``read_section``. Finalize is no
                  longer a tool.
- ``finalize``  — one JSON-mode LLM call (via ``call_json``) that turns
                  the investigation transcript into an
                  ``AssessmentResponse`` and writes it to
                  ``structured_response`` in state.

The conditional edge on ``agent`` routes to ``tools`` while the last AI
message carries tool calls and to ``finalize`` once the model answers
with plain text. The outer wrapper (``run_retrieval_agent``) captures
per-invocation mutable state in ``_AgentRun`` via tool closures so
search hits accumulate into ``seen_hits`` / ``seen_sections`` across
turns, then routes the finalize response through the shared
``finalize_assessment`` post-validator.

Iteration cap is enforced by ``config={"recursion_limit": ...}`` at
invoke time. A ``GraphRecursionError`` (or a missing structured
response) yields the low-confidence ``_fallback_assessment``. MLflow
autolog traces the subgraph's inner LLM + tool spans; a single
``@mlflow.trace`` wraps the wrapper so each per-control invocation
shows up as a labelled parent span.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from importlib import resources
from typing import Any, Literal

import mlflow
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from ai_auditor.embedding import Embedder
from ai_auditor.graph.nodes.assessment import finalize_assessment
from ai_auditor.llm import call_json
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
_FINALIZE_PROMPT = (
    resources.files("ai_auditor.prompts").joinpath("agent_finalize.md").read_text(encoding="utf-8")
)

MAX_ITERATIONS_DEFAULT = 6
MIN_SEARCHES_BEFORE_NOT_COVERED = 2
# Each ReAct turn visits two nodes (agent + tools); the finalize tail
# adds two more (agent no-tools + finalize). Keep a small padding so a
# legitimately chatty run does not trip the cap on edge turns.
_RECURSION_LIMIT_PADDING = 4


# ---------------------------------------------------------------------------
# Tool argument + response schemas
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


class AssessmentResponse(BaseModel):
    """Structured terminus of the subgraph — what the ``finalize`` node emits."""

    coverage: str = Field(description="One of: covered, partial, not_covered.")
    evidence: list[EvidenceArg] = Field(default_factory=list)
    reasoning: str = Field(description="2-4 sentences justifying the coverage judgment.")
    confidence: str = Field(description="One of: low, medium, high.")


class AgentState(MessagesState):
    """``MessagesState`` extended with the finalize node's structured output."""

    structured_response: AssessmentResponse | None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@mlflow.trace(name="retrieval_agent")
def run_retrieval_agent(
    control: Control,
    document: ParsedDocument,
    embedder: Embedder,
    store: VectorStore,
    llm_tools: BaseChatModel,
    llm_json: BaseChatModel,
    *,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
) -> ControlAssessment:
    """Compile and run the agent subgraph for ``control``.

    ``llm_tools`` is the tool-calling model used inside the ``agent`` node;
    ``llm_json`` is the JSON-mode model used once in the ``finalize`` node
    to emit a structured assessment. The subgraph is compiled per
    invocation because tools close over per-control mutable state;
    compiling a three-node graph is cheap.
    """
    mlflow.update_current_trace(tags={"control_id": control.id, "control_title": control.title})
    run = _AgentRun(control=control, document=document)
    tools = _build_tools(run, embedder, store)
    subgraph = _build_subgraph(llm_tools.bind_tools(tools), llm_json, tools)

    initial: dict[str, Any] = {
        "messages": [
            SystemMessage(content=_AGENT_PROMPT),
            HumanMessage(content=_control_prompt(control)),
        ],
        "structured_response": None,
    }
    recursion_limit = max_iterations * 2 + _RECURSION_LIMIT_PADDING

    try:
        result = subgraph.invoke(initial, config={"recursion_limit": recursion_limit})
    except GraphRecursionError:
        logger.warning(
            "Agent for %s hit recursion limit (%d); returning fallback",
            control.id,
            recursion_limit,
        )
        return _fallback_assessment(run, reason="recursion_limit")

    response = result.get("structured_response")
    if response is None:
        logger.warning(
            "Agent for %s terminated without a structured_response; returning fallback",
            control.id,
        )
        return _fallback_assessment(run, reason="no_structured_response")

    return _build_assessment_from_response(run, response)


# ---------------------------------------------------------------------------
# Subgraph construction
# ---------------------------------------------------------------------------


def _build_subgraph(
    llm_with_tools: BaseChatModel,
    llm_json: BaseChatModel,
    tools: list[StructuredTool],
) -> Any:
    """Compile the three-node agent subgraph.

    Edges: ``START → agent``, conditional from ``agent`` to either
    ``tools`` (if tool calls present) or ``finalize`` (otherwise),
    ``tools → agent``, ``finalize → END``.
    """

    def agent(state: AgentState) -> dict[str, list[BaseMessage]]:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "finalize"]:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "finalize"

    def finalize(state: AgentState) -> dict[str, Any]:
        transcript = _render_investigation(state["messages"])
        response = call_json(
            llm_json,
            system=_FINALIZE_PROMPT,
            user=transcript,
            schema=AssessmentResponse,
        )
        return {"structured_response": response}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


def _render_investigation(messages: list[BaseMessage]) -> str:
    """Stringify the agent transcript for the finalize LLM.

    ``call_json`` takes a ``user: str`` so the transcript is rendered as
    plain text rather than passed as structured messages. The finalize
    prompt (in ``agent_finalize.md``) is loaded as the system slot.
    """
    lines: list[str] = ["Investigation transcript:", ""]
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, HumanMessage):
            lines.append(f"USER:\n{msg.content}\n")
        elif isinstance(msg, AIMessage):
            if msg.content:
                lines.append(f"ASSISTANT NOTE:\n{msg.content}\n")
            for call in getattr(msg, "tool_calls", None) or []:
                args_json = json.dumps(call.get("args", {}), ensure_ascii=False)
                lines.append(f"ASSISTANT CALLED: {call['name']}({args_json})")
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            lines.append(f"TOOL RESULT ({name}):\n{msg.content}\n")
    lines.append("")
    lines.append("Produce the AssessmentResponse JSON based on this investigation.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-invocation state + tool construction
# ---------------------------------------------------------------------------


class _AgentRun:
    """Mutable per-invocation state the tools accumulate into."""

    def __init__(self, control: Control, document: ParsedDocument) -> None:
        self.control = control
        self.document = document
        self.seen_hits: dict[str, QueryHit] = {}
        self.seen_sections: set[str] = set()
        self.search_count = 0


def _build_tools(
    run: _AgentRun, embedder: Embedder, store: VectorStore
) -> list[StructuredTool]:
    """Return the three investigation tools, each closed over ``run``.

    The ``finalize`` tool is gone — the subgraph's ``finalize`` node
    produces the structured response instead.
    """

    def search_policy(query: str, top_k: int = 5) -> str:
        run.search_count += 1
        [hits] = store.query(embedder.encode([query]), top_k=top_k)
        for h in hits:
            run.seen_hits[h.chunk_id] = h
            sid = h.metadata.get("section_id")
            if isinstance(sid, str):
                run.seen_sections.add(sid)
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
        for section in run.document.sections:
            if section.id == section_id:
                run.seen_sections.add(section.id)
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
            for s in run.document.sections
        ]
        return json.dumps({"sections": payload}, ensure_ascii=False)

    return [
        StructuredTool.from_function(
            func=list_sections,
            name="list_sections",
            description="List section IDs, headings, and page ranges of the policy document.",
            args_schema=ListSectionsArgs,
        ),
        StructuredTool.from_function(
            func=search_policy,
            name="search_policy",
            description="Semantic search over policy chunks. Returns section ids + text previews.",
            args_schema=SearchPolicyArgs,
        ),
        StructuredTool.from_function(
            func=read_section,
            name="read_section",
            description="Read one section's full text given its section_id.",
            args_schema=ReadSectionArgs,
        ),
    ]


# ---------------------------------------------------------------------------
# Assessment construction + validation
# ---------------------------------------------------------------------------


def _build_assessment_from_response(
    run: _AgentRun, response: AssessmentResponse
) -> ControlAssessment:
    coverage = _coerce_coverage(response.coverage)
    confidence = _coerce_confidence(response.confidence)
    reasoning = response.reasoning.strip() or "(no reasoning provided)"

    if coverage == "not_covered" and run.search_count < MIN_SEARCHES_BEFORE_NOT_COVERED:
        logger.warning(
            "Agent for %s finalised not_covered after only %d searches; downgrading confidence",
            run.control.id,
            run.search_count,
        )
        confidence = "low"

    evidence_spans: list[EvidenceSpan] = [
        EvidenceSpan(section_id=e.section_id, relevance_note=e.relevance_note)
        for e in response.evidence
    ]

    return finalize_assessment(
        control_id=run.control.id,
        coverage=coverage,
        confidence=confidence,
        reasoning=reasoning,
        evidence=evidence_spans,
        valid_section_ids=run.seen_sections,
    )


def _fallback_assessment(run: _AgentRun, *, reason: str) -> ControlAssessment:
    return ControlAssessment(
        control_id=run.control.id,
        coverage="not_covered",
        evidence=[],
        reasoning=(
            f"Agent terminated without a structured assessment ({reason}). "
            f"Searches performed: {run.search_count}. "
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
        "Investigate the policy document using your tools. When you have "
        "enough evidence, reply with a short plain-text summary (no more "
        "tool calls) — a follow-up step will record the structured assessment."
    )


# ---------------------------------------------------------------------------
# Factory wiring into the graph — used by build.py under --agentic
# ---------------------------------------------------------------------------


def make_agentic_assess_node(
    embedder: Embedder,
    store: VectorStore,
    llm: BaseChatModel,
    finalize_llm: BaseChatModel,
    *,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
) -> Callable[[dict[str, Any]], dict[str, list[ControlAssessment]]]:
    """Build a per-control node that runs the agentic retrieval subgraph.

    ``llm`` is the tool-calling model used in the ``agent`` node;
    ``finalize_llm`` is the JSON-mode model used once in the ``finalize``
    node to emit the structured assessment. The returned node receives a
    ``PerControlState``-shaped dict (control + parsed document).
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
            finalize_llm,
            max_iterations=max_iterations,
        )
        return {"assessments": [assessment]}

    return node
