"""Tests for the LangGraph agentic retrieval subgraph.

Two things these tests pin down:

1. ``read_section`` contributes to the set of citable sections — the
   motivation of the chunk_id → section_id refactor.
2. The agent's per-span ``relevance_note`` flows through into the final
   ``ControlAssessment`` via the finalize node's structured response.

We stub the embedder, vector store, tool-calling LLM, and JSON-mode LLM
so the tests don't need sentence-transformers or a real Ollama. The
scripted LLMs exercise the ``agent → tools → agent → finalize → END``
path without any tool-invented behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from ai_auditor.graph.nodes.agentic_retrieval import (
    _AgentRun,
    _build_tools,
    run_retrieval_agent,
)
from ai_auditor.models import Control, ParsedDocument, Section
from ai_auditor.vector_store import QueryHit


class _NullEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0]]


class _NullStore:
    def query(self, qs: list[list[float]], top_k: int = 5) -> list[list[QueryHit]]:
        return [[]]


def _control() -> Control:
    return Control(
        id="A.5.15",
        title="Access control",
        theme="Organizational",
        description="Rules for access to information and associated assets.",
    )


def _parsed_document() -> ParsedDocument:
    return ParsedDocument(
        path=Path("fake.pdf"),
        title="Fake policy",
        page_count=2,
        sections=[
            Section(
                id="s_02",
                heading="Access Control",
                level=1,
                page_start=1,
                page_end=2,
                text="All employees receive role-based access reviewed quarterly.",
            ),
        ],
    )


def test_read_section_populates_seen_sections() -> None:
    run = _AgentRun(control=_control(), document=_parsed_document())
    tools = {t.name: t for t in _build_tools(run, _NullEmbedder(), _NullStore())}  # type: ignore[arg-type]
    payload = tools["read_section"].invoke({"section_id": "s_02"})
    body = json.loads(payload)
    assert body["section_id"] == "s_02"
    assert "s_02" in run.seen_sections


class _ReadThenPlainTextLLM:
    """Fake tool-calling LLM: read_section once, then reply with plain text.

    Mimics LangChain's ``.bind_tools`` interface so ``run_retrieval_agent``
    can drive the subgraph without a real model. The second turn returns
    plain text (no tool calls), which routes the subgraph from ``agent``
    to ``finalize``.
    """

    def __init__(self) -> None:
        self._step = 0
        self._bound_tools: list[Any] | None = None

    def bind_tools(self, tools: list[Any]) -> "_ReadThenPlainTextLLM":
        self._bound_tools = tools
        return self

    def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        self._step += 1
        if self._step == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_section",
                        "args": {"section_id": "s_02"},
                        "id": "call_1",
                    }
                ],
            )
        return AIMessage(
            content=(
                "Done investigating. Coverage is covered based on section s_02, "
                "which specifies quarterly access reviews."
            )
        )


class _CannedFinalizeLLM:
    """JSON-mode stub for the finalize node.

    ``call_json`` calls ``invoke`` once and parses the string content as
    JSON against the pydantic schema. Returns a canned response that
    cites the section the agent read.
    """

    def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        return AIMessage(
            content=json.dumps(
                {
                    "coverage": "covered",
                    "evidence": [
                        {
                            "section_id": "s_02",
                            "relevance_note": "Quarterly access reviews.",
                        }
                    ],
                    "reasoning": "Section 2 specifies the review cadence.",
                    "confidence": "medium",
                }
            )
        )


def test_agent_can_cite_section_reached_only_via_read_section() -> None:
    """Read-section-only citations are preserved end-to-end.

    A ``section_id`` that was touched by ``read_section`` but never
    returned by ``search_policy`` must still be a valid citation: the
    agent accumulates it in ``seen_sections`` and ``finalize_assessment``
    keeps it. This also exercises the full subgraph transition
    ``agent → tools → agent → finalize → END``.
    """
    assessment = run_retrieval_agent(
        control=_control(),
        document=_parsed_document(),
        embedder=_NullEmbedder(),  # type: ignore[arg-type]
        store=_NullStore(),  # type: ignore[arg-type]
        llm_tools=_ReadThenPlainTextLLM(),  # type: ignore[arg-type]
        llm_json=_CannedFinalizeLLM(),  # type: ignore[arg-type]
        max_iterations=3,
    )
    assert assessment.coverage == "covered"
    assert len(assessment.evidence) == 1
    span = assessment.evidence[0]
    assert span.section_id == "s_02"
    assert span.relevance_note == "Quarterly access reviews."
