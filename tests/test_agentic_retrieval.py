"""Tests for the bounded ReAct retrieval agent.

Two things these tests pin down:

1. ``read_section`` contributes to the set of citable sections — the
   motivation of the chunk_id → section_id refactor.
2. The agent's per-span ``relevance_note`` flows through into the final
   ``ControlAssessment`` via the new structured ``FinalizeArgs.evidence``.

We stub the embedder + vector store at minimum so the tests don't need
sentence-transformers or a real Ollama.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from ai_auditor.graph.nodes.agentic_retrieval import (
    _AgentState,
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
    state = _AgentState(control=_control(), document=_parsed_document())
    tools = dict(_build_tools(state, _NullEmbedder(), _NullStore()))  # type: ignore[arg-type]
    payload = tools["read_section"].invoke({"section_id": "s_02"})
    body = json.loads(payload)
    assert body["section_id"] == "s_02"
    assert "s_02" in state.seen_sections


class _ReadThenFinalizeLLM:
    """Fake tool-calling LLM: read_section once, then finalize.

    Mimics LangChain's ``.bind_tools`` interface so ``run_retrieval_agent``
    can call ``.invoke`` against it without a real model.
    """

    def __init__(self) -> None:
        self._step = 0
        self._bound_tools: list[Any] | None = None

    def bind_tools(self, tools: list[Any]) -> _ReadThenFinalizeLLM:
        self._bound_tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
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
            content="",
            tool_calls=[
                {
                    "name": "finalize",
                    "args": {
                        "coverage": "covered",
                        "evidence": [
                            {
                                "section_id": "s_02",
                                "relevance_note": "Quarterly access reviews.",
                            }
                        ],
                        "reasoning": "Section 2 specifies the review cadence.",
                        "confidence": "medium",
                    },
                    "id": "call_2",
                }
            ],
        )


def test_agent_can_cite_section_reached_only_via_read_section() -> None:
    """Read-section-only citations are preserved end-to-end.

    Before the refactor the agent could only cite chunks returned by
    ``search_policy``; a ``section_id`` touched by ``read_section`` alone
    would be dropped as "fabricated" in ``finalize_assessment``. This
    test proves the gap is closed.
    """
    assessment = run_retrieval_agent(
        control=_control(),
        document=_parsed_document(),
        embedder=_NullEmbedder(),  # type: ignore[arg-type]
        store=_NullStore(),  # type: ignore[arg-type]
        llm=_ReadThenFinalizeLLM(),  # type: ignore[arg-type]
        max_iterations=3,
    )
    assert assessment.coverage == "covered"
    assert len(assessment.evidence) == 1
    span = assessment.evidence[0]
    assert span.section_id == "s_02"
    assert span.relevance_note == "Quarterly access reviews."
