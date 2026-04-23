"""Tests for the assessment node — uses a fake LLM so no Ollama is required.

The fake LLM is a drop-in for ``ChatOllama.invoke``: it returns a canned
``AIMessage`` whose ``content`` is a JSON string. This lets us exercise the
full ``call_json`` path (JSON parsing, validation, post-repair) without a
running model.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from ai_auditor.graph.nodes.assessment import assess_control
from ai_auditor.models import Control
from ai_auditor.vector_store import QueryHit


class FakeLLM:
    """Canned-response stand-in for a ``ChatOllama`` with JSON mode."""

    def __init__(self, responses: list[str | dict[str, Any]]):
        self._responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self._idx = 0
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        if self._idx >= len(self._responses):
            raise AssertionError("FakeLLM ran out of canned responses")
        out = self._responses[self._idx]
        self._idx += 1
        return AIMessage(content=out)


def _control(cid: str = "A.5.15") -> Control:
    return Control(
        id=cid,
        title="Access control",
        theme="Organizational",
        description="Rules for access to information and associated assets.",
    )


def _hit(
    chunk_id: str,
    *,
    section_id: str = "s_01",
    text: str = "Access is reviewed quarterly.",
    similarity: float = 0.87,
) -> QueryHit:
    return QueryHit(
        chunk_id=chunk_id,
        similarity=similarity,
        metadata={
            "section_id": section_id,
            "section_heading": "Access",
            "page_start": 1,
            "page_end": 1,
        },
        document=text,
    )


def test_assessment_parses_covered_response() -> None:
    evidence = [_hit("c_0001", section_id="s_01"), _hit("c_0002", section_id="s_02")]
    llm = FakeLLM(
        [
            {
                "control_id": "A.5.15",
                "coverage": "covered",
                "evidence": [
                    {
                        "section_id": "s_01",
                        "relevance_note": "Sets a review cadence.",
                    }
                ],
                "reasoning": "The policy defines access review cadence.",
                "confidence": "high",
            }
        ]
    )
    result = assess_control(_control(), evidence, llm)  # type: ignore[arg-type]
    assert result.coverage == "covered"
    assert result.confidence == "high"
    assert [e.section_id for e in result.evidence] == ["s_01"]


def test_fabricated_section_ids_are_dropped() -> None:
    evidence = [_hit("c_0001", section_id="s_01")]
    llm = FakeLLM(
        [
            {
                "control_id": "A.5.15",
                "coverage": "partial",
                "evidence": [
                    {"section_id": "s_01", "relevance_note": "real"},
                    {"section_id": "s_99", "relevance_note": "fabricated"},
                ],
                "reasoning": "Mixed evidence.",
                "confidence": "medium",
            }
        ]
    )
    result = assess_control(_control(), evidence, llm)  # type: ignore[arg-type]
    assert [e.section_id for e in result.evidence] == ["s_01"]
    assert result.coverage == "partial"


def test_covered_with_only_fabricated_ids_coerced_to_not_covered() -> None:
    evidence = [_hit("c_0001", section_id="s_01")]
    llm = FakeLLM(
        [
            {
                "control_id": "A.5.15",
                "coverage": "covered",
                "evidence": [
                    {"section_id": "s_99", "relevance_note": "fake"},
                ],
                "reasoning": "All evidence fabricated.",
                "confidence": "high",
            }
        ]
    )
    result = assess_control(_control(), evidence, llm)  # type: ignore[arg-type]
    assert result.coverage == "not_covered"
    assert result.evidence == []
    assert result.confidence == "low"
    assert "post-validation" in result.reasoning


def test_not_covered_empty_evidence_ok() -> None:
    llm = FakeLLM(
        [
            {
                "control_id": "A.5.15",
                "coverage": "not_covered",
                "evidence": [],
                "reasoning": "Nothing relevant in the document.",
                "confidence": "medium",
            }
        ]
    )
    result = assess_control(_control(), [], llm)  # type: ignore[arg-type]
    assert result.coverage == "not_covered"
    assert result.evidence == []


def test_retry_on_validation_error() -> None:
    """First response is invalid JSON-for-schema; retry succeeds."""
    evidence = [_hit("c_0001", section_id="s_01")]
    llm = FakeLLM(
        [
            # First attempt: missing required "reasoning" field.
            {
                "control_id": "A.5.15",
                "coverage": "covered",
                "evidence": [],
                "confidence": "high",
            },
            # Retry: well-formed.
            {
                "control_id": "A.5.15",
                "coverage": "not_covered",
                "evidence": [],
                "reasoning": "Corrected response after retry.",
                "confidence": "low",
            },
        ]
    )
    result = assess_control(_control(), evidence, llm)  # type: ignore[arg-type]
    assert result.coverage == "not_covered"
    assert result.reasoning == "Corrected response after retry."
    # Two LLM calls were made (first + retry).
    assert len(llm.calls) == 2


def test_double_failure_raises() -> None:
    llm = FakeLLM([{"bad": "response"}, {"still": "bad"}])
    with pytest.raises(ValidationError):
        assess_control(_control(), [], llm)  # type: ignore[arg-type]
