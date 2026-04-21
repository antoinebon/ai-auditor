"""End-to-end graph test with real parsing/retrieval and a fake LLM.

We supply a tiny controls corpus (2 controls) and a handful of query
strings that the ``FakeEmbedder`` knows about, so we can run the whole
pipeline — parse_pdf → chunk_document → embed_chunks → fan-out → assess
→ synthesize — against the sample_policy_pdf fixture without touching
Ollama or sentence-transformers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from ai_auditor.config import Settings
from ai_auditor.graph.build import compile_graph
from ai_auditor.models import Control

CHUNK_ID_RE = re.compile(r"chunk_id=(c_\d+)")


class FakeEmbedder:
    """Hashes text to a 4-dim vector so every input gets a deterministic embedding."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            h = hash(text)
            out.append(
                [
                    ((h >> 0) & 0xFF) / 255.0,
                    ((h >> 8) & 0xFF) / 255.0,
                    ((h >> 16) & 0xFF) / 255.0,
                    ((h >> 24) & 0xFF) / 255.0,
                ]
            )
        return out


class ScriptedLLM:
    """Returns a canned ControlAssessment JSON keyed by control_id in the prompt."""

    def __init__(self, per_control: dict[str, dict[str, Any]]) -> None:
        self._per_control = per_control
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        user_content = messages[-1].content
        real_ids = CHUNK_ID_RE.findall(user_content)
        for cid, payload in self._per_control.items():
            if cid in user_content:
                # Inject a real chunk_id into the canned evidence so the
                # assessment node's post-validation doesn't strip it.
                resolved = dict(payload)
                if resolved.get("coverage") in {"covered", "partial"} and real_ids:
                    resolved["evidence"] = [
                        {
                            "chunk_id": real_ids[0],
                            "quote": "<canned for test>",
                            "relevance_note": "<canned for test>",
                        }
                    ]
                return AIMessage(content=json.dumps(resolved))
        # Default: a not_covered stub so the graph always completes.
        return AIMessage(
            content=json.dumps(
                {
                    "control_id": "UNKNOWN",
                    "coverage": "not_covered",
                    "evidence": [],
                    "reasoning": "No rule matched this control id.",
                    "confidence": "low",
                }
            )
        )


def _tiny_controls() -> list[Control]:
    return [
        Control(
            id="A.5.15",
            title="Access control",
            theme="Organizational",
            description="Rules for access to information and associated assets.",
            queries=["role-based access", "access reviewed quarterly"],
        ),
        Control(
            id="A.5.24",
            title="Information security incident management planning and preparation",
            theme="Organizational",
            description="Incident response plans, roles, and escalation.",
            queries=["incident response plan", "security incident escalation"],
        ),
    ]


def test_graph_runs_end_to_end(sample_policy_pdf: Path) -> None:
    settings = Settings()
    fake_llm = ScriptedLLM(
        per_control={
            "A.5.15": {
                "control_id": "A.5.15",
                "coverage": "covered",
                "evidence": [],
                "reasoning": "Evidence supports quarterly access review.",
                "confidence": "medium",
            },
            "A.5.24": {
                "control_id": "A.5.24",
                "coverage": "partial",
                "evidence": [],
                "reasoning": "Incident handling is described but escalation chain is incomplete.",
                "confidence": "medium",
            },
        }
    )
    graph = compile_graph(
        settings,
        controls=_tiny_controls(),
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        assessment_llm=fake_llm,  # type: ignore[arg-type]
        summary_llm=None,  # deterministic summary — no LLM for the summary step
    )
    out = graph.invoke({"document_path": sample_policy_pdf})
    report = out["report"]

    # Both controls produced an assessment, in some order.
    ids = sorted(a.control_id for a in report.assessments)
    assert ids == ["A.5.15", "A.5.24"]

    # Stats tally correctly.
    assert report.stats.total_controls == 2
    assert report.stats.covered == 1
    assert report.stats.partial == 1
    assert report.stats.not_covered == 0

    # Theme breakdown is populated.
    assert "Organizational" in report.stats.by_theme
    assert report.stats.by_theme["Organizational"]["total"] == 2

    # Deterministic summary kicks in when summary_llm is None.
    assert "2 ISO 27001:2022" in report.summary or "Analysed 2" in report.summary


def test_agentic_flag_compiles_graph() -> None:
    """The agentic path compiles even with an empty controls list."""
    settings = Settings()
    graph = compile_graph(
        settings,
        agentic=True,
        controls=[],
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        assessment_llm=ScriptedLLM(per_control={}),  # type: ignore[arg-type]
    )
    assert graph is not None
