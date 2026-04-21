"""End-to-end integration test against a real generated sample PDF.

Runs the full pipeline — parse_pdf → chunk → embed → fan-out → assess →
synthesize — on ``data/examples/minimal_policy.pdf`` with a ``FakeEmbedder``
and a ``ScriptedLLM`` that declines coverage for every control. The point
is to prove: the graph wires up correctly against a real document, the
stats tally across the real corpus (33 controls), and the fan-out produces
one assessment per control.

We do *not* hit Ollama here. Hitting a live model is something to do
manually during the demo walkthrough, not in an automated test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from ai_auditor.config import Settings
from ai_auditor.graph.build import compile_graph
from ai_auditor.ingestion.control_index import load_controls

REPO_ROOT = Path(__file__).resolve().parents[1]
MINIMAL_PDF = REPO_ROOT / "data" / "examples" / "minimal_policy.pdf"
CORPUS_PATH = REPO_ROOT / "data" / "controls" / "iso27001_annex_a.yaml"


class HashEmbedder:
    """Deterministic 8-dim embedder keyed by text hash — no ML deps needed."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = abs(hash(t))
            out.append([((h >> (8 * i)) & 0xFF) / 255.0 for i in range(8)])
        return out


class AlwaysNotCoveredLLM:
    """Every assessment call returns not_covered so we can count branches."""

    def __init__(self) -> None:
        self.call_count = 0

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.call_count += 1
        user = messages[-1].content
        # Extract the control id out of the rendered prompt.
        control_id = "UNKNOWN"
        for line in user.splitlines():
            if line.startswith("Control: "):
                control_id = line.split()[1]
                break
        return AIMessage(
            content=json.dumps(
                {
                    "control_id": control_id,
                    "coverage": "not_covered",
                    "evidence": [],
                    "reasoning": "Integration-test stub: policy is intentionally thin.",
                    "confidence": "medium",
                }
            )
        )


def test_integration_minimal_policy_runs_end_to_end() -> None:
    assert MINIMAL_PDF.exists(), "Run `uv run python scripts/build_samples.py` first"
    settings = Settings(controls_path=CORPUS_PATH)
    llm = AlwaysNotCoveredLLM()
    bundle = compile_graph(
        settings,
        embedder=HashEmbedder(),  # type: ignore[arg-type]
        assessment_llm=llm,  # type: ignore[arg-type]
        summary_llm=None,
    )
    result = bundle.graph.invoke({"document_path": MINIMAL_PDF})
    report = result["report"]

    corpus = load_controls(CORPUS_PATH)
    # Every control in the corpus should have an assessment.
    assert report.stats.total_controls == len(corpus)
    assert {a.control_id for a in report.assessments} == {c.id for c in corpus}
    # The LLM always declined, so coverage breakdown reflects that.
    assert report.stats.covered == 0
    assert report.stats.partial == 0
    assert report.stats.not_covered == len(corpus)
    # Deterministic summary was used (no summary_llm), reports the right total.
    assert (
        f"{len(corpus)} ISO 27001:2022" in report.summary
        or f"Analysed {len(corpus)}" in report.summary
    )
    # One LLM call per control.
    assert llm.call_count == len(corpus)
