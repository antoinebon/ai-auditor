"""Smoke tests for the typer CLI.

We test ``version`` directly, and ``analyze`` by monkey-patching
``compile_graph`` with a fake bundle — the aim is to check wiring (args
parsing, report writing, exit code), not to re-test the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ai_auditor import cli
from ai_auditor.models import ControlAssessment, Report, ReportStats

runner = CliRunner()


def test_version_prints_version() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert "ai-auditor" in result.stdout


@dataclass
class _FakeGraph:
    report: Report

    def invoke(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"report": self.report}


def test_analyze_writes_json_and_md(
    tmp_path: Path,
    sample_policy_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canned_report = Report(
        document_path=sample_policy_pdf,
        document_title="Example",
        analyzed_at=datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC),
        model="qwen2.5:7b-instruct",
        agentic=False,
        assessments=[
            ControlAssessment(
                control_id="A.5.15",
                coverage="partial",
                evidence=[],
                reasoning="...",
                confidence="medium",
            ),
        ],
        summary="Single partial finding.",
        stats=ReportStats(
            total_controls=1,
            covered=0,
            partial=1,
            not_covered=0,
            by_theme={
                "Organizational": {
                    "covered": 0,
                    "partial": 1,
                    "not_covered": 0,
                    "total": 1,
                }
            },
        ),
    )

    def _fake_compile_graph(*_: Any, **__: Any) -> _FakeGraph:
        return _FakeGraph(canned_report)

    monkeypatch.setattr(cli, "compile_graph", _fake_compile_graph)
    monkeypatch.setattr(cli, "make_llm", lambda *_, **__: None)

    out_dir = tmp_path / "out"
    result = runner.invoke(
        cli.app,
        ["analyze", str(sample_policy_pdf), "--output", str(out_dir), "--skip-summary"],
    )
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()
    assert "Coverage by theme" in result.stdout


def test_analyze_rejects_missing_pdf(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["analyze", str(tmp_path / "does-not-exist.pdf")])
    assert result.exit_code != 0
