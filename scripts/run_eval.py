"""Run both graph strategies on a set of policy PDFs and compare them.

Produces ``out-eval/metrics.json`` + ``out-eval/report.md`` plus one
subdirectory per (doc, strategy) holding the individual run's JSON +
Markdown report. Logs the full session to MLflow under experiment
``ai-auditor`` unless ``--no-mlflow`` is passed.

Usage::

    uv run python scripts/run_eval.py \
        --docs data/examples/minimal_policy.pdf \
               data/examples/sans_acceptable_use.pdf \
               data/examples/northwestern_infosec_policy.pdf \
        --output out-eval/
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ai_auditor.config import Settings, load_settings
from ai_auditor.evaluation.metrics import (
    AggregateMetrics,
    DocComparison,
    aggregate,
    compare_docs,
)
from ai_auditor.evaluation.mlflow_logger import log_session
from ai_auditor.evaluation.runner import StrategyRun, run_strategy
from ai_auditor.render import write_outputs
from ai_auditor.tracing import init_tracing

app = typer.Typer(
    name="run_eval",
    help="Compare deterministic vs agentic graph strategies on policy PDFs.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    docs: Annotated[
        list[Path],
        typer.Option(
            "--docs",
            "-d",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Policy PDFs to evaluate. Defaults to the three sample PDFs.",
        ),
    ] = [  # noqa: B006  (typer pattern — the list is not mutated)
        Path("data/examples/minimal_policy.pdf"),
        Path("data/examples/sans_acceptable_use.pdf"),
        Path("data/examples/northwestern_infosec_policy.pdf"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to write metrics.json + report.md."),
    ] = Path("out-eval"),
    controls: Annotated[
        Path | None,
        typer.Option(
            "--controls",
            help="Override the control corpus (e.g. iso27001_annex_a_small.yaml).",
        ),
    ] = None,
    mlflow_enabled: Annotated[
        bool,
        typer.Option("--mlflow/--no-mlflow", help="Log the session to MLflow."),
    ] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run both strategies on each doc, compute comparison metrics, write artefacts."""
    _configure_logging(verbose)
    settings = load_settings()
    if controls is not None:
        settings = settings.model_copy(update={"controls_path": controls})

    init_tracing(settings, enabled=mlflow_enabled)
    output.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Eval session[/bold]  docs={len(docs)}  mlflow={mlflow_enabled}")
    console.print(f"  model={settings.ollama_model} @ {settings.ollama_host}")
    console.print(f"  controls={settings.controls_path}")

    runs: list[StrategyRun] = []
    comparisons: list[DocComparison] = []

    for doc in docs:
        console.print(f"\n[bold cyan]▶[/bold cyan] {doc.name}")
        det = run_strategy(doc, agentic=False, settings=settings)
        _write_per_run(det, output)
        console.print(
            f"  deterministic: {det.wall_time_s:5.1f}s  llm={det.n_llm_calls}"
        )
        agt = run_strategy(doc, agentic=True, settings=settings)
        _write_per_run(agt, output)
        console.print(
            f"  agentic:       {agt.wall_time_s:5.1f}s  llm={agt.n_llm_calls}  "
            f"tools={sum(agt.n_tool_calls.values())}"
        )
        runs.extend([det, agt])
        comparisons.append(compare_docs(det.report, agt.report))

    agg = aggregate(comparisons)
    _write_metrics_json(output, runs, comparisons, agg, settings)
    _write_markdown_report(output, runs, comparisons, agg, settings)
    _print_summary(comparisons, runs)

    log_session(
        runs=runs,
        comparisons=comparisons,
        aggregate=agg,
        settings=settings,
        output_dir=output,
        enabled=mlflow_enabled,
    )

    console.print(f"\n[green]wrote[/green] {output / 'metrics.json'}")
    console.print(f"[green]wrote[/green] {output / 'report.md'}")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_per_run(run: StrategyRun, output: Path) -> None:
    per_run_dir = output / run.doc_path.stem / run.strategy
    write_outputs(run.report, per_run_dir)


def _write_metrics_json(
    output: Path,
    runs: Iterable[StrategyRun],
    comparisons: Iterable[DocComparison],
    agg: AggregateMetrics,
    settings: Settings,
) -> None:
    payload = {
        "session": {
            "timestamp": datetime.now(UTC).isoformat(),
            "model": settings.ollama_model,
            "controls_path": str(settings.controls_path),
        },
        "aggregate": {
            "mean_agreement_pct": agg.mean_agreement_pct,
            "mean_kappa": agg.mean_kappa,
            "mean_evidence_jaccard": agg.mean_evidence_jaccard,
        },
        "per_doc": [
            {
                "doc": c.doc_path.name,
                "total_controls": c.total_controls,
                "agreement_pct": c.agreement_pct,
                "cohens_kappa": c.cohens_kappa,
                "evidence_jaccard_mean": c.evidence_jaccard_mean,
                "disagreements": [
                    {"deterministic": d, "agentic": a, "count": n}
                    for (d, a), n in c.disagreement_counts.items()
                ],
            }
            for c in comparisons
        ],
        "runs": [
            {
                "doc": r.doc_path.name,
                "strategy": r.strategy,
                "wall_time_s": r.wall_time_s,
                "n_llm_calls": r.n_llm_calls,
                "n_tool_calls": dict(r.n_tool_calls),
            }
            for r in runs
        ],
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown_report(
    output: Path,
    runs: Iterable[StrategyRun],
    comparisons: Iterable[DocComparison],
    agg: AggregateMetrics,
    settings: Settings,
) -> None:
    runs_list = list(runs)
    comparisons_list = list(comparisons)
    lines: list[str] = []
    lines.append("# Strategy evaluation — deterministic vs agentic")
    lines.append("")
    lines.append(f"- Timestamp: {datetime.now(UTC).isoformat()}")
    lines.append(f"- Model: `{settings.ollama_model}`")
    lines.append(f"- Controls corpus: `{settings.controls_path}`")
    lines.append(f"- Documents evaluated: {len({r.doc_path.name for r in runs_list})}")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Mean agreement: **{agg.mean_agreement_pct:.1%}**")
    lines.append(f"- Mean Cohen's kappa: **{agg.mean_kappa:.3f}**")
    lines.append(f"- Mean evidence Jaccard (where coverage matches): "
                 f"**{agg.mean_evidence_jaccard:.3f}**")
    lines.append("")
    lines.append("## Per-document comparison")
    lines.append("")
    lines.append("| Document | Controls | Agreement | Kappa | Evidence Jaccard |")
    lines.append("| --- | ---:| ---:| ---:| ---:|")
    for c in comparisons_list:
        lines.append(
            f"| {c.doc_path.name} | {c.total_controls} | {c.agreement_pct:.1%} | "
            f"{c.cohens_kappa:.3f} | {c.evidence_jaccard_mean:.3f} |"
        )
    lines.append("")
    lines.append("## Disagreement matrix")
    lines.append("")
    lines.append("Counts of `(deterministic_coverage, agentic_coverage)` pairs where "
                 "the two strategies disagree.")
    lines.append("")
    for c in comparisons_list:
        lines.append(f"### {c.doc_path.name}")
        lines.append("")
        if not c.disagreement_counts:
            lines.append("_No disagreements._")
            lines.append("")
            continue
        lines.append("| deterministic | agentic | count |")
        lines.append("| --- | --- | ---:|")
        for (d, a), n in sorted(
            c.disagreement_counts.items(), key=lambda item: item[1], reverse=True
        ):
            lines.append(f"| {d} | {a} | {n} |")
        lines.append("")
    lines.append("## Performance")
    lines.append("")
    lines.append("| Document | Strategy | Wall (s) | LLM calls | Tool calls |")
    lines.append("| --- | --- | ---:| ---:| ---:|")
    for r in runs_list:
        lines.append(
            f"| {r.doc_path.name} | {r.strategy} | {r.wall_time_s:.1f} | "
            f"{r.n_llm_calls} | {sum(r.n_tool_calls.values())} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(comparisons: list[DocComparison], runs: list[StrategyRun]) -> None:
    table = Table(title="Eval summary", show_lines=False)
    table.add_column("Document")
    table.add_column("Agreement", justify="right")
    table.add_column("Kappa", justify="right")
    table.add_column("det (s)", justify="right", style="green")
    table.add_column("agt (s)", justify="right", style="cyan")
    by_doc: dict[str, dict[str, float]] = {}
    for r in runs:
        by_doc.setdefault(r.doc_path.name, {})[r.strategy] = r.wall_time_s
    for c in comparisons:
        times = by_doc.get(c.doc_path.name, {})
        table.add_row(
            c.doc_path.name,
            f"{c.agreement_pct:.1%}",
            f"{c.cohens_kappa:.3f}",
            f"{times.get('deterministic', 0):.1f}",
            f"{times.get('agentic', 0):.1f}",
        )
    console.print("\n")
    console.print(table)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    app()
