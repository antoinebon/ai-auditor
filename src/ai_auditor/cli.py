"""Typer CLI entrypoint for ai-auditor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ai_auditor import __version__
from ai_auditor.config import load_settings
from ai_auditor.graph.build import compile_graph
from ai_auditor.llm import make_llm
from ai_auditor.models import Report
from ai_auditor.render import write_outputs
from ai_auditor.tracing import init_tracing

app = typer.Typer(
    name="ai-auditor",
    help="ISO 27001:2022 policy gap analyzer.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.command()
def version() -> None:
    """Print version and exit."""
    typer.echo(f"ai-auditor {__version__}")


@app.command()
def analyze(
    pdf: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the policy PDF to analyse.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where report.json and report.md will be written.",
        ),
    ] = Path("out"),
    agentic: Annotated[
        bool,
        typer.Option(
            "--agentic/--no-agentic",
            help="Use the bounded ReAct retrieval agent instead of multi-query (WIP).",
        ),
    ] = False,
    skip_summary: Annotated[
        bool,
        typer.Option(
            "--skip-summary",
            help="Skip the LLM-generated executive summary; use a deterministic one.",
        ),
    ] = False,
    mlflow_enabled: Annotated[
        bool,
        typer.Option(
            "--mlflow/--no-mlflow",
            help="Enable MLflow tracing (default: on). Traces go to MLFLOW_TRACKING_URI, "
            "or ./mlruns when that env var is empty.",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Analyse a policy PDF against the ISO 27001 Annex A corpus."""
    _configure_logging(verbose)
    settings = load_settings()
    init_tracing(settings, enabled=mlflow_enabled)
    console.print(
        f"[bold]Analysing[/bold] {pdf}\n"
        f"  model={settings.ollama_model} @ {settings.ollama_host}\n"
        f"  controls={settings.controls_path}\n"
        f"  agentic={agentic}  mlflow={mlflow_enabled}\n"
    )
    summary_llm = None if skip_summary else make_llm(settings, temperature=0.2)
    graph = compile_graph(settings, agentic=agentic, summary_llm=summary_llm)
    result = graph.invoke({"document_path": pdf})
    report: Report = result["report"]
    json_path, md_path = write_outputs(report, output)
    _print_summary_table(report)
    console.print(f"\n[green]wrote[/green] {json_path}")
    console.print(f"[green]wrote[/green] {md_path}")


def _print_summary_table(report: Report) -> None:
    table = Table(title="Coverage by theme", show_lines=False)
    table.add_column("Theme")
    table.add_column("Total", justify="right")
    table.add_column("Covered", justify="right", style="green")
    table.add_column("Partial", justify="right", style="yellow")
    table.add_column("Not covered", justify="right", style="red")
    for theme, counts in sorted(report.stats.by_theme.items()):
        table.add_row(
            theme,
            str(counts.get("total", 0)),
            str(counts.get("covered", 0)),
            str(counts.get("partial", 0)),
            str(counts.get("not_covered", 0)),
        )
    table.add_section()
    s = report.stats
    table.add_row(
        "[bold]Total",
        f"[bold]{s.total_controls}",
        f"[bold green]{s.covered}",
        f"[bold yellow]{s.partial}",
        f"[bold red]{s.not_covered}",
    )
    console.print(table)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    app()
