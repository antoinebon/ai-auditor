"""Populate ``Control.queries`` by prompting the LLM per control.

The deterministic retrieval path multi-queries the vector store with
the list of phrasings stored under each control's ``queries`` field
in the YAML. When that list is empty, retrieval falls back to a single
``title — description`` query, which is long and vague and tends to
average into a generic embedding. This script generates 3-5 short,
policy-style paraphrases per control (via the prompt template at
``src/ai_auditor/prompts/query_generation.md``) and writes them back
to the YAML.

Usage::

    uv run python scripts/generate_queries.py                       # default corpus
    uv run python scripts/generate_queries.py --only A.5.15 A.5.24  # specific ids
    uv run python scripts/generate_queries.py --force               # regenerate all
    uv run python scripts/generate_queries.py --dry-run             # print, don't write

The script reuses ``call_json`` so the one-retry JSON-validation
behaviour applies here too.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, Field
from rich.console import Console

from ai_auditor.config import load_settings
from ai_auditor.ingestion.control_index import dump_controls, load_controls
from ai_auditor.llm import call_json, make_llm
from ai_auditor.models import Control

app = typer.Typer(
    name="generate_queries",
    help="Populate Control.queries with LLM-generated retrieval phrasings.",
    add_completion=False,
)
console = Console()

_QUERY_PROMPT = (
    resources.files("ai_auditor.prompts")
    .joinpath("query_generation.md")
    .read_text(encoding="utf-8")
)


class _QueryResponse(BaseModel):
    """Validated shape of the LLM's response."""

    queries: list[str] = Field(min_length=3, max_length=6)


@app.command()
def main(
    controls_path: Annotated[
        Path,
        typer.Option(
            "--controls",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="YAML corpus to update. Defaults to Settings.controls_path.",
        ),
    ] = Path("data/controls/iso27001_annex_a.yaml"),
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Generate only for these control ids (repeatable). Default: all.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Regenerate even for controls that already have queries."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the generated queries instead of writing back."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate retrieval phrasings for each control and write the YAML back."""
    _configure_logging(verbose)
    settings = load_settings()
    llm = make_llm(settings, json_mode=True, temperature=0.2)

    controls = load_controls(controls_path)
    targeted = _select_targets(controls, only=only, force=force)

    console.print(f"[bold]Generating queries[/bold]  model={settings.ollama_model}")
    console.print(f"  corpus={controls_path}  controls={len(targeted)}/{len(controls)}")
    if not targeted:
        console.print("[yellow]Nothing to do.[/yellow] Use --force to regenerate.")
        raise typer.Exit(code=0)

    updated: dict[str, list[str]] = {}
    for control in targeted:
        console.print(f"  [cyan]▶[/cyan] {control.id} — {control.title}")
        queries = _generate_for(control, llm)
        updated[control.id] = queries
        for q in queries:
            console.print(f"      • {q}")

    if dry_run:
        console.print("\n[yellow]--dry-run:[/yellow] not writing any files.")
        raise typer.Exit(code=0)

    new_controls = [
        c.model_copy(update={"queries": updated[c.id]}) if c.id in updated else c for c in controls
    ]
    dump_controls(new_controls, controls_path)
    console.print(f"\n[green]wrote[/green] {controls_path}  ({len(updated)} controls updated)")


def _select_targets(
    controls: list[Control], *, only: list[str] | None, force: bool
) -> list[Control]:
    by_id = {c.id: c for c in controls}
    if only:
        missing = [cid for cid in only if cid not in by_id]
        if missing:
            raise typer.BadParameter(f"Unknown control ids: {', '.join(missing)}")
        candidates = [by_id[cid] for cid in only]
    else:
        candidates = list(controls)
    return [c for c in candidates if force or not c.queries]


def _generate_for(control: Control, llm: object) -> list[str]:
    user = (
        f"Control: {control.id} — {control.title}\n"
        f"Theme: {control.theme}\n"
        f"Description:\n{control.description}\n"
    )
    response = call_json(llm, system=_QUERY_PROMPT, user=user, schema=_QueryResponse)  # type: ignore[arg-type]
    return [q.strip() for q in response.queries if q.strip()]


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    app()
