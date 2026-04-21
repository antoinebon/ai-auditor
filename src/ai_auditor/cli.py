"""Typer CLI entrypoint for ai-auditor."""

from __future__ import annotations

import typer

from ai_auditor import __version__

app = typer.Typer(
    name="ai-auditor",
    help="ISO 27001:2022 policy gap analyzer.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print version and exit."""
    typer.echo(f"ai-auditor {__version__}")


if __name__ == "__main__":
    app()
