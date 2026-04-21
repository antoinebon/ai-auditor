"""Load the Annex A control corpus from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter

from ai_auditor.models import Control

_CONTROLS_ADAPTER = TypeAdapter(list[Control])


def load_controls(path: Path) -> list[Control]:
    """Parse ``path`` into validated ``Control`` objects.

    Raises a ``pydantic.ValidationError`` with the offending entry's index
    if any control is malformed — fail loudly rather than silently dropping.
    """
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected top-level YAML list in {path}, got {type(raw).__name__}")
    return _CONTROLS_ADAPTER.validate_python(raw)


def dump_controls(controls: list[Control], path: Path) -> None:
    """Write ``controls`` back to YAML (used after query generation)."""
    payload = [c.model_dump(mode="json") for c in controls]
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=80),
        encoding="utf-8",
    )
