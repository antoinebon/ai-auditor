"""Load the Annex A control corpus from YAML.

Loading uses PyYAML (we validate into pydantic models and discard
formatting). Dumping uses ``ruamel.yaml`` in round-trip mode so header
comments and folded-scalar ``description`` blocks survive the write:
``dump_controls`` reads the existing file, updates only the fields that
changed (today: ``queries``), and writes back with comments and style
intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter
from ruamel.yaml import YAML

from ai_auditor.models import Control

_CONTROLS_ADAPTER = TypeAdapter(list[Control])

_ROUND_TRIP = YAML(typ="rt")
_ROUND_TRIP.preserve_quotes = True
_ROUND_TRIP.width = 80
_ROUND_TRIP.indent(mapping=2, sequence=2, offset=0)


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
    """Write ``controls`` back to ``path`` preserving comments + style.

    Strategy: load the existing YAML with ruamel's round-trip parser to
    capture header comments, block-scalar styles, and key order, then
    update the ``queries`` list in-place on each matching entry. Controls
    that don't appear in the file are appended verbatim. Fields other
    than ``queries`` are left untouched on disk even if they differ in
    the in-memory ``Control`` — this keeps the dumper focused on the one
    mutation ``generate_queries.py`` actually performs.
    """
    if path.exists():
        existing: Any = _ROUND_TRIP.load(path.read_text(encoding="utf-8"))
        if existing is None:
            existing = []
    else:
        existing = []
    by_id = {c.id: c for c in controls}
    seen: set[str] = set()
    for entry in existing:
        cid = entry.get("id")
        if isinstance(cid, str) and cid in by_id:
            entry["queries"] = list(by_id[cid].queries)
            seen.add(cid)
    for c in controls:
        if c.id not in seen:
            existing.append(c.model_dump(mode="json"))
    with path.open("w", encoding="utf-8") as f:
        _ROUND_TRIP.dump(existing, f)
