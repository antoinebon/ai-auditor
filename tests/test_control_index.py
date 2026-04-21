"""Tests for the controls YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ai_auditor.ingestion.control_index import dump_controls, load_controls

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "controls" / "iso27001_annex_a.yaml"


def test_corpus_loads_and_is_non_trivial() -> None:
    controls = load_controls(CORPUS_PATH)
    assert len(controls) >= 30, "Demo corpus should hold ~30 controls"


def test_corpus_has_all_four_themes() -> None:
    controls = load_controls(CORPUS_PATH)
    themes = {c.theme for c in controls}
    assert themes == {"Organizational", "People", "Physical", "Technological"}


def test_corpus_ids_are_unique() -> None:
    controls = load_controls(CORPUS_PATH)
    ids = [c.id for c in controls]
    assert len(ids) == len(set(ids)), "Duplicate control ids in corpus"


def test_corpus_ids_are_well_formed() -> None:
    controls = load_controls(CORPUS_PATH)
    for c in controls:
        # Annex A identifiers look like A.<theme>.<num> where theme is 5..8
        assert c.id.startswith("A."), f"Bad id: {c.id}"
        theme_num = int(c.id.split(".")[1])
        assert theme_num in {5, 6, 7, 8}, f"Unknown theme number in {c.id}"


def test_loader_rejects_non_list_top_level(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_list: true\n")
    with pytest.raises(ValueError, match="Expected top-level YAML list"):
        load_controls(bad)


def test_loader_rejects_invalid_theme(tmp_path: Path) -> None:
    bad = tmp_path / "bad_theme.yaml"
    bad.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "A.5.1",
                    "title": "x",
                    "theme": "Invented-Theme",
                    "description": "y",
                    "queries": [],
                }
            ]
        )
    )
    with pytest.raises(ValidationError):
        load_controls(bad)


def test_round_trip_through_dump_and_load(tmp_path: Path) -> None:
    controls = load_controls(CORPUS_PATH)
    out = tmp_path / "round_trip.yaml"
    dump_controls(controls, out)
    reloaded = load_controls(out)
    assert reloaded == controls
