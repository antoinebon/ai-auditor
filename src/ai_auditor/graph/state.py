"""LangGraph state schemas.

We use ``TypedDict`` (not pydantic) for graph states because LangGraph
treats the state dict as the bag that nodes read from and merge into. The
domain-model contents of each field are still pydantic — that validation
lives in ``ai_auditor.models``.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, TypedDict

from ai_auditor.models import (
    Control,
    ControlAssessment,
    ParsedDocument,
    PolicyChunk,
    Report,
)


class MainState(TypedDict, total=False):
    """State of the top-level document-analysis graph.

    ``assessments`` has a list-concatenation reducer so that the parallel
    per-control invocations dispatched by ``Send`` each append their single
    ``ControlAssessment`` and LangGraph merges them automatically.
    """

    document_path: Path
    parsed: ParsedDocument
    chunks: list[PolicyChunk]
    assessments: Annotated[list[ControlAssessment], operator.add]
    report: Report


class PerControlState(TypedDict):
    """Payload dispatched by ``Send`` to each per-control fan-out branch.

    The agentic retrieval path needs access to the parsed document
    (for ``list_sections`` / ``read_section``); the deterministic path
    ignores it.
    """

    control: Control
    parsed: ParsedDocument
