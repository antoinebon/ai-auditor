"""State-shaped wrappers around the pure pipeline functions.

The pure functions (``parse_pdf``, ``chunk_document``, ``assess_control``,
etc.) are free of LangGraph. This module adapts them to the LangGraph
convention: each callable takes a state dict and returns a partial update.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_ollama import ChatOllama

from ai_auditor.embedding import Embedder
from ai_auditor.graph.nodes.assessment import assess_control
from ai_auditor.graph.nodes.chunking import chunk_document
from ai_auditor.graph.nodes.parsing import parse_pdf
from ai_auditor.graph.nodes.retrieval import retrieve_for_control
from ai_auditor.graph.state import MainState, PerControlState
from ai_auditor.models import ControlAssessment, ParsedDocument, PolicyChunk
from ai_auditor.vector_store import VectorStore


def parse_pdf_node(state: MainState) -> dict[str, ParsedDocument]:
    return {"parsed": parse_pdf(state["document_path"])}


def chunk_document_node(state: MainState) -> dict[str, list[PolicyChunk]]:
    parsed = state["parsed"]
    return {"chunks": chunk_document(parsed)}


def make_assess_one_control_node(
    embedder: Embedder,
    store: VectorStore,
    llm: ChatOllama,
) -> Callable[[PerControlState], dict[str, list[ControlAssessment]]]:
    """Closure factory for the per-control fan-out node.

    Returned callable takes a ``PerControlState`` payload (just the control)
    and returns a single-element list under ``assessments`` so LangGraph's
    ``operator.add`` reducer on ``MainState.assessments`` merges parallel
    branches automatically.
    """

    def assess_one_control(sub_state: PerControlState) -> dict[str, list[ControlAssessment]]:
        control = sub_state["control"]
        hits = retrieve_for_control(control, embedder, store)
        assessment = assess_control(control, hits, llm)
        return {"assessments": [assessment]}

    return assess_one_control
