"""Compile the end-to-end gap-analysis graph.

Dependencies (embedder, vector store, LLM, control corpus) are resolved
here once and captured in node closures so the rest of the code stays free
of globals or hidden imports. Tests can override any of them via the
``compile_graph`` keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from ai_auditor.config import Settings
from ai_auditor.embedding import Embedder
from ai_auditor.graph.nodes.embedding import make_embed_chunks_node
from ai_auditor.graph.nodes.orchestration import (
    chunk_document_node,
    make_assess_one_control_node,
    parse_pdf_node,
)
from ai_auditor.graph.nodes.reporting import make_synthesize_node
from ai_auditor.graph.state import MainState
from ai_auditor.ingestion.control_index import load_controls
from ai_auditor.llm import make_llm
from ai_auditor.models import Control
from ai_auditor.vector_store import VectorStore


@dataclass
class GraphBundle:
    """A compiled graph together with the objects it closes over.

    Exposing the underlying dependencies is useful in tests (so we can
    inspect the vector store after a run) and in the CLI (so we can reuse
    the LLM for the executive-summary step instead of opening a new one).
    """

    graph: CompiledStateGraph[Any, Any, Any, Any]
    controls: list[Control]
    embedder: Embedder
    store: VectorStore
    llm: ChatOllama


def compile_graph(
    settings: Settings,
    *,
    agentic: bool = False,
    controls: list[Control] | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    assessment_llm: ChatOllama | None = None,
    summary_llm: ChatOllama | None = None,
) -> GraphBundle:
    """Build the compiled LangGraph and expose the dependencies it uses."""
    if agentic:
        # Wired up in Phase 8: the agentic retrieval branch ships behind a
        # CLI flag, not in this first graph.
        raise NotImplementedError(
            "agentic retrieval is not wired up yet; deterministic path only"
        )

    resolved_controls = controls if controls is not None else load_controls(settings.controls_path)
    resolved_embedder = embedder if embedder is not None else Embedder(settings.embedding_model)
    resolved_store = store if store is not None else VectorStore()
    resolved_llm = (
        assessment_llm if assessment_llm is not None else make_llm(settings, json_mode=True)
    )

    # LangGraph's node-signature generics don't infer our plain callables
    # well (they fall back to Never); we silence mypy on add_node rather
    # than pollute the pure node functions with framework-specific types.
    builder: StateGraph[Any, Any, Any, Any] = StateGraph(MainState)
    builder.add_node("parse_pdf", parse_pdf_node)
    builder.add_node("chunk_document", chunk_document_node)
    builder.add_node(
        "embed_chunks",
        make_embed_chunks_node(resolved_embedder, resolved_store),  # type: ignore[arg-type]
    )
    builder.add_node(
        "assess_one_control",
        make_assess_one_control_node(  # type: ignore[arg-type]
            resolved_embedder, resolved_store, resolved_llm
        ),
    )
    builder.add_node(
        "synthesize",
        make_synthesize_node(  # type: ignore[arg-type]
            settings.ollama_model,
            agentic=agentic,
            summary_llm=summary_llm,
        ),
    )

    builder.add_edge(START, "parse_pdf")
    builder.add_edge("parse_pdf", "chunk_document")
    builder.add_edge("chunk_document", "embed_chunks")
    builder.add_conditional_edges(
        "embed_chunks",
        _make_fan_out(resolved_controls),
        ["assess_one_control"],
    )
    builder.add_edge("assess_one_control", "synthesize")
    builder.add_edge("synthesize", END)

    compiled = builder.compile()
    return GraphBundle(
        graph=compiled,
        controls=resolved_controls,
        embedder=resolved_embedder,
        store=resolved_store,
        llm=resolved_llm,
    )


def _make_fan_out(controls: list[Control]):  # type: ignore[no-untyped-def]
    """Build a closure that returns one ``Send`` per control."""

    def fan_out(_: MainState) -> list[Send]:
        return [Send("assess_one_control", {"control": c}) for c in controls]

    return fan_out
