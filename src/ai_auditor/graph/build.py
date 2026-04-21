"""Compile the end-to-end gap-analysis graph.

Dependencies (embedder, vector store, LLM, control corpus) are resolved
here once and captured in node closures so the rest of the code stays free
of globals or hidden imports. Tests can override any of them via the
``compile_graph`` keyword arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from ai_auditor.config import Settings
from ai_auditor.embedding import Embedder
from ai_auditor.graph.nodes.agentic_retrieval import make_agentic_assess_node
from ai_auditor.graph.nodes.assessment import make_assess_one_control_node
from ai_auditor.graph.nodes.chunking import chunk_document
from ai_auditor.graph.nodes.embedding import make_embed_chunks_node
from ai_auditor.graph.nodes.parsing import parse_pdf
from ai_auditor.graph.nodes.reporting import make_synthesize_node
from ai_auditor.graph.state import MainState
from ai_auditor.ingestion.control_index import load_controls
from ai_auditor.llm import make_llm
from ai_auditor.models import Control, ParsedDocument, PolicyChunk
from ai_auditor.vector_store import VectorStore


def compile_graph(
    settings: Settings,
    *,
    agentic: bool = False,
    controls: list[Control] | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    assessment_llm: ChatOllama | None = None,
    summary_llm: ChatOllama | None = None,
    audit_trail_path: Path | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build and return the compiled LangGraph.

    Dependency resolution order: caller-supplied kwargs win, otherwise we
    construct defaults from ``settings``. The agentic path uses the
    tool-calling variant of Ollama; the deterministic path uses JSON mode.
    """
    resolved_controls = controls if controls is not None else load_controls(settings.controls_path)
    resolved_embedder = embedder if embedder is not None else Embedder(settings.embedding_model)
    resolved_store = store if store is not None else VectorStore()
    resolved_llm = (
        assessment_llm if assessment_llm is not None else make_llm(settings, json_mode=not agentic)
    )

    assess_node: Any = (
        make_agentic_assess_node(
            resolved_embedder,
            resolved_store,
            resolved_llm,
            audit_trail_path=audit_trail_path,
        )
        if agentic
        else make_assess_one_control_node(resolved_embedder, resolved_store, resolved_llm)
    )

    # State-shaped adapters over the pure functions. Inlined here because
    # they don't close over deps and only exist so the graph can call them
    # with the MainState convention.
    def parse_pdf_node(state: MainState) -> dict[str, ParsedDocument]:
        return {"parsed": parse_pdf(state["document_path"])}

    def chunk_document_node(state: MainState) -> dict[str, list[PolicyChunk]]:
        return {"chunks": chunk_document(state["parsed"])}

    def fan_out(state: MainState) -> list[Send]:
        parsed = state["parsed"]
        return [
            Send("assess_one_control", {"control": c, "parsed": parsed})
            for c in resolved_controls
        ]

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
    builder.add_node("assess_one_control", assess_node)
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
    builder.add_conditional_edges("embed_chunks", fan_out, ["assess_one_control"])
    builder.add_edge("assess_one_control", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()
