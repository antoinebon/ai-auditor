"""Run one strategy on one document and return a timed ``StrategyRun``.

This is the side-effect-free workhorse: instantiate the graph with the
chosen strategy, invoke it, measure wall time + per-call counters, and
return the whole bundle. MLflow parent/nested runs are the caller's
responsibility (``scripts/run_eval.py``). Autolog traces produced
during ``graph.invoke`` are captured by the MLflow runtime regardless
of whether the caller opens a run — they can be linked later via
``mlflow.search_traces``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ai_auditor.config import Settings
from ai_auditor.evaluation.callbacks import EvalCallbackHandler
from ai_auditor.graph.build import compile_graph
from ai_auditor.llm import make_llm
from ai_auditor.models import Report

Strategy = Literal["deterministic", "agentic"]


@dataclass(frozen=True)
class StrategyRun:
    """A completed (doc, strategy) run and its performance stats."""

    doc_path: Path
    strategy: Strategy
    report: Report
    wall_time_s: float
    n_llm_calls: int
    n_tool_calls: dict[str, int]


def run_strategy(
    doc_path: Path,
    *,
    agentic: bool,
    settings: Settings,
    skip_summary: bool = True,
) -> StrategyRun:
    """Execute one strategy on one document and return its ``StrategyRun``.

    ``skip_summary=True`` is the eval default: the executive-summary LLM
    call isn't part of the strategy comparison, and it's stochastic
    (temperature 0.2), which would add noise if included.
    """
    handler = EvalCallbackHandler()
    summary_llm = None if skip_summary else make_llm(settings, temperature=0.2)
    graph = compile_graph(settings, agentic=agentic, summary_llm=summary_llm)

    start = time.perf_counter()
    result = graph.invoke(
        {"document_path": doc_path},
        config={"callbacks": [handler]},
    )
    wall = time.perf_counter() - start

    report: Report = result["report"]
    strategy: Strategy = "agentic" if agentic else "deterministic"
    return StrategyRun(
        doc_path=doc_path,
        strategy=strategy,
        report=report,
        wall_time_s=wall,
        n_llm_calls=handler.n_llm_calls,
        n_tool_calls=dict(handler.n_tool_calls),
    )
