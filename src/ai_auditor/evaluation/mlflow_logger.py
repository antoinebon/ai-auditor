"""MLflow experiment logging for the eval harness.

Single parent run per session with one nested child per
``(doc, strategy)``. The parent carries aggregate metrics + artefacts;
children carry per-invocation metrics. Autolog traces produced during
the invocations are stored separately in MLflow's trace store — the
nested runs and the traces share the same experiment so the UI lets
you pivot between them.

Everything here is a no-op when ``enabled=False`` so the eval CLI can
run offline (``--no-mlflow``) and produce identical local artefacts.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from ai_auditor.config import Settings
from ai_auditor.evaluation.metrics import AggregateMetrics, DocComparison
from ai_auditor.evaluation.runner import StrategyRun

logger = logging.getLogger(__name__)


def log_session(
    *,
    runs: Iterable[StrategyRun],
    comparisons: Iterable[DocComparison],
    aggregate: AggregateMetrics,
    settings: Settings,
    output_dir: Path,
    enabled: bool = True,
) -> None:
    """Log an eval session as a parent MLflow run with nested children."""
    if not enabled:
        logger.debug("MLflow eval logging disabled")
        return

    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    runs_list = list(runs)
    comparisons_list = list(comparisons)

    with mlflow.start_run(run_name=f"eval-session-{ts}"):
        mlflow.log_params(
            {
                "ollama_model": settings.ollama_model,
                "controls_path": str(settings.controls_path),
                "doc_count": len({r.doc_path.name for r in runs_list}),
                "strategy_count": 2,
            }
        )
        mlflow.log_metrics(
            {
                "mean_agreement_pct": aggregate.mean_agreement_pct,
                "mean_kappa": aggregate.mean_kappa,
                "mean_evidence_jaccard": aggregate.mean_evidence_jaccard,
                **_per_strategy_totals(runs_list),
            }
        )
        for artefact in ("metrics.json", "report.md"):
            path = output_dir / artefact
            if path.exists():
                mlflow.log_artifact(str(path))

        for cmp in comparisons_list:
            mlflow.log_metric(
                f"agreement_pct.{cmp.doc_path.stem}", cmp.agreement_pct
            )
            mlflow.log_metric(f"kappa.{cmp.doc_path.stem}", cmp.cohens_kappa)

        for run in runs_list:
            _log_nested(run)


def _log_nested(run: StrategyRun) -> None:
    name = f"{run.doc_path.stem}:{run.strategy}"
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params({"doc": run.doc_path.name, "strategy": run.strategy})
        mlflow.log_metrics(
            {
                "wall_time_s": run.wall_time_s,
                "n_llm_calls": run.n_llm_calls,
                "total_tool_calls": sum(run.n_tool_calls.values()),
            }
        )
        for tool, count in run.n_tool_calls.items():
            mlflow.log_metric(f"tool_calls.{tool}", count)
        mlflow.log_dict(run.report.model_dump(mode="json"), "report.json")


def _per_strategy_totals(runs: list[StrategyRun]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for strategy in ("deterministic", "agentic"):
        matching = [r for r in runs if r.strategy == strategy]
        if not matching:
            continue
        totals[f"total_wall_time_{strategy}_s"] = sum(r.wall_time_s for r in matching)
        totals[f"total_llm_calls_{strategy}"] = float(sum(r.n_llm_calls for r in matching))
    return totals
