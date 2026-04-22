"""MLflow tracing bootstrap.

Centralises the "set up MLflow so autolog captures LangChain + LangGraph
calls" boilerplate in one place. Called once per CLI invocation from
``cli.analyze`` and ``scripts/run_eval.py``.

Design points:

- **Idempotent.** Safe to call multiple times in the same process —
  ``mlflow.langchain.autolog`` deduplicates its callback registration.
- **Graceful when disabled.** ``enabled=False`` (used by ``--no-mlflow``
  and by tests that don't want any side-effects) makes this a no-op.
- **File-backed by default.** If ``MLFLOW_TRACKING_URI`` is unset, we
  rely on MLflow's default (``./mlruns``) — traces still get captured,
  just locally. Pointing ``MLFLOW_TRACKING_URI`` at ``http://mlflow:5000``
  (Docker compose) or any remote server is a config change, not a code
  change.
"""

from __future__ import annotations

import logging

import mlflow

from ai_auditor.config import Settings

logger = logging.getLogger(__name__)


def init_tracing(settings: Settings, *, enabled: bool = True) -> None:
    """Turn on MLflow autolog for LangChain/LangGraph if ``enabled``."""
    if not enabled:
        logger.debug("MLflow tracing disabled (enabled=False)")
        return
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog(log_traces=True)
    logger.debug(
        "MLflow tracing enabled (tracking_uri=%s, experiment=%s)",
        mlflow.get_tracking_uri(),
        settings.mlflow_experiment,
    )
