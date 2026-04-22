"""Tests for the EvalCallbackHandler — counter + duration tracking."""

from __future__ import annotations

from uuid import uuid4

from ai_auditor.evaluation.callbacks import EvalCallbackHandler


def test_counts_llm_calls() -> None:
    handler = EvalCallbackHandler()
    run_id_a = uuid4()
    run_id_b = uuid4()
    handler.on_llm_start({}, [], run_id=run_id_a)
    handler.on_llm_end(None, run_id=run_id_a)
    handler.on_llm_start({}, [], run_id=run_id_b)
    handler.on_llm_end(None, run_id=run_id_b)
    assert handler.n_llm_calls == 2
    assert len(handler.llm_durations_s) == 2


def test_chat_model_start_also_counted() -> None:
    handler = EvalCallbackHandler()
    run_id = uuid4()
    handler.on_chat_model_start({}, [[]], run_id=run_id)
    handler.on_llm_end(None, run_id=run_id)
    assert handler.n_llm_calls == 1


def test_counts_tool_calls_by_name() -> None:
    handler = EvalCallbackHandler()
    run_id_search = uuid4()
    run_id_read = uuid4()
    handler.on_tool_start({"name": "search_policy"}, "...", run_id=run_id_search)
    handler.on_tool_end("...", run_id=run_id_search)
    handler.on_tool_start({"name": "read_section"}, "...", run_id=run_id_read)
    handler.on_tool_end("...", run_id=run_id_read)
    handler.on_tool_start({"name": "search_policy"}, "...", run_id=uuid4())
    assert handler.n_tool_calls["search_policy"] == 2
    assert handler.n_tool_calls["read_section"] == 1
    assert handler.total_tool_calls == 3


def test_tool_without_name_bucketed_as_unknown() -> None:
    handler = EvalCallbackHandler()
    handler.on_tool_start({}, "...", run_id=uuid4())
    assert handler.n_tool_calls["unknown"] == 1


def test_mean_llm_duration_returns_zero_when_no_calls() -> None:
    handler = EvalCallbackHandler()
    assert handler.mean_llm_duration_s == 0.0


def test_mean_llm_duration_computed_over_finished_calls() -> None:
    handler = EvalCallbackHandler()
    # Fake three durations by populating the list directly — the real path
    # relies on monotonic time which is annoying to stub.
    handler.llm_durations_s = [0.1, 0.2, 0.3]
    import pytest

    assert handler.mean_llm_duration_s == pytest.approx(0.2)
