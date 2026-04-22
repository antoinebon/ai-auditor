"""LangChain callback handler that counts LLM + tool calls and durations.

The eval harness needs numeric counters (how many LLM calls did each
strategy make, how long did they take, which tools did the agent hit) to
feed into MLflow as run metrics. MLflow autolog gives us the trace tree
but not easy aggregates — this handler fills that gap.

Attached to the graph via ``graph.invoke(input, config={"callbacks":
[handler]})``. Coexists with autolog (they're independent
BaseCallbackHandler instances, both called on every LLM/tool event).
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class EvalCallbackHandler(BaseCallbackHandler):
    """Counts LLM + tool invocations and tracks per-call wall durations."""

    def __init__(self) -> None:
        self.n_llm_calls: int = 0
        self.n_tool_calls: dict[str, int] = defaultdict(int)
        self.llm_durations_s: list[float] = []
        self.tool_durations_s: dict[str, list[float]] = defaultdict(list)
        self._llm_start_times: dict[UUID, float] = {}
        self._tool_start_times: dict[UUID, tuple[str, float]] = {}

    # --- LLM events --------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.n_llm_calls += 1
        self._llm_start_times[run_id] = time.perf_counter()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # Chat models use on_chat_model_start rather than on_llm_start; count
        # both so the handler works regardless of which path the model takes.
        self.n_llm_calls += 1
        self._llm_start_times[run_id] = time.perf_counter()

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._llm_start_times.pop(run_id, None)
        if start is not None:
            self.llm_durations_s.append(time.perf_counter() - start)

    # --- Tool events -------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "unknown")
        self.n_tool_calls[name] += 1
        self._tool_start_times[run_id] = (name, time.perf_counter())

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        entry = self._tool_start_times.pop(run_id, None)
        if entry is not None:
            name, start = entry
            self.tool_durations_s[name].append(time.perf_counter() - start)

    # --- Aggregates --------------------------------------------------------

    @property
    def total_tool_calls(self) -> int:
        return sum(self.n_tool_calls.values())

    @property
    def mean_llm_duration_s(self) -> float:
        if not self.llm_durations_s:
            return 0.0
        return sum(self.llm_durations_s) / len(self.llm_durations_s)
