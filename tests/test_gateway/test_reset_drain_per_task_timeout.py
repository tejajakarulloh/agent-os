"""Draining runtime tasks on session reset/delete.

Regression coverage for #1538: the final drain loop wrapped the whole ``for`` in
one ``try/except TimeoutError``, so the first task that exceeded the drain
timeout aborted the loop and every remaining active task was never waited on --
reset/delete then went on to touch session storage while those tasks were still
running against it, and the single warning gave no hint of how many were skipped.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from agentos.gateway import rpc_sessions

_SESSION_KEY = "agent:main:drain"


class _FakeTaskRuntime:
    """Runtime whose ``wait`` never returns for the task ids in *hanging*."""

    def __init__(self, task_ids: list[str], hanging: set[str]) -> None:
        self._task_ids = task_ids
        self._hanging = hanging
        self.waited: list[str] = []
        self.cancelled = 0

    async def list(self, session_key: str) -> list[Any]:
        return [SimpleNamespace(task_id=tid, status="running") for tid in self._task_ids]

    async def wait(self, task_id: str) -> None:
        self.waited.append(task_id)
        if task_id in self._hanging:
            await asyncio.Event().wait()

    async def cancel(self, session_key: str) -> int:
        self.cancelled += 1
        return len(self._task_ids)


@pytest.fixture(autouse=True)
def _fast_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rpc_sessions, "_RESET_RUNTIME_SETTLE_SECONDS", 0.01)
    monkeypatch.setattr(rpc_sessions, "_RESET_RUNTIME_CANCEL_DRAIN_SECONDS", 0.02)


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    recorded: list[tuple[str, dict[str, Any]]] = []

    def record(event: str, **kwargs: Any) -> None:
        recorded.append((event, kwargs))

    monkeypatch.setattr(rpc_sessions.log, "warning", record)
    return recorded


async def _drain(runtime: _FakeTaskRuntime) -> None:
    await rpc_sessions._drain_task_runtime_for_session(
        runtime,
        _SESSION_KEY,
        source="sessions_reset",
        reason="session_reset",
        op="reset",
    )


@pytest.mark.asyncio
async def test_drain_waits_for_every_active_task_even_when_an_earlier_one_times_out(
    warnings: list[tuple[str, dict[str, Any]]],
) -> None:
    runtime = _FakeTaskRuntime(["t1", "t2", "t3"], hanging={"t1"})

    await _drain(runtime)

    # The drain loop runs after the settle loop, so each task id is visited
    # twice; what matters is that t2 and t3 were still waited on after t1
    # timed out rather than being stranded.
    assert runtime.waited.count("t2") == 2
    assert runtime.waited.count("t3") == 2


@pytest.mark.asyncio
async def test_drain_timeout_warning_reports_how_many_tasks_did_not_drain(
    warnings: list[tuple[str, dict[str, Any]]],
) -> None:
    runtime = _FakeTaskRuntime(["t1", "t2", "t3"], hanging={"t1", "t3"})

    await _drain(runtime)

    timeouts = [kw for event, kw in warnings if event.endswith("task_runtime_drain_timeout")]
    assert len(timeouts) == 1
    assert timeouts[0]["undrained_tasks"] == 2
    assert timeouts[0]["session_key"] == _SESSION_KEY


@pytest.mark.asyncio
async def test_drain_logs_no_timeout_warning_when_every_task_drains(
    warnings: list[tuple[str, dict[str, Any]]],
) -> None:
    runtime = _FakeTaskRuntime(["t1", "t2"], hanging=set())

    await _drain(runtime)

    assert [event for event, _ in warnings if event.endswith("task_runtime_drain_timeout")] == []


@pytest.mark.asyncio
async def test_drain_still_cancels_before_waiting(
    warnings: list[tuple[str, dict[str, Any]]],
) -> None:
    runtime = _FakeTaskRuntime(["t1"], hanging=set())

    await _drain(runtime)

    assert runtime.cancelled == 1
