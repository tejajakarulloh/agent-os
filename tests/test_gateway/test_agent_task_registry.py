"""Regression tests for issue #1026: replacement task callback race in AgentTaskRegistry."""

from __future__ import annotations

import asyncio

import pytest

from agentos.gateway.agent_tasks import AgentTaskRegistry


@pytest.mark.asyncio
async def test_predecessor_callback_does_not_remove_replacement() -> None:
    """A cancelled predecessor's done callback must not evict the replacement.

    Regression for #1026: register → replace → predecessor callback fires →
    replacement must still be returned by get() and is_running().
    """
    registry = AgentTaskRegistry()
    session_key = "test-session"

    # A long-running predecessor that we'll cancel.
    predecessor_started = asyncio.Event()

    async def predecessor_work() -> None:
        predecessor_started.set()
        await asyncio.sleep(3600)  # will be cancelled

    predecessor = asyncio.create_task(predecessor_work())
    registry.register(session_key, predecessor, cancel_existing=True)
    await predecessor_started.wait()

    # Register a replacement — this cancels the predecessor.
    replacement_started = asyncio.Event()

    async def replacement_work() -> None:
        replacement_started.set()
        await asyncio.sleep(3600)  # stays running

    replacement = asyncio.create_task(replacement_work())
    registry.register(session_key, replacement, cancel_existing=True)
    await replacement_started.wait()

    # Let the predecessor's done callback run.
    await asyncio.sleep(0)  # yield to event loop for callback dispatch
    await asyncio.sleep(0)  # extra yield for safety

    # The replacement must still be tracked.
    assert registry.get(session_key) is replacement
    assert registry.is_running(session_key) is True

    # Cleanup: cancel the replacement so the test doesn't leak.
    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement


@pytest.mark.asyncio
async def test_single_task_cleanup_still_works() -> None:
    """Normal single-task completion must still remove the entry."""
    registry = AgentTaskRegistry()
    session_key = "test-session"

    done_event = asyncio.Event()

    async def short_work() -> None:
        done_event.set()

    task = asyncio.create_task(short_work())
    registry.register(session_key, task, cancel_existing=True)
    await done_event.wait()

    # Let the done callback fire.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert registry.get(session_key) is None
    assert registry.is_running(session_key) is False


@pytest.mark.asyncio
async def test_replacement_cleans_up_after_itself() -> None:
    """After both predecessor and replacement finish, entry is removed."""
    registry = AgentTaskRegistry()
    session_key = "test-session"

    predecessor_started = asyncio.Event()

    async def predecessor_work() -> None:
        predecessor_started.set()
        await asyncio.sleep(3600)

    predecessor = asyncio.create_task(predecessor_work())
    registry.register(session_key, predecessor, cancel_existing=True)
    await predecessor_started.wait()

    # Replace and let predecessor's callback fire.
    async def replacement_work() -> None:
        pass  # finishes immediately

    replacement = asyncio.create_task(replacement_work())
    registry.register(session_key, replacement, cancel_existing=True)

    # Let both callbacks run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Replacement is done → entry removed.
    assert registry.get(session_key) is None
    assert registry.is_running(session_key) is False


@pytest.mark.asyncio
async def test_cancel_existing_false_raises_when_task_running() -> None:
    """cancel_existing=False must raise if a task is still running."""
    registry = AgentTaskRegistry()
    session_key = "test-session"

    started = asyncio.Event()

    async def long_work() -> None:
        started.set()
        await asyncio.sleep(3600)

    task = asyncio.create_task(long_work())
    registry.register(session_key, task, cancel_existing=True)
    await started.wait()

    with pytest.raises(RuntimeError, match="cancel_existing=False"):
        dummy = asyncio.create_task(asyncio.sleep(0))
        registry.register(session_key, dummy, cancel_existing=False)

    # Cleanup.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
