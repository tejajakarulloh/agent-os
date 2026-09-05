"""Every session removal path must drop the process-global runtime state.

``SessionManager.finish`` used to be the only caller of the eviction hook, so
deleting or pruning a session left its spawn-group, routing-history, and
spawn-lock entries behind forever — an unbounded leak on a gateway that runs
for weeks (issue #750).
"""

import pytest
import pytest_asyncio

from agentos.engine.steps.agentos_router import _history_store
from agentos.gateway.subagent_announce import _tracker
from agentos.session.manager import SessionManager
from agentos.session.storage import SessionStorage
from agentos.tools.builtin.sessions import _get_spawn_lock, _spawn_locks


@pytest_asyncio.fixture
async def manager():
    storage = SessionStorage(":memory:")
    await storage.connect()
    yield SessionManager(storage, inject_time_prefix=False)
    await storage.close()


def _seed_runtime_state(session_key: str) -> None:
    _tracker.mark_closed(session_key, "task-1")
    _tracker.mark_woken((session_key, "task-1"))
    _history_store.set(session_key, [{"turn_index": 0}])
    _get_spawn_lock(session_key)


def _runtime_state_present(session_key: str) -> bool:
    return (
        _tracker.is_closed(session_key, "task-1")
        or _tracker.is_woken((session_key, "task-1"))
        or _history_store.get(session_key) is not None
        or session_key in _spawn_locks
    )


@pytest.fixture(autouse=True)
def _clean_runtime_state():
    yield
    for key in list(_spawn_locks):
        _spawn_locks.pop(key, None)
    _history_store.clear()
    _tracker._closed.clear()
    _tracker._woken.clear()


class _RecordingTaskRuntime:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel(self, *, session_key: str, source: str = "", reason: str = "") -> int:
        self.cancelled.append(session_key)
        return 0


@pytest.mark.asyncio
async def test_delete_evicts_runtime_state_and_cancels_tasks(manager):
    key = "agent:main:direct:u1"
    runtime = _RecordingTaskRuntime()
    manager.attach_task_runtime(runtime)
    await manager.create(key)
    _seed_runtime_state(key)

    await manager.delete(key)

    assert not _runtime_state_present(key)
    assert runtime.cancelled == [key]
    assert await manager._storage.get_session(key) is None


@pytest.mark.asyncio
async def test_delete_is_idempotent_after_finish(manager):
    key = "agent:main:direct:u1"
    await manager.create(key)
    _seed_runtime_state(key)

    await manager.finish(key)
    await manager.delete(key)

    assert not _runtime_state_present(key)


@pytest.mark.asyncio
async def test_delete_tolerates_a_task_runtime_without_kwargs(manager):
    """Duck-typed runtimes without source/reason must not block the delete."""

    class _MinimalRuntime:
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        async def cancel(self, *, session_key: str) -> int:
            self.cancelled.append(session_key)
            return 0

    key = "agent:main:direct:u1"
    runtime = _MinimalRuntime()
    manager.attach_task_runtime(runtime)
    await manager.create(key)
    _seed_runtime_state(key)

    await manager.delete(key)

    assert runtime.cancelled == [key]
    assert not _runtime_state_present(key)


@pytest.mark.asyncio
async def test_cap_entries_evicts_runtime_state(manager):
    keys = [f"agent:main:direct:u{i}" for i in range(3)]
    for key in keys:
        await manager.create(key)
        _seed_runtime_state(key)

    deleted = await manager.cap_entries(max_entries=1)

    assert deleted == 2
    remaining = {s.session_key for s in await manager._storage.list_sessions(limit=10)}
    for key in keys:
        assert _runtime_state_present(key) is (key in remaining)


@pytest.mark.asyncio
async def test_prune_stale_evicts_runtime_state(manager):
    stale = "agent:main:direct:stale"
    fresh = "agent:main:direct:fresh"
    node = await manager.create(stale)
    node.updated_at = 1
    await manager._storage.upsert_session(node)
    await manager.create(fresh)
    _seed_runtime_state(stale)
    _seed_runtime_state(fresh)

    pruned = await manager.prune_stale(max_age_ms=1000)

    assert pruned == 1
    assert not _runtime_state_present(stale)
    assert _runtime_state_present(fresh)
    assert await manager._storage.get_session(stale) is None


@pytest.mark.asyncio
async def test_list_stale_session_keys_returns_keys_before_delete(manager):
    node = await manager.create("agent:main:direct:stale")
    node.updated_at = 1
    await manager._storage.upsert_session(node)
    await manager.create("agent:main:direct:fresh")

    assert await manager._storage.list_stale_session_keys(1000) == ["agent:main:direct:stale"]


@pytest.mark.asyncio
async def test_reaper_evicts_runtime_state_for_expired_cron_sessions(manager):
    from agentos.scheduler.reaper import SessionReaper

    key = "cron:job1:run:r1"
    node = await manager.create(key)
    node.updated_at = 1
    await manager._storage.upsert_session(node)
    _seed_runtime_state(key)

    await SessionReaper(manager._storage, retention_seconds=1)._do_reap()

    assert not _runtime_state_present(key)
    assert await manager._storage.get_session(key) is None
