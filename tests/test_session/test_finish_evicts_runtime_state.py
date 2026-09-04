"""SessionManager.finish drops module-level subagent + routing bookkeeping."""

from __future__ import annotations

import pytest

from agentos.engine.steps.agentos_router import _history_store
from agentos.gateway.subagent_announce import _tracker
from agentos.session.manager import SessionManager
from agentos.session.models import SessionStatus


class _MemoryStorage:
    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}

    async def get_session(self, session_key: str):
        return self._sessions.get(session_key)

    async def upsert_session(self, node) -> None:
        self._sessions[node.session_key] = node


@pytest.mark.asyncio
async def test_finish_evicts_spawn_group_tracker_and_routing_history() -> None:
    from agentos.session.models import SessionNode

    storage = _MemoryStorage()
    node = SessionNode(
        session_key="agent:main:main",
        session_id="abc",
        agent_id="main",
        created_at=1,
        updated_at=1,
        started_at=1,
        status=SessionStatus.RUNNING,
    )
    await storage.upsert_session(node)

    _tracker.mark_closed("agent:main:main", "task-X")
    _history_store.set("agent:main:main", [{"turn_index": 0}])
    assert _tracker.is_closed("agent:main:main", "task-X")
    assert _history_store.get("agent:main:main") is not None

    mgr = SessionManager(storage)  # type: ignore[arg-type]
    await mgr.finish("agent:main:main", status=SessionStatus.DONE)

    assert not _tracker.is_closed("agent:main:main", "task-X")
    assert _history_store.get("agent:main:main") is None


@pytest.mark.asyncio
async def test_delete_evicts_spawn_group_tracker_and_routing_history() -> None:
    from agentos.session.models import SessionNode

    class _DeleteMemoryStorage(_MemoryStorage):
        def __init__(self) -> None:
            super().__init__()
            self.deleted_keys: list[str] = []

        async def delete_session(self, session_key: str) -> bool:
            self.deleted_keys.append(session_key)
            self._sessions.pop(session_key, None)
            return True

    storage = _DeleteMemoryStorage()
    node = SessionNode(
        session_key="agent:main:chat-1",
        session_id="abc-1",
        agent_id="main",
        created_at=1,
        updated_at=1,
        started_at=1,
        status=SessionStatus.RUNNING,
    )
    await storage.upsert_session(node)

    _tracker.mark_closed("agent:main:chat-1", "task-1")
    _history_store.set("agent:main:chat-1", [{"turn_index": 0}])
    assert _tracker.is_closed("agent:main:chat-1", "task-1")
    assert _history_store.get("agent:main:chat-1") is not None

    mgr = SessionManager(storage)  # type: ignore[arg-type]
    await mgr.delete("agent:main:chat-1")

    assert not _tracker.is_closed("agent:main:chat-1", "task-1")
    assert _history_store.get("agent:main:chat-1") is None
    assert "agent:main:chat-1" in storage.deleted_keys


@pytest.mark.asyncio
async def test_cap_entries_and_prune_stale_evict_runtime_state() -> None:
    from agentos.session.models import SessionNode

    class _MaintenanceStorage(_MemoryStorage):
        def __init__(self) -> None:
            super().__init__()
            self.deleted_keys: list[str] = []

        async def count_sessions(self) -> int:
            return len(self._sessions)

        async def list_sessions(self, limit: int | None = None) -> list[SessionNode]:
            return list(self._sessions.values())  # type: ignore[return-value]

        async def delete_session(self, session_key: str) -> bool:
            self.deleted_keys.append(session_key)
            self._sessions.pop(session_key, None)
            return True

        async def prune_stale_sessions(self, before_ms: int) -> list[str]:
            stale = [k for k, s in self._sessions.items() if s.updated_at < before_ms]  # type: ignore[attr-defined]
            for k in stale:
                await self.delete_session(k)
            return stale

    storage = _MaintenanceStorage()
    for i in range(3):
        node = SessionNode(
            session_key=f"agent:main:s-{i}",
            session_id=f"id-{i}",
            agent_id="main",
            created_at=i * 100,
            updated_at=i * 100,
            started_at=i * 100,
            status=SessionStatus.RUNNING,
        )
        await storage.upsert_session(node)
        _tracker.mark_closed(f"agent:main:s-{i}", f"task-{i}")
        _history_store.set(f"agent:main:s-{i}", [{"turn_index": 0}])

    mgr = SessionManager(storage)  # type: ignore[arg-type]

    # Cap to 2: oldest (s-0) should be deleted and evicted
    deleted_count = await mgr.cap_entries(max_entries=2)
    assert deleted_count == 1
    assert not _tracker.is_closed("agent:main:s-0", "task-0")
    assert _history_store.get("agent:main:s-0") is None
    assert _tracker.is_closed("agent:main:s-1", "task-1")

    # Prune stale: s-1 has updated_at 100, if cutoff is 150 it should be pruned and evicted
    pruned = await mgr.prune_stale(max_age_ms=0)  # max_age_ms=0 prunes all remaining
    assert pruned == 2
    assert not _tracker.is_closed("agent:main:s-1", "task-1")
    assert not _tracker.is_closed("agent:main:s-2", "task-2")
