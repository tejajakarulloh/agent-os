"""Regression tests for issue #956.

When ``session_indexer=None`` and a message notification has been received,
every subsequent ``sync(reason="search:*")`` call used to perform a full
``_scan_files()`` pass even though the workspace was unchanged.  The pending
session delta was never consumed because the reset condition required
``self._session_indexer is not None``.

These tests verify the fix: drop the ``_session_indexer is not None`` guard
so that a successful no-op session sync (disabled indexing) still consumes
the pending delta, restoring the clean-search fast path.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentos.memory.sync_manager import MemorySyncManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class NoopStore:
    """Minimal store that records calls without side effects."""

    def __init__(self) -> None:
        self.indexed: list[str] = []
        self.removed: list[str] = []

    async def index_file(
        self,
        *,
        path: str,
        content: str,
        source: object,
        mtime: float | None = None,
    ) -> int:
        self.indexed.append(path)
        return 1

    async def remove_file(self, path: str) -> None:
        self.removed.append(path)


@dataclass
class FakeSessionResult:
    indexed: int = 0
    removed: int = 0
    skipped: int = 0


class FakeSessionIndexer:
    """Session indexer that optionally fails N times before succeeding."""

    def __init__(self, *, failures: int = 0) -> None:
        self._failures = failures
        self.sync_calls: int = 0

    async def sync(self, *, force: bool = False) -> FakeSessionResult:
        self.sync_calls += 1
        if self._failures > 0:
            self._failures -= 1
            raise RuntimeError("transient session sync failure")
        return FakeSessionResult()


def _make_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    memory = workspace / "memory"
    memory.mkdir(parents=True)
    (workspace / "MEMORY.md").write_text("root\n", encoding="utf-8")
    return workspace, memory


def _manager(store, workspace, memory, *, session_indexer=None) -> MemorySyncManager:
    return MemorySyncManager(
        store=store,
        workspace_dir=workspace,
        memory_dir=memory,
        session_indexer=session_indexer,
    )


def _spy_scan(manager: MemorySyncManager) -> list[int]:
    """Wrap ``_scan_files`` to count invocations."""
    original = manager._scan_files
    counts: list[int] = [0]

    def counting_scan():
        counts[0] += 1
        return original()

    manager._scan_files = counting_scan  # type: ignore[method-assign]
    return counts


# ---------------------------------------------------------------------------
# No indexer — core regression for #956
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_indexer_search_consumes_delta_after_one_sync(tmp_path):
    """One search-time sync must consume the pending delta."""
    workspace, memory = _make_workspace(tmp_path)
    manager = _manager(NoopStore(), workspace, memory)

    manager.notify_message(20)
    assert manager._delta.has_pending()

    await manager.sync(reason="search:tool")
    assert not manager._delta.has_pending(), "delta should be consumed after sync"


@pytest.mark.asyncio
async def test_no_indexer_repeated_searches_skip_scan(tmp_path):
    """After delta is consumed, unchanged searches must not scan again."""
    workspace, memory = _make_workspace(tmp_path)
    manager = _manager(NoopStore(), workspace, memory)
    scans = _spy_scan(manager)

    manager.notify_message(20)
    await manager.sync(reason="search:tool")
    first_scans = scans[0]

    await manager.sync(reason="search:tool")
    await manager.sync(reason="search:tool")
    assert scans[0] == first_scans, "no extra scans on unchanged searches"


@pytest.mark.asyncio
async def test_no_indexer_all_search_reasons(tmp_path):
    """search, search:tool, and search:control all consume delta."""
    workspace, memory = _make_workspace(tmp_path)

    for reason in ("search", "search:tool", "search:control"):
        manager = _manager(NoopStore(), workspace, memory)
        manager.notify_message(20)
        await manager.sync(reason=reason)
        assert not manager._delta.has_pending(), f"{reason} should consume delta"


@pytest.mark.asyncio
async def test_no_indexer_large_message_above_threshold(tmp_path):
    """A message exceeding the byte threshold is still consumed on search."""
    workspace, memory = _make_workspace(tmp_path)
    manager = _manager(NoopStore(), workspace, memory)

    manager.notify_message(200_000)  # well above 100KB threshold
    assert manager._delta.should_sync()

    await manager.sync(reason="search:tool")
    assert not manager._delta.has_pending()


@pytest.mark.asyncio
async def test_no_indexer_new_messages_retrigger_scan(tmp_path):
    """New messages after a reset must trigger sync again."""
    workspace, memory = _make_workspace(tmp_path)
    manager = _manager(NoopStore(), workspace, memory)
    scans = _spy_scan(manager)

    manager.notify_message(20)
    await manager.sync(reason="search:tool")
    scans_after_first = scans[0]

    # New message -> should scan again
    manager.notify_message(30)
    await manager.sync(reason="search:tool")
    assert scans[0] > scans_after_first, "new message should trigger a new scan"
    assert not manager._delta.has_pending()


@pytest.mark.asyncio
async def test_no_indexer_dirty_state_triggers_scan(tmp_path):
    """mark_dirty() must still trigger sync even without pending delta."""
    workspace, memory = _make_workspace(tmp_path)
    manager = _manager(NoopStore(), workspace, memory)
    scans = _spy_scan(manager)

    # Initial sync to populate mtimes
    await manager.sync(reason="manual")
    scans_after_init = scans[0]

    # Search with nothing pending -- should skip
    await manager.sync(reason="search")
    assert scans[0] == scans_after_init

    # Mark dirty -> search should scan
    manager.mark_dirty()
    await manager.sync(reason="search")
    assert scans[0] > scans_after_init


@pytest.mark.asyncio
async def test_no_indexer_force_triggers_scan(tmp_path):
    """force=True must bypass the clean-search fast path."""
    workspace, memory = _make_workspace(tmp_path)
    manager = _manager(NoopStore(), workspace, memory)
    scans = _spy_scan(manager)

    await manager.sync(reason="manual")
    scans_after_init = scans[0]

    await manager.sync(reason="search:tool", force=True)
    assert scans[0] > scans_after_init


# ---------------------------------------------------------------------------
# With indexer — existing behavior preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_indexer_success_resets_delta(tmp_path):
    """Enabled indexer + successful sync must consume delta."""
    workspace, memory = _make_workspace(tmp_path)
    indexer = FakeSessionIndexer()
    manager = _manager(NoopStore(), workspace, memory, session_indexer=indexer)

    manager.notify_message(20)
    await manager.sync(reason="search:tool")

    assert not manager._delta.has_pending()
    assert indexer.sync_calls == 1


@pytest.mark.asyncio
async def test_with_indexer_failure_retains_delta(tmp_path):
    """Failed session sync must keep delta pending for retry."""
    workspace, memory = _make_workspace(tmp_path)
    indexer = FakeSessionIndexer(failures=1)
    manager = _manager(NoopStore(), workspace, memory, session_indexer=indexer)

    manager.notify_message(20)
    await manager.sync(reason="search:tool")

    assert manager._delta.has_pending(), "delta must stay pending after failure"
    assert manager._dirty is True


@pytest.mark.asyncio
async def test_with_indexer_failure_then_success_consumes(tmp_path):
    """After failure, a successful retry must consume the delta."""
    workspace, memory = _make_workspace(tmp_path)
    indexer = FakeSessionIndexer(failures=1)
    manager = _manager(NoopStore(), workspace, memory, session_indexer=indexer)

    manager.notify_message(20)
    await manager.sync(reason="search:tool")  # fails
    assert manager._delta.has_pending()

    await manager.sync(reason="search:tool")  # succeeds
    assert not manager._delta.has_pending()
    assert indexer.sync_calls == 2


@pytest.mark.asyncio
async def test_session_delta_reason_resets_without_indexer(tmp_path):
    """reason='session-delta' must reset delta even without an indexer."""
    workspace, memory = _make_workspace(tmp_path)
    manager = _manager(NoopStore(), workspace, memory)

    # Exceed threshold so session-delta reason proceeds
    manager.notify_message(200_000)
    assert manager._delta.should_sync()

    await manager.sync(reason="session-delta")
    assert not manager._delta.has_pending()


@pytest.mark.asyncio
async def test_no_delta_no_scan_on_search(tmp_path):
    """Without pending delta or dirty state, search must not scan at all."""
    workspace, memory = _make_workspace(tmp_path)
    manager = _manager(NoopStore(), workspace, memory)

    # Bootstrap: initial sync populates mtimes
    await manager.sync(reason="manual")
    scans = _spy_scan(manager)

    await manager.sync(reason="search")
    await manager.sync(reason="search:tool")
    await manager.sync(reason="search:control")

    assert scans[0] == 0, "no scans should occur without pending delta or dirty"
