"""Snapshot cache eviction tests.

Verifies that _memory_snapshots and _bootstrap_snapshots don't grow
without bound in TurnRunner.  Additive guard - existing behavior unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

from agentos.engine.runtime import (
    _SNAPSHOT_CACHE_MAX_ENTRIES,
    TurnRunner,
    _evict_stale_snapshot_entries,
)

# ---------------------------------------------------------------------------
# Unit tests for _evict_stale_snapshot_entries
# ---------------------------------------------------------------------------


class TestEvictStaleSnapshotEntries:
    """Direct unit tests for the eviction function."""

    def test_no_eviction_when_at_max(self) -> None:
        snaps = {f"k{i}": object() for i in range(_SNAPSHOT_CACHE_MAX_ENTRIES)}
        count = _evict_stale_snapshot_entries(snaps)
        assert count == 0
        assert len(snaps) == _SNAPSHOT_CACHE_MAX_ENTRIES

    def test_evicts_oldest_when_over_max(self) -> None:
        snaps = {f"k{i}": object() for i in range(_SNAPSHOT_CACHE_MAX_ENTRIES + 10)}
        count = _evict_stale_snapshot_entries(snaps)
        assert count == 10
        assert len(snaps) == _SNAPSHOT_CACHE_MAX_ENTRIES
        assert "k0" not in snaps  # oldest evicted
        assert "k9" not in snaps
        assert f"k{_SNAPSHOT_CACHE_MAX_ENTRIES - 1}" in snaps  # newest kept

    def test_empty_dict_is_noop(self) -> None:
        count = _evict_stale_snapshot_entries({})
        assert count == 0

    def test_small_dict_is_noop(self) -> None:
        snaps = {"a": object()}
        count = _evict_stale_snapshot_entries(snaps)
        assert count == 0
        assert len(snaps) == 1

    def test_evicts_exact_excess(self) -> None:
        snaps = {f"k{i}": object() for i in range(_SNAPSHOT_CACHE_MAX_ENTRIES + 50)}
        count = _evict_stale_snapshot_entries(snaps)
        assert count == 50
        assert len(snaps) == _SNAPSHOT_CACHE_MAX_ENTRIES

    def test_preserves_remaining_entries(self) -> None:
        snaps = {f"k{i}": object() for i in range(_SNAPSHOT_CACHE_MAX_ENTRIES + 3)}
        count = _evict_stale_snapshot_entries(snaps)
        assert count == 3
        remaining_keys = set(snaps.keys())
        assert len(remaining_keys) == _SNAPSHOT_CACHE_MAX_ENTRIES
        assert "k0" not in remaining_keys
        assert "k2" not in remaining_keys


# ---------------------------------------------------------------------------
# Integration tests: called during boot + refresh
# ---------------------------------------------------------------------------


class TestSnapshotCacheIntegration:
    """Integration tests: memory_snapshots eviction via TurnRunner methods."""

    def test_refresh_memory_snapshot_triggers_eviction(self) -> None:
        runner = TurnRunner(provider_selector=None)
        # Fill past max
        for i in range(_SNAPSHOT_CACHE_MAX_ENTRIES + 10):
            runner._memory_snapshots[(f"agent-{i}", f"session-{i}")] = SimpleNamespace(  # type: ignore[assignment]
                memory_md="",
                daily_notes={},
            )
        # Refresh for one agent -> triggers eviction
        runner.refresh_memory_snapshot("agent-0")
        assert len(runner._memory_snapshots) <= _SNAPSHOT_CACHE_MAX_ENTRIES

    def test_bootstrap_source_write_triggers_eviction(self) -> None:
        from agentos.bootstrap_types import BootstrapFileReport
        from agentos.engine.runtime import BootstrapSnapshot

        runner = TurnRunner(provider_selector=None)
        report = [BootstrapFileReport(filename="USER.md", raw_chars=4, injected_chars=4)]
        bootstrap = BootstrapSnapshot(workspace_files={"USER.md": "x"}, report=report)
        # Fill past max
        for i in range(_SNAPSHOT_CACHE_MAX_ENTRIES + 10):
            runner._bootstrap_snapshots[(f"agent-{i}", f"session-{i}", "full")] = bootstrap
        # Write triggers eviction
        runner._handle_bootstrap_source_write("agent-0", "USER.md")
        assert len(runner._bootstrap_snapshots) <= _SNAPSHOT_CACHE_MAX_ENTRIES

    def test_mixed_snapshots_evict_independently(self) -> None:
        runner = TurnRunner(provider_selector=None)
        from agentos.bootstrap_types import BootstrapFileReport
        from agentos.engine.runtime import BootstrapSnapshot

        report = [BootstrapFileReport(filename="USER.md", raw_chars=4, injected_chars=4)]
        bootstrap = BootstrapSnapshot(workspace_files={"USER.md": "x"}, report=report)

        # Fill both past max
        for i in range(_SNAPSHOT_CACHE_MAX_ENTRIES + 5):
            runner._memory_snapshots[(f"agent-{i}", f"session-{i}")] = SimpleNamespace(  # type: ignore[assignment]
                memory_md="",
                daily_notes={},
            )
            runner._bootstrap_snapshots[(f"agent-{i}", f"session-{i}", "full")] = bootstrap

        # Refresh memory only
        runner.refresh_memory_snapshot("agent-0")
        assert len(runner._memory_snapshots) <= _SNAPSHOT_CACHE_MAX_ENTRIES
        # Bootstrap should still be over (no write triggered yet)
        assert len(runner._bootstrap_snapshots) == _SNAPSHOT_CACHE_MAX_ENTRIES + 5
