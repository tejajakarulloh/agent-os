"""Budget-rejection regression tests for the raw tool-result store.

Covers the data-loss bug where ``_prune_to_fit`` ran its eviction loop to
completion before it evaluated the oversized case, so a write that could never
fit destroyed every existing record and was then refused anyway.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentos.engine import agent as agent_module
from agentos.engine.agent import Agent
from agentos.engine.tool_result_store import (
    TOOL_RESULT_META_NAME,
    ToolResultStore,
    ToolResultStoreBudgetError,
)


def _write(
    store: ToolResultStore,
    content: str,
    *,
    tool_use_id: str,
    disk_budget_bytes: int,
    max_bytes: int | None = None,
) -> str:
    record = store.write(
        content,
        tool_use_id=tool_use_id,
        tool_name="x",
        session_id="s1",
        session_key="k",
        agent_id="a",
        max_bytes=max_bytes,
        disk_budget_bytes=disk_budget_bytes,
        retention_seconds=None,
    )
    return record.handle


def _handles_on_disk(root: Path) -> set[str]:
    return {path.parent.name for path in root.rglob(TOOL_RESULT_META_NAME)}


def test_oversized_write_preserves_existing_records(tmp_path: Path) -> None:
    store = ToolResultStore(tmp_path)
    handles = [
        _write(store, f"record-{index}", tool_use_id=f"tu-{index}", disk_budget_bytes=500)
        for index in range(3)
    ]
    assert _handles_on_disk(tmp_path) == set(handles)

    with pytest.raises(ToolResultStoreBudgetError):
        _write(store, "X" * 2000, tool_use_id="tu-big", disk_budget_bytes=500)

    assert _handles_on_disk(tmp_path) == set(handles)
    for index, handle in enumerate(handles):
        assert store.read(handle, session_id="s1").content == f"record-{index}"


def test_oversized_write_does_not_evict_another_session(tmp_path: Path) -> None:
    # The disk budget is global: _iter_records walks every session bucket, so a
    # rejected write in one session used to take unrelated sessions down with it.
    store = ToolResultStore(tmp_path)
    others = [
        store.write(
            f"data-{session}",
            tool_use_id=f"tu-{session}",
            tool_name="x",
            session_id=session,
            session_key="k",
            agent_id="a",
            max_bytes=None,
            disk_budget_bytes=500,
            retention_seconds=None,
        ).handle
        for session in ("session-a", "session-b")
    ]

    with pytest.raises(ToolResultStoreBudgetError):
        store.write(
            "X" * 2000,
            tool_use_id="tu-big",
            tool_name="x",
            session_id="session-a",
            session_key="k",
            agent_id="a",
            max_bytes=None,
            disk_budget_bytes=500,
            retention_seconds=None,
        )

    assert _handles_on_disk(tmp_path) == set(others)
    assert store.read(others[1], session_id="session-b").content == "data-session-b"


def test_per_result_rejection_keeps_existing_records(tmp_path: Path) -> None:
    store = ToolResultStore(tmp_path)
    handle = _write(store, "keep-me", tool_use_id="tu-0", disk_budget_bytes=500)

    with pytest.raises(ToolResultStoreBudgetError, match="per-result budget"):
        _write(
            store,
            "Y" * 50,
            tool_use_id="tu-big",
            disk_budget_bytes=500,
            max_bytes=10,
        )

    assert _handles_on_disk(tmp_path) == {handle}


def test_prune_still_evicts_oldest_when_write_fits(tmp_path: Path) -> None:
    store = ToolResultStore(tmp_path)
    # Three 8-byte records against a 30-byte budget; a 10-byte write fits once
    # the oldest record is evicted, so pruning must still happen.
    handles = [
        _write(store, f"record-{index}", tool_use_id=f"tu-{index}", disk_budget_bytes=30)
        for index in range(3)
    ]

    new_handle = _write(store, "Y" * 10, tool_use_id="tu-new", disk_budget_bytes=30)

    assert _handles_on_disk(tmp_path) == {handles[1], handles[2], new_handle}


def test_write_exactly_at_budget_prunes_and_succeeds(tmp_path: Path) -> None:
    store = ToolResultStore(tmp_path)
    for index in range(3):
        _write(store, f"record-{index}", tool_use_id=f"tu-{index}", disk_budget_bytes=20)

    # incoming == budget is storable, but only after the store is cleared.
    new_handle = _write(store, "Z" * 20, tool_use_id="tu-new", disk_budget_bytes=20)

    assert _handles_on_disk(tmp_path) == {new_handle}


def test_budget_rejection_logs_distinct_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ToolResultStore(tmp_path)
    handle = _write(store, "keep-me", tool_use_id="tu-0", disk_budget_bytes=500)

    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        agent_module,
        "logger",
        SimpleNamespace(
            info=lambda event, **kw: events.append((event, kw)),
            warning=lambda event, **kw: events.append((event, kw)),
        ),
    )

    config = SimpleNamespace(
        tool_result_store_dir=str(tmp_path),
        tool_result_store_session_id="s1",
        tool_result_store_session_key="k",
        tool_result_store_agent_id="a",
        tool_result_store_max_bytes=None,
        tool_result_store_disk_budget_bytes=500,
        tool_result_store_retention_seconds=None,
        metadata={},
    )
    holder = SimpleNamespace(config=config, _session_key="k")

    record = Agent._store_tool_result_snapshot(
        holder,  # type: ignore[arg-type]
        "X" * 2000,
        tool_use_id="tu-big",
        tool_name="x",
    )

    assert record is None
    assert config.metadata["tool_result_store_skips"] == 1
    assert len(events) == 1
    event, fields = events[0]
    assert event == "tool_result_store.budget_rejected"
    assert fields["tool_use_id"] == "tu-big"
    assert "exceeds disk budget" in fields["reason"]
    # The rejected write must not have cost us the record already on disk.
    assert _handles_on_disk(tmp_path) == {handle}
