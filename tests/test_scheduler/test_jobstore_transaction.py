"""Tests for JobStore transaction serialization, rollback, and reentrancy."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentos.scheduler.persistence import JobStore
from agentos.scheduler.types import CronJob


def _dummy_job(job_id: str, name: str = "Test Job") -> CronJob:
    return CronJob(
        id=job_id,
        name=name,
        cron_expr="*/5 * * * *",
        schedule_raw="*/5 * * * *",
        handler_key="agent_run",
    )


@pytest.mark.asyncio
async def test_transaction_commits_all_jobs_on_success(tmp_path: Path) -> None:
    store = JobStore(str(tmp_path / "test.db"))
    await store.open()
    try:
        async with store.transaction():
            await store.save_no_commit(_dummy_job("job-1", "Job One"))
            await store.save_no_commit(_dummy_job("job-2", "Job Two"))

        assert await store.get("job-1") is not None
        assert await store.get("job-2") is not None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_transaction_rolls_back_on_exception(tmp_path: Path) -> None:
    store = JobStore(str(tmp_path / "test.db"))
    await store.open()
    try:
        with pytest.raises(RuntimeError, match="abort"):
            async with store.transaction():
                await store.save_no_commit(_dummy_job("job-rollback", "Rollback Job"))
                raise RuntimeError("abort")

        # The uncommitted job must not exist in the store
        assert await store.get("job-rollback") is None

        # Subsequent independent save must not commit the rolled-back job
        await store.save(_dummy_job("job-after", "After Job"))
        assert await store.get("job-after") is not None
        assert await store.get("job-rollback") is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_nested_transaction_reentrant(tmp_path: Path) -> None:
    store = JobStore(str(tmp_path / "test.db"))
    await store.open()
    try:
        async with store.transaction():
            await store.save_no_commit(_dummy_job("outer", "Outer Job"))
            async with store.transaction():
                await store.save_no_commit(_dummy_job("inner", "Inner Job"))

        assert await store.get("outer") is not None
        assert await store.get("inner") is not None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_nested_transaction_rollback(tmp_path: Path) -> None:
    store = JobStore(str(tmp_path / "test.db"))
    await store.open()
    try:
        with pytest.raises(ValueError, match="fail inside"):
            async with store.transaction():
                await store.save_no_commit(_dummy_job("outer", "Outer Job"))
                async with store.transaction():
                    await store.save_no_commit(_dummy_job("inner", "Inner Job"))
                    raise ValueError("fail inside")

        assert await store.get("outer") is None
        assert await store.get("inner") is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_concurrent_save_serialized_with_transaction(tmp_path: Path) -> None:
    store = JobStore(str(tmp_path / "test.db"))
    await store.open()
    try:
        order: list[str] = []

        async def run_transaction():
            async with store.transaction():
                await store.save_no_commit(_dummy_job("tx-job", "Tx Job"))
                order.append("tx_saved")
                await asyncio.sleep(0.05)
                order.append("tx_committing")

        async def run_concurrent_save():
            # Wait briefly to let transaction start first
            await asyncio.sleep(0.01)
            order.append("concurrent_saving")
            await store.save(_dummy_job("direct-job", "Direct Job"))
            order.append("concurrent_saved")

        await asyncio.gather(run_transaction(), run_concurrent_save())

        # Concurrent save must wait until transaction completes
        assert order == ["tx_saved", "concurrent_saving", "tx_committing", "concurrent_saved"]
        assert await store.get("tx-job") is not None
        assert await store.get("direct-job") is not None
    finally:
        await store.close()
