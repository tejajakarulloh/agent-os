"""An inline approval that timed out is pending, not a rejection (#1563).

The Web UI polls the approval queue itself, so ``_check_exec_approval`` waits
inline for the operator's click instead of bouncing ``approval_required`` back
to the model. When that wait expired it returned::

    {"status": "approval_denied", ..., "message": "Approval was denied or timed out."}

for anything that was not ``entry.approved`` — collapsing "the human said no"
and "the human is still reading it" into one outcome. ``exec_command`` and
``spawn_command`` act on that status and record
``DenialReason.HUMAN_REJECTED`` in the sandbox ledger, so an operator who took
longer than the window is written into the audit trail as having rejected the
command — and ``post_denial_guard`` reads a recorded denial as a repeat-intent
signal, so the false rejection can auto-deny the agent's honest retry.

These tests assert on **what reached the ledger**, not only on the status the
tool returned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentos.gateway.approval_queue import get_approval_queue, reset_approval_queue
from agentos.sandbox.config import SandboxSettings
from agentos.sandbox.integration import configure_runtime, get_runtime, reset_runtime
from agentos.sandbox.intent_cache import reset_intent_cache
from agentos.tools.builtin import shell
from agentos.tools.envelope import is_denial_payload
from agentos.tools.types import CallerKind, ToolContext, current_tool_context

SESSION_KEY = "agent:main:test-1563"
_WARNED_COMMAND = "rm -rf build"


@pytest.fixture
def web_ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A Web UI caller whose inline approval wait expires immediately."""
    reset_approval_queue()
    reset_intent_cache()
    reset_runtime()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    configure_runtime(
        SandboxSettings(sandbox=True, backend="noop", security_grading=False),
        workspace=workspace,
    )
    monkeypatch.setattr(shell, "_sandbox_effectively_off", lambda: True)
    # The production wait is three minutes; the branch under test is what
    # happens when it expires, so shorten it rather than sleep through it.
    monkeypatch.setattr(shell, "_APPROVAL_RETRY_WAIT_SECONDS", 0.01)
    elevate_token = shell._elevate_current_call.set(False)
    context = ToolContext(
        caller_kind=CallerKind.WEB,
        session_key=SESSION_KEY,
        workspace_dir=str(workspace),
    )
    token = current_tool_context.set(context)
    yield context
    current_tool_context.reset(token)
    shell._elevate_current_call.reset(elevate_token)
    reset_approval_queue()
    reset_intent_cache()
    reset_runtime()


async def _ledger_state() -> tuple[int, object]:
    """(denials recorded for this session, reason of the last one)."""
    runtime = get_runtime()
    assert runtime is not None
    total = await runtime.ledger.count_session(SESSION_KEY)
    _fingerprint, reason = await runtime.ledger.last_denial(SESSION_KEY)
    return total, reason


async def _check() -> dict[str, object] | None:
    return await shell._check_exec_approval(
        tool_name="exec_command",
        command=_WARNED_COMMAND,
        workdir=None,
        warning="command requires approval",
        approval_id=None,
        background=False,
    )


# ── The returned status ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_expired_inline_wait_is_pending(web_ctx: ToolContext) -> None:
    """Fails without the fix: the status was approval_denied."""
    result = await _check()

    assert result is not None
    assert result["status"] == "approval_pending"
    assert result["approval_id"]
    assert result["command"] == _WARNED_COMMAND


@pytest.mark.asyncio
async def test_a_pending_approval_stays_in_the_queue(web_ctx: ToolContext) -> None:
    """The operator's click must still land after the model was told to wait."""
    result = await _check()

    assert result is not None
    pending = get_approval_queue().list_pending("exec")
    assert [entry["id"] for entry in pending] == [result["approval_id"]]


@pytest.mark.asyncio
async def test_a_pending_payload_is_not_classified_as_a_denial(
    web_ctx: ToolContext,
) -> None:
    result = await _check()

    assert is_denial_payload(result) is False


@pytest.mark.asyncio
async def test_a_real_rejection_is_still_a_denial(
    web_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard: the fix must not swallow an actual "no"."""
    queue = get_approval_queue()
    original_request = queue.request

    def _request_and_deny(**kwargs: object) -> str:
        approval_id = original_request(**kwargs)
        queue.resolve(approval_id, approved=False)
        return approval_id

    monkeypatch.setattr(queue, "request", _request_and_deny)

    result = await _check()

    assert result is not None
    assert result["status"] == "approval_denied"
    assert "denied" in str(result["message"]).lower()
    assert "timed out" not in str(result["message"]).lower()
    assert is_denial_payload(result) is True


@pytest.mark.asyncio
async def test_an_approval_granted_inline_still_runs(
    web_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard: passes either way by design — the approved branch is untouched."""
    queue = get_approval_queue()
    original_request = queue.request

    def _request_and_approve(**kwargs: object) -> str:
        approval_id = original_request(**kwargs)
        queue.resolve(approval_id, approved=True)
        return approval_id

    monkeypatch.setattr(queue, "request", _request_and_approve)

    assert await _check() is None
    assert shell._elevate_current_call.get() is True


# ── What reached the ledger ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exec_command_records_no_denial_for_a_pending_approval(
    web_ctx: ToolContext,
) -> None:
    """Fails without the fix: a HUMAN_REJECTED denial was written for an
    operator who had not rejected anything."""
    payload = json.loads(await shell.exec_command(_WARNED_COMMAND))

    assert payload["status"] == "approval_pending"
    total, reason = await _ledger_state()
    assert total == 0
    assert reason is None


@pytest.mark.asyncio
async def test_execute_code_surfaces_pending_rather_than_denied(
    web_ctx: ToolContext,
) -> None:
    """The destructive-Python gate shares ``_check_exec_approval``.

    It records no ledger denial of its own, but it hands the model the same
    payload, so before the fix the model was told its code had been rejected
    while the operator was still reading the prompt.
    """
    from agentos.tools.builtin.code_exec import execute_code

    payload = json.loads(await execute_code("import os\nos.remove('target.txt')"))

    assert payload["status"] == "approval_pending"
    assert is_denial_payload(payload) is False


@pytest.mark.asyncio
async def test_exec_command_still_records_a_real_rejection(
    web_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard: the audit trail for an actual rejection must survive the fix."""
    from agentos.sandbox.types import DenialReason

    queue = get_approval_queue()
    original_request = queue.request

    def _request_and_deny(**kwargs: object) -> str:
        approval_id = original_request(**kwargs)
        queue.resolve(approval_id, approved=False)
        return approval_id

    monkeypatch.setattr(queue, "request", _request_and_deny)

    payload = json.loads(await shell.exec_command(_WARNED_COMMAND))

    assert payload["status"] == "approval_denied"
    total, reason = await _ledger_state()
    assert total == 1
    assert reason is DenialReason.HUMAN_REJECTED


@pytest.mark.asyncio
async def test_a_pending_approval_does_not_run_the_command(
    web_ctx: ToolContext, tmp_path: Path
) -> None:
    """Pending is not permission: nothing may execute while it is unresolved."""
    target = tmp_path / "workspace" / "build"
    target.mkdir()

    payload = json.loads(await shell.exec_command(_WARNED_COMMAND))

    assert payload["status"] == "approval_pending"
    assert target.exists()


# ── The CLI/TUI surface is unaffected ──────────────────────────────────────


@pytest.mark.asyncio
async def test_a_cli_caller_still_gets_approval_required(
    web_ctx: ToolContext,
) -> None:
    """Guard: passes either way by design. Only the Web UI waits inline; the
    CLI drives its prompt from the approval_required result, and that branch
    is not the one being changed."""
    web_ctx.caller_kind = CallerKind.CLI

    result = await _check()

    assert result is not None
    assert result["status"] == "approval_required"
    total, _reason = await _ledger_state()
    assert total == 0
