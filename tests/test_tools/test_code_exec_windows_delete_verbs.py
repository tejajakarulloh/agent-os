"""The destructive-code gate must know the Windows delete vocabulary (#1466).

``_DESTRUCTIVE_PY_PATTERNS`` and ``_DestructiveCodeVisitor`` only ever armed on
``rm``/``rmdir``. On a Windows host the agent's deletion vocabulary is ``del``,
``erase``, ``rd`` and ``Remove-Item`` — every one of which returned ``None``
from ``_check_code_destructive``, and ``None`` is not a soft miss:
``execute_code`` gates on it, so the code ran with no prompt at all. That is
the exact "pivot from ``rm``" bypass the module comment says it exists to catch.

The verbs are matched on every platform, not behind an ``os.name`` guard: the
check runs where the code is *submitted*, and a Linux gateway can be driving a
Windows target. Tests below pin that by simulating both platforms rather than
skipping on one.
"""

from __future__ import annotations

import os

import pytest

from agentos.tools.builtin.code_exec import _check_code_destructive, execute_code
from agentos.tools.types import (
    CallerKind,
    InteractionMode,
    ToolContext,
    UnsupportedSurfaceError,
    current_tool_context,
)

_MISSING = "agentos-1466-does-not-exist.txt"


# ── Every verb the issue enumerates, through every shell carrier ────────────


@pytest.mark.parametrize(
    "code",
    [
        # os.system — the form in the report
        f'import os; os.system("del {_MISSING}")',
        f'import os; os.system("erase {_MISSING}")',
        'import os; os.system("rd /s /q C:\\\\temp\\\\x")',
        f'import os; os.system("Remove-Item -Recurse -Force {_MISSING}")',
        # case-insensitive: PowerShell does not care, and neither should we
        f'import os; os.system("DEL {_MISSING}")',
        f'import os; os.system("remove-item {_MISSING}")',
        # os.popen — same vocabulary, different entrypoint
        f'import os; os.popen("del {_MISSING}")',
        f'import os; os.popen("rd /s /q {_MISSING}")',
        # subprocess, argv form — the second form in the report
        f'import subprocess; subprocess.run(["cmd.exe", "/c", "del", "{_MISSING}"])',
        f'import subprocess; subprocess.run(["powershell", "-c", "Remove-Item", "{_MISSING}"])',
        f'import subprocess; subprocess.Popen(["cmd", "/c", "rd", "/s", "/q", "{_MISSING}"])',
        f'import subprocess; subprocess.check_call(["erase", "{_MISSING}"])',
        # subprocess, string form
        f'import subprocess; subprocess.run("del {_MISSING}", shell=True)',
        f'import subprocess as sp; sp.call("rd /s /q {_MISSING}", shell=True)',
        # behind an interpreter or privilege prefix
        f'import os; os.system("cmd /c del {_MISSING}")',
        f'import os; os.system("powershell -Command Remove-Item {_MISSING}")',
        f'import os; os.system("sudo del {_MISSING}")',
        # after a shell separator, not at the start of the command
        f'import os; os.system("cd /tmp && del {_MISSING}")',
        f'import os; os.system("echo hi; erase {_MISSING}")',
        f'import os; os.system("dir | rd /s /q {_MISSING}")',
    ],
)
def test_windows_delete_commands_are_detected(code: str) -> None:
    warning = _check_code_destructive(code)
    assert warning is not None, f"no warning for: {code}"
    assert "destructive Python operation detected:" in warning


# ── Indirection must not bypass it either (the #848 evasion set) ────────────


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (f'import os; getattr(os, "system")("del {_MISSING}")', "os.system with del"),
        (f'import os; getattr(os, "sys" + "tem")("rd /s /q {_MISSING}")', "os.system with rd"),
        (f'import os; getattr(os, "popen")("erase {_MISSING}")', "os.popen with erase"),
        (
            f'getattr(__import__("os"), "system")("Remove-Item {_MISSING}")',
            "os.system with remove-item",
        ),
        (
            'import subprocess; getattr(subprocess, "run")(["cmd", "/c", "del", "x"])',
            "subprocess invoking del",
        ),
        (
            "exec(compile('os.sys' + 'tem(\"del x\")', '', 'exec'))",
            "os.system with del",
        ),
    ],
)
def test_indirect_windows_delete_calls_are_detected(code: str, expected: str) -> None:
    """The verb reaching the human in the approval prompt is the one submitted."""
    warning = _check_code_destructive(code)
    assert warning is not None, f"no warning for: {code}"
    assert expected in warning.lower()


# ── Negative: short verbs must not arm on ordinary text ─────────────────────


@pytest.mark.parametrize(
    "code",
    [
        # `del` and `rd` inside longer words or path segments
        'import os; os.system("cat order_details.txt")',
        'import os; os.system("python train_model.py")',
        'import os; os.system("cat card_reader.conf")',
        'import os; os.system("ls /mnt/rd")',
        'import os; os.system("cat /var/log/hard_drive.log")',
        'import os; os.system("git log --oneline")',
        # the verb as an *argument*, not a command
        'import os; os.system("grep del file.txt")',
        'import os; os.system("echo processing delete queue")',
        'import subprocess; subprocess.run(["git", "commit", "-m", "del old rows"])',
        'import subprocess; subprocess.run(["myapp", "--del", "x"])',
        # read-only utilities that merely mention it
        'import subprocess; subprocess.run(["git", "status"])',
        'import subprocess; subprocess.run(["ri", "String#length"])',
        # plain Python that happens to contain the word
        'print("del")',
        "rd = 5\ndel rd",
    ],
)
def test_benign_code_does_not_arm_the_gate(code: str) -> None:
    warning = _check_code_destructive(code)
    assert warning is None, f"false positive: {warning}"


# ── Platform independence, simulated rather than skipped ───────────────────


@pytest.mark.parametrize("platform_name", ["nt", "posix"])
def test_detection_does_not_depend_on_the_host_platform(
    platform_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Linux gateway can be driving a Windows target, so the arms must not
    be gated on ``os.name``."""
    monkeypatch.setattr(os, "name", platform_name)

    assert _check_code_destructive(f'import os; os.system("del {_MISSING}")') is not None
    assert _check_code_destructive('import os; os.system("rd /s /q x")') is not None
    assert _check_code_destructive('import os; os.system("rm -rf /tmp/x")') is not None
    assert _check_code_destructive('import os; os.system("cat order_details.txt")') is None


# ── The POSIX arms and their warning text are unchanged ────────────────────


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ('import os; os.system("rm -rf /etc/x")', "os.system with rm"),
        ('import os; os.system("rmdir /tmp/x")', "os.system with rm"),
        ('import os; getattr(os, "popen")("rm -rf /etc/x")', "os.popen with rm"),
        ('import subprocess; subprocess.run(["rm", "-rf", "/tmp/x"])', "subprocess invoking rm"),
        ('import subprocess as sp; sp.call(["rmdir", "/tmp/x"])', "subprocess invoking rm"),
        ('import os; os.remove("/tmp/x")', "os.remove()"),
    ],
)
def test_existing_posix_warnings_are_unchanged(code: str, expected: str) -> None:
    """Guard: passes either way by design. The warning strings are what the
    approval record and the existing suite assert on, so widening the gate
    must not rewrite them."""
    warning = _check_code_destructive(code)
    assert warning is not None
    assert expected in warning


# ── The warning actually reaches the approval gate ─────────────────────────


@pytest.mark.asyncio
async def test_execute_code_gates_a_windows_delete_instead_of_running_it() -> None:
    """``None`` from the check means ``execute_code`` runs the code unprompted.

    Driven on an unattended surface so the gate's refusal is observable
    without a queue round-trip: reaching ``_check_exec_approval`` at all is
    the thing being proven. Before the fix this returned a normal execution
    result.
    """
    token = current_tool_context.set(
        ToolContext(
            caller_kind=CallerKind.CLI,
            session_key="agent:main:test-1466",
            interaction_mode=InteractionMode.UNATTENDED,
        )
    )
    try:
        with pytest.raises(UnsupportedSurfaceError):
            await execute_code(f'import os\nos.system("del {_MISSING}")')
    finally:
        current_tool_context.reset(token)
