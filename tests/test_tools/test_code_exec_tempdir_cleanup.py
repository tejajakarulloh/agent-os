"""``execute_code`` must not leak its scratch directory on early returns.

Scoped to the only path that can leak: ``runtime is None`` with no
``workspace_dir``, where ``mkdtemp`` runs and the gate then denies fail-closed.
The sandbox branch reuses ``runtime.workspace`` and creates no tempdir, so a
test written against it passes with or without the fix.

``TMPDIR`` is redirected per test rather than globbing the shared system temp
directory, so the count cannot be disturbed by anything else on the machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentos.tools.builtin import code_exec as code_exec_module
from agentos.tools.builtin.code_exec import execute_code


@pytest.fixture()
def scratch_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point tempfile at an empty directory this test owns."""
    target = tmp_path / "tmp"
    target.mkdir()
    monkeypatch.setattr(code_exec_module.tempfile, "tempdir", str(target))
    return target


def _scratch_dirs(root: Path) -> list[Path]:
    return sorted(root.glob("agentos_exec_*"))


@pytest.mark.asyncio
async def test_denied_call_leaves_no_scratch_directory(scratch_tmp: Path) -> None:
    result = await execute_code("print('hello')")

    assert json.loads(result)["status"] == "denied"
    assert _scratch_dirs(scratch_tmp) == []


@pytest.mark.asyncio
async def test_repeated_denials_do_not_accumulate_scratch_directories(
    scratch_tmp: Path,
) -> None:
    for _ in range(5):
        await execute_code("print('hello')")

    assert _scratch_dirs(scratch_tmp) == []


@pytest.mark.asyncio
async def test_scratch_directory_is_created_on_this_path(
    scratch_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guards the test itself: if execute_code ever stops calling mkdtemp here,
    # the assertions above would pass vacuously and prove nothing.
    created: list[str] = []
    real_mkdtemp = code_exec_module.tempfile.mkdtemp

    def _record(*args: object, **kwargs: object) -> str:
        path = real_mkdtemp(*args, **kwargs)  # type: ignore[arg-type]
        created.append(path)
        return path

    monkeypatch.setattr(code_exec_module.tempfile, "mkdtemp", _record)

    await execute_code("print('hello')")

    assert len(created) == 1
    assert not Path(created[0]).exists()


@pytest.mark.asyncio
async def test_gate_exception_still_propagates_and_cleans_up(
    scratch_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cleanup wrapper must not swallow errors from the gate the way the
    # inner subprocess handler turns them into an "Execution error" payload.
    async def _boom(**_kwargs: object) -> None:
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(code_exec_module, "gate_action", _boom)

    with pytest.raises(RuntimeError, match="gate exploded"):
        await execute_code("print('hello')")

    assert _scratch_dirs(scratch_tmp) == []
