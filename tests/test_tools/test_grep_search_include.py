from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from agentos.tools.builtin import filesystem as fs
from agentos.tools.types import CallerKind, ToolContext, current_tool_context


@contextmanager
def tool_context(workspace: Path) -> Iterator[None]:
    token = current_tool_context.set(
        ToolContext(
            caller_kind=CallerKind.CLI,
            channel_kind="cli",
            channel_id="cli:test",
            workspace_dir=str(workspace),
            workspace_strict=True,
        )
    )
    try:
        yield
    finally:
        current_tool_context.reset(token)


def _make_tree(root: Path) -> None:
    (root / "tests").mkdir()
    (root / "tests" / "test_basic.py").write_text("def test_x(): pass\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "utils.py").write_text("def test_x(): pass\n", encoding="utf-8")
    (root / "src" / "pkg").mkdir()
    (root / "src" / "pkg" / "helpers.py").write_text("def test_x(): pass\n", encoding="utf-8")
    (root / "test_basic.py").write_text("def test_x(): pass\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_include_with_path_qualified_glob_matches_relative_path(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    with tool_context(tmp_path):
        out = await fs.grep_search("def test_x", path=str(tmp_path), include="tests/*.py")

    assert "No matches" not in out
    assert str(tmp_path / "tests" / "test_basic.py") in out
    assert str(tmp_path / "src" / "utils.py") not in out
    assert str(tmp_path / "test_basic.py") not in out


@pytest.mark.asyncio
async def test_include_with_bare_filename_glob_still_matches_by_basename(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    with tool_context(tmp_path):
        out = await fs.grep_search("def test_x", path=str(tmp_path), include="*.py")

    assert str(tmp_path / "tests" / "test_basic.py") in out
    assert str(tmp_path / "src" / "utils.py") in out
    assert str(tmp_path / "test_basic.py") in out


@pytest.mark.asyncio
async def test_include_with_double_star_path_glob_matches_nested_paths(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    with tool_context(tmp_path):
        out = await fs.grep_search("def test_x", path=str(tmp_path), include="src/**/*.py")

    assert str(tmp_path / "src" / "pkg" / "helpers.py") in out
    assert str(tmp_path / "tests" / "test_basic.py") not in out


@pytest.mark.asyncio
async def test_include_path_glob_that_matches_nothing_reports_no_matches(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    with tool_context(tmp_path):
        out = await fs.grep_search("def test_x", path=str(tmp_path), include="docs/*.py")

    assert "No matches" in out
