"""stdio must report a tool's failure and keep its non-text content (#1558).

``MCPStdioClient.call_tool`` ended in ``return MCPToolResult(content=text)``,
which takes ``is_error``'s ``False`` default no matter what the server said, and
built ``text`` from ``type == "text"`` blocks alone. Two defects in four lines:

* an application-level failure reached the model as a *successful* result whose
  text happens to describe an error — no signal to retry, no signal to stop,
  and the tool-error path never ran;
* image blocks, embedded resources and ``structuredContent`` were dropped, so a
  server that answers with structured content only produced ``content=""``.

``MCPSessionClient`` (the streamable-HTTP / SSE transports) already did both
correctly. Rather than copy its five lines into stdio and let them drift again,
the rendering now lives in one shared ``tool_result_from_call``; the parity
tests below pin that the two transports answer the same payload identically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agentos.mcp.client import MCPSessionClient, tool_result_from_call
from agentos.mcp.stdio import MCPStdioClient
from agentos.mcp.types import MCPServerConfig, MCPToolResult

_TEXT_ONLY = {"content": [{"type": "text", "text": "plain answer"}]}
_FAILED = {
    "content": [{"type": "text", "text": "upstream API rejected the write"}],
    "isError": True,
}
_IMAGE = {"content": [{"type": "image", "data": "aGk=", "mimeType": "image/png"}]}
_MIXED = {
    "content": [
        {"type": "text", "text": "chart follows"},
        {"type": "image", "data": "aGk=", "mimeType": "image/png"},
    ],
    "isError": False,
}
_RESOURCE = {
    "content": [{"type": "resource", "resource": {"uri": "memo://notes/1", "text": "note body"}}]
}
_STRUCTURED_ONLY = {"content": [], "structuredContent": {"rows": 3, "ok": True}}
_STRUCTURED_AND_TEXT = {
    "content": [{"type": "text", "text": "summary"}],
    "structuredContent": {"rows": 3},
}
_FAILED_STRUCTURED = {"content": [], "structuredContent": {"code": 42}, "isError": True}

_ALL_PAYLOADS = [
    _TEXT_ONLY,
    _FAILED,
    _IMAGE,
    _MIXED,
    _RESOURCE,
    _STRUCTURED_ONLY,
    _STRUCTURED_AND_TEXT,
    _FAILED_STRUCTURED,
]


def _stdio_client() -> MCPStdioClient:
    return MCPStdioClient(MCPServerConfig(name="demo", transport="stdio", command="demo"))


async def _stdio_result(payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> MCPToolResult:
    client = _stdio_client()

    async def _send_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": 1, "result": payload}

    monkeypatch.setattr(client, "_send_request", _send_request)
    return await client.call_tool("demo_tool", {})


async def _session_result(payload: dict[str, Any]) -> MCPToolResult:
    """What the streamable-HTTP / SSE transports make of the same payload."""
    from mcp.types import CallToolResult

    class _FakeSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            return CallToolResult.model_validate(payload)

    class _Client(MCPSessionClient):
        async def _open_session(self, stack: Any) -> Any:  # pragma: no cover - unused
            raise NotImplementedError

    client = _Client(MCPServerConfig(name="demo", transport="streamable_http", url="http://x"))
    client._session = _FakeSession()
    return await client.call_tool("demo_tool", {})


# ── The reported half: isError was dropped ──────────────────────────────────


@pytest.mark.asyncio
async def test_application_level_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails without the fix: is_error took its False default."""
    result = await _stdio_result(_FAILED, monkeypatch)

    assert result.is_error is True
    assert result.content == "upstream API rejected the write"


@pytest.mark.asyncio
async def test_failure_is_reported_even_with_no_text_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _stdio_result(_FAILED_STRUCTURED, monkeypatch)

    assert result.is_error is True
    assert json.loads(result.content) == {"code": 42}


@pytest.mark.asyncio
async def test_a_successful_call_is_still_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard: passes either way by design — False was the old default, and the
    flag must not now be true for everything."""
    result = await _stdio_result(_TEXT_ONLY, monkeypatch)

    assert result.is_error is False
    assert result.content == "plain answer"


# ── The folded-in half (#1572): non-text content was dropped ────────────────


@pytest.mark.asyncio
async def test_an_image_block_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails without the fix: content was "" for an image-only result."""
    result = await _stdio_result(_IMAGE, monkeypatch)

    assert "aGk=" in result.content
    assert "image/png" in result.content


@pytest.mark.asyncio
async def test_an_embedded_resource_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    result = await _stdio_result(_RESOURCE, monkeypatch)

    assert "memo://notes/1" in result.content


@pytest.mark.asyncio
async def test_structured_content_is_the_fallback_when_there_are_no_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails without the fix: a structured-only server produced content=""."""
    result = await _stdio_result(_STRUCTURED_ONLY, monkeypatch)

    assert json.loads(result.content) == {"rows": 3, "ok": True}


@pytest.mark.asyncio
async def test_structured_content_does_not_displace_text_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It is a fallback, not an addition — same rule the session client uses."""
    result = await _stdio_result(_STRUCTURED_AND_TEXT, monkeypatch)

    assert result.content == "summary"


@pytest.mark.asyncio
async def test_text_and_non_text_blocks_are_both_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _stdio_result(_MIXED, monkeypatch)

    first, second = result.content.split("\n", 1)
    assert first == "chart follows"
    assert "image/png" in second


# ── Parity: the two transports must not drift apart again ──────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", _ALL_PAYLOADS)
async def test_stdio_and_session_clients_agree(
    payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same server payload must produce the same MCPToolResult on both."""
    assert await _stdio_result(payload, monkeypatch) == await _session_result(payload)


# ── Tolerance: an extensible payload is not a tool error ───────────────────


@pytest.mark.asyncio
async def test_a_content_block_this_sdk_cannot_model_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP content types outgrow a pinned SDK.

    Rendering such a result through the SDK model alone would turn an ordinary
    success into a failure whose body is a validation traceback — strictly
    worse than the drop being fixed here. The raw block is kept instead.
    """
    payload = {"content": [{"type": "widget", "spec": {"kind": "table"}}], "isError": False}

    result = await _stdio_result(payload, monkeypatch)

    assert result.is_error is False
    assert "widget" in result.content
    assert "ValidationError" not in result.content


@pytest.mark.asyncio
async def test_a_result_with_no_content_key_is_empty_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _stdio_result({}, monkeypatch)

    assert result == MCPToolResult(content="", is_error=False)


@pytest.mark.asyncio
async def test_a_jsonrpc_error_response_is_still_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard: passes either way by design — the transport-level error branch
    predates this change and must keep reporting is_error."""
    client = _stdio_client()

    async def _send_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "no such tool"}}

    monkeypatch.setattr(client, "_send_request", _send_request)
    result = await client.call_tool("demo_tool", {})

    assert result.is_error is True
    assert result.content == "no such tool"


def test_the_shared_renderer_is_what_both_paths_use() -> None:
    """Guard: the helper is importable and behaves on a plain object, which is
    what keeps the two transports from drifting again."""

    class _Block:
        text = "hello"

    class _Result:
        # MCP wire names, which is why these are not snake_case.
        content = [_Block()]
        isError = True  # noqa: N815
        structuredContent = None  # noqa: N815

    assert tool_result_from_call(_Result()) == MCPToolResult(content="hello", is_error=True)


# ── End to end, over a real subprocess server ──────────────────────────────


_FAILING_SERVER = """
import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    if "id" not in message:
        continue
    method = message["method"]
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {}}
    elif method == "tools/call":
        result = {
            "content": [
                {"type": "text", "text": "quota exceeded"},
                {"type": "image", "data": "aGk=", "mimeType": "image/png"},
            ],
            "isError": True,
        }
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}) + "\\n")
    sys.stdout.flush()
"""


@pytest.mark.asyncio
async def test_end_to_end_over_a_real_stdio_server(tmp_path: Path) -> None:
    server = tmp_path / "server.py"
    server.write_text(_FAILING_SERVER)
    client = MCPStdioClient(
        MCPServerConfig(name="demo", transport="stdio", command=sys.executable, args=[str(server)])
    )

    try:
        await client.connect()
        result = await client.call_tool("write_row", {})
    finally:
        await client.close()

    assert result.is_error is True
    assert result.content.startswith("quota exceeded")
    assert "image/png" in result.content
