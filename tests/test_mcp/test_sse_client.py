"""Unit tests for ``MCPSSEClient`` in ``agentos.mcp.sse``."""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, patch

import httpx
import pytest

from agentos.mcp.discovery import create_client
from agentos.mcp.sse import MCPSSEClient
from agentos.mcp.types import MCPServerConfig


def test_factory_builds_sse_client() -> None:
    config = MCPServerConfig(
        name="sse-server",
        transport="sse",
        url="https://example.test/sse",
    )

    assert isinstance(create_client(config), MCPSSEClient)


@pytest.mark.asyncio
async def test_connect_configures_timeout_and_headers() -> None:
    config = MCPServerConfig(
        name="sse-server",
        transport="sse",
        url="https://example.test/sse",
        headers={"X-Custom-Auth": "secret-123"},
        tool_timeout_seconds=45.0,
    )
    client = MCPSSEClient(config)

    mock_instance = AsyncMock()
    mock_instance.aclose = AsyncMock()

    with patch("agentos.mcp.sse.httpx.AsyncClient", return_value=mock_instance) as mock_http_cls:
        with (
            patch.object(client, "_send_and_receive", new_callable=AsyncMock) as mock_send_recv,
            patch.object(client, "_send_notification", new_callable=AsyncMock) as mock_notify,
        ):
            await client.connect()

            mock_http_cls.assert_called_once()
            _, kwargs = mock_http_cls.call_args
            assert kwargs.get("headers") == {"X-Custom-Auth": "secret-123"}
            timeout = kwargs.get("timeout")
            assert isinstance(timeout, httpx.Timeout)
            assert timeout.read == 45.0
            mock_send_recv.assert_awaited_once_with("initialize", ANY)
            mock_notify.assert_awaited_once_with("notifications/initialized")

        await client.close()
        mock_instance.aclose.assert_awaited_once()


def test_parse_sse_event_and_stream() -> None:
    raw_event = 'data: {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}\n\n'
    events = MCPSSEClient._parse_sse_stream(raw_event)
    assert len(events) == 1
    assert events[0]["id"] == 1
    assert events[0]["result"]["tools"] == []
