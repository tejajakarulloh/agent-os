from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from agentos.gateway_client import GatewayRPCClient, normalize_gateway_url


class _SilentWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def close(self) -> None:
        self.closed = True


def test_normalize_gateway_url_preserves_query_and_fragment() -> None:
    assert (
        normalize_gateway_url("https://gateway.example.com/ws?token=abc#trace")
        == "wss://gateway.example.com/ws?token=abc#trace"
    )


def test_normalize_gateway_url_adds_ws_path_without_dropping_query() -> None:
    assert normalize_gateway_url("gateway.example.com?token=abc") == "ws://gateway.example.com/ws?token=abc"


@pytest.mark.asyncio
async def test_gateway_rpc_call_times_out_and_clears_pending_request() -> None:
    client = GatewayRPCClient(request_timeout_s=0.01)
    client._ws = _SilentWebSocket()

    with pytest.raises(TimeoutError, match="sessions.list timed out"):
        await client.call("sessions.list", {"limit": 1})

    assert client._pending == {}


@pytest.mark.asyncio
async def test_gateway_connect_closes_socket_after_bad_handshake(monkeypatch) -> None:
    class BadHandshakeWebSocket(_SilentWebSocket):
        async def recv(self) -> str:
            return json.dumps({"type": "event", "event": "unexpected"})

    ws = BadHandshakeWebSocket()

    async def connect(_url: str):
        return ws

    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=connect))
    client = GatewayRPCClient()

    with pytest.raises(RuntimeError, match="Unexpected gateway handshake frame"):
        await client.connect("ws://127.0.0.1:18791/ws")

    assert ws.closed is True
    assert client._ws is None


async def _start_pending_call(client: GatewayRPCClient, method: str = "sessions.list"):
    """Start a ``call`` and wait until its request is registered as pending."""

    task = asyncio.ensure_future(client.call(method, {"limit": 1}))
    while not client._pending:
        await asyncio.sleep(0)
    return task


@pytest.mark.asyncio
async def test_close_fails_pending_call_without_a_request_timeout() -> None:
    # With no request timeout there is no other escape hatch: before the fix
    # this task stayed blocked on ``await fut`` forever.
    client = GatewayRPCClient(request_timeout_s=None)
    client._ws = _SilentWebSocket()
    task = await _start_pending_call(client)

    await client.close()

    with pytest.raises(ConnectionError, match="Gateway connection closed"):
        await asyncio.wait_for(task, timeout=2)
    assert client._pending == {}


@pytest.mark.asyncio
async def test_close_fails_pending_call_promptly_under_a_request_timeout() -> None:
    # A generous timeout would eventually fire, but with the wrong story: the
    # connection was closed, it did not time out.
    client = GatewayRPCClient(request_timeout_s=300.0)
    client._ws = _SilentWebSocket()
    task = await _start_pending_call(client)

    await client.close()

    with pytest.raises(ConnectionError, match="Gateway connection closed"):
        await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_close_fails_every_pending_call() -> None:
    client = GatewayRPCClient(request_timeout_s=None)
    client._ws = _SilentWebSocket()
    first = await _start_pending_call(client, "sessions.list")
    second = asyncio.ensure_future(client.call("sessions.resolve", {"key": "agent:main:main"}))
    while len(client._pending) < 2:
        await asyncio.sleep(0)

    await client.close()

    for task in (first, second):
        with pytest.raises(ConnectionError, match="Gateway connection closed"):
            await asyncio.wait_for(task, timeout=2)
    assert client._pending == {}


@pytest.mark.asyncio
async def test_reconnect_fails_the_previous_connections_pending_calls(monkeypatch) -> None:
    # ``connect`` closes a live client first, so the old socket's in-flight
    # calls must not be left waiting on a connection that no longer exists.
    client = GatewayRPCClient(request_timeout_s=None)
    ws = _SilentWebSocket()
    client._ws = ws
    task = await _start_pending_call(client)

    async def connect(_url: str):
        raise RuntimeError("gateway unreachable")

    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=connect))

    with pytest.raises(RuntimeError, match="gateway unreachable"):
        await client.connect("ws://127.0.0.1:18791/ws")

    assert ws.closed is True
    with pytest.raises(ConnectionError, match="Gateway connection closed"):
        await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_close_leaves_an_already_settled_future_alone() -> None:
    # A cancelled call can linger in ``_pending``; setting an exception on it
    # would raise InvalidStateError or log an unretrieved exception.
    client = GatewayRPCClient(request_timeout_s=None)
    client._ws = _SilentWebSocket()
    task = await _start_pending_call(client)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await client.close()

    assert client._pending == {}


@pytest.mark.asyncio
async def test_close_still_closes_the_socket_with_no_pending_calls() -> None:
    client = GatewayRPCClient(request_timeout_s=None)
    ws = _SilentWebSocket()
    client._ws = ws

    await client.close()

    assert ws.closed is True
    assert client._ws is None
    assert client._pending == {}


@pytest.mark.asyncio
async def test_connection_failure_still_fails_pending_calls() -> None:
    # ``_mark_connection_failed`` shares the settle-and-clear step with
    # ``close``; an unexpected drop must keep reporting the lost connection.
    client = GatewayRPCClient(request_timeout_s=None)
    client._ws = _SilentWebSocket()
    task = await _start_pending_call(client)

    client._mark_connection_failed(OSError("socket reset"))

    with pytest.raises(ConnectionError, match="Gateway connection lost: socket reset"):
        await asyncio.wait_for(task, timeout=2)
    assert client._pending == {}
