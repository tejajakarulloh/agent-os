from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agentos.mcp_server.bridge import (
    _MAX_EVENTS_UPPER,
    _MAX_HISTORY_LIMIT_UPPER,
    _MAX_SESSIONS_LIMIT_UPPER,
    _MAX_TIMEOUT_MS_UPPER,
    AgentOSMCPBridge,
)


class FakeGatewayClient:
    def __init__(self) -> None:
        self.connected_url: str | None = None
        self.closed = False
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.last_recv_timeout: float | None = None

    async def connect(self, url: str) -> None:
        self.connected_url = url

    async def close(self) -> None:
        self.closed = True

    async def list_sessions(self, limit: int = 50) -> dict[str, Any]:
        self.calls.append(("sessions.list", {"limit": limit}))
        return {"sessions": [{"key": "agent:main:main", "entry_count": 2}], "count": 1}

    async def resolve_session(self, key: str) -> dict[str, Any]:
        self.calls.append(("sessions.resolve", {"key": key}))
        return {"key": key, "session_id": "sid-1", "agent_id": "main"}

    async def session_history(self, session_key: str, limit: int = 1000) -> dict[str, Any]:
        self.calls.append(("chat.history", {"sessionKey": session_key, "limit": limit}))
        return {
            "messages": [
                {
                    "id": "m1",
                    "role": "user",
                    "text": "hello",
                    "timestamp": 1,
                },
                {
                    "id": "m2",
                    "role": "assistant",
                    "text": "looked it up",
                    "timestamp": 2,
                    "tool_calls": [
                        {
                            "type": "tool_use",
                            "tool_use_id": "tool-1",
                            "name": "lookup",
                            "input": {"q": "hello"},
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "name": "lookup",
                            "result": "world",
                            "is_error": False,
                            "execution_status": {
                                "version": 1,
                                "status": "success",
                                "exit_code": 0,
                                "timed_out": False,
                                "truncated": False,
                                "reason": None,
                                "source": "adapter",
                                "preservation_class": "normal",
                            },
                        },
                    ],
                },
            ]
        }

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((method, params))
        if method == "sessions.messages.subscribe":
            return {
                "subscribed": True,
                "key": params["key"],
                "current_stream_seq": 7,
                "replay_complete": True,
            }
        if method == "sessions.send":
            return {"status": "accepted", "key": params["key"], "task_id": "task-1"}
        raise AssertionError(f"unexpected method: {method}")

    async def recv_event(self, timeout: float | None = None) -> dict[str, Any]:
        # Recorded for events_wait()'s deadline-clamp assertion — see
        # test_events_wait_clamps_effective_recv_deadline below.
        self.last_recv_timeout = timeout
        if timeout is None:
            return await self.events.get()
        return await asyncio.wait_for(self.events.get(), timeout=timeout)


@pytest.mark.asyncio
async def test_bridge_reuses_gateway_read_rpcs() -> None:
    client = FakeGatewayClient()
    bridge = AgentOSMCPBridge(
        gateway_url="ws://127.0.0.1:18791/ws",
        gateway_client_factory=lambda: client,
    )

    sessions = await bridge.conversations_list(limit=10)
    resolved = await bridge.session_resolve("agent:main:main")
    messages = await bridge.messages_read("agent:main:main", limit=5)

    assert client.connected_url == "ws://127.0.0.1:18791/ws"
    assert sessions["sessions"][0]["key"] == "agent:main:main"
    assert resolved["session_id"] == "sid-1"
    assert messages["messages"][0]["text"] == "hello"
    assert client.calls[:3] == [
        ("sessions.list", {"limit": 10}),
        ("sessions.resolve", {"key": "agent:main:main"}),
        ("chat.history", {"sessionKey": "agent:main:main", "limit": 5}),
    ]


@pytest.mark.asyncio
async def test_transcript_jsonl_preserves_tool_result_execution_status() -> None:
    client = FakeGatewayClient()
    bridge = AgentOSMCPBridge(gateway_client_factory=lambda: client)

    transcript = await bridge.transcript_jsonl("agent:main:main", limit=5)

    rows = [json.loads(line) for line in transcript.splitlines()]
    tool_result_message = rows[2]["message"]
    assert tool_result_message["isError"] is False
    assert tool_result_message["executionStatus"]["status"] == "success"


@pytest.mark.asyncio
async def test_messages_send_subscribes_before_accepting_turn() -> None:
    client = FakeGatewayClient()
    bridge = AgentOSMCPBridge(gateway_client_factory=lambda: client)

    result = await bridge.messages_send("agent:main:main", "continue", intent="continue")

    assert result == {
        "status": "accepted",
        "key": "agent:main:main",
        "task_id": "task-1",
        "current_stream_seq": 7,
        "replay_complete": True,
        "replay_gap_reason": None,
    }
    assert client.closed is True
    assert client.calls == [
        ("sessions.messages.subscribe", {"key": "agent:main:main", "since_stream_seq": None}),
        (
            "sessions.send",
            {
                "key": "agent:main:main",
                "message": "continue",
                "attachments": [],
                "intent": "continue",
                "_source": {
                    "caller_kind": "cli",
                    "channel_kind": "cli",
                    "channel_id": "mcp:bridge",
                    "source_kind": "mcp",
                    "source_name": "mcp_server",
                },
            },
        ),
    ]


@pytest.mark.asyncio
async def test_events_wait_returns_session_events_until_terminal() -> None:
    client = FakeGatewayClient()
    await client.events.put(
        {
            "event": "session.event.text_delta",
            "payload": {"session_key": "agent:main:main", "stream_seq": 8, "text": "hi"},
        }
    )
    await client.events.put(
        {
            "event": "session.event.done",
            "payload": {"session_key": "agent:main:main", "stream_seq": 9, "reason": "stop"},
        }
    )
    bridge = AgentOSMCPBridge(gateway_client_factory=lambda: client)

    result = await bridge.events_wait("agent:main:main", since_stream_seq=7, timeout_ms=1000)

    assert result["current_stream_seq"] == 9
    assert [event["event"] for event in result["events"]] == [
        "session.event.text_delta",
        "session.event.done",
    ]


@pytest.mark.asyncio
async def test_transcript_jsonl_exports_standard_tool_evidence() -> None:
    bridge = AgentOSMCPBridge(gateway_client_factory=FakeGatewayClient)

    text = await bridge.transcript_jsonl("agent:main:main")
    rows = [json.loads(line) for line in text.splitlines()]

    assert rows[0]["message"]["role"] == "user"
    assert rows[1]["message"]["role"] == "assistant"
    assert rows[1]["message"]["content"][0]["type"] == "toolCall"
    assert rows[1]["message"]["content"][0]["name"] == "lookup"
    assert rows[2]["message"]["role"] == "toolResult"
    assert rows[2]["message"]["toolCallId"] == "tool-1"


def test_mcp_server_package_does_not_import_cli_layer() -> None:
    package_root = Path("src/agentos/mcp_server")
    imported_modules: set[str] = set()
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

    assert not any(module.startswith("agentos.cli") for module in imported_modules)
    assert not any(
        module == "agentos.gateway" or module.startswith("agentos.gateway.")
        for module in imported_modules
    )


@pytest.mark.asyncio
async def test_events_wait_uses_dedicated_connection_and_closes_it() -> None:
    clients: list[FakeGatewayClient] = []
    event_client = FakeGatewayClient()

    def factory() -> FakeGatewayClient:
        if len(clients) == 1:
            clients.append(event_client)
            return event_client
        client = FakeGatewayClient()
        clients.append(client)
        return client

    bridge = AgentOSMCPBridge(gateway_client_factory=factory)
    await bridge.conversations_list()
    await clients[0].events.put(
        {
            "event": "session.event.done",
            "payload": {"session_key": "agent:main:main", "stream_seq": 3},
        }
    )
    await event_client.events.put(
        {
            "event": "session.event.done",
            "payload": {"session_key": "agent:main:main", "stream_seq": 8},
        }
    )

    result = await bridge.events_wait("agent:main:main", timeout_ms=1000)

    assert len(clients) == 2
    assert result["current_stream_seq"] == 8
    assert clients[0].closed is False
    assert event_client.closed is True


# --- Resource-exhaustion clamps (issue #685) --------------------------------
#
# Every parameter on these four call paths is chosen by the model on every
# call, so a runaway value is a self-inflicted availability problem rather than
# a remote DoS. The bridge silently clamps to module-level caps defined in
# `agentos.mcp_server.bridge`. These tests assert the clamp on both bounds for
# every vector.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        # Below the cap: pass through untouched.
        (50, 50),
        (1, 1),
        # At the cap: pass through.
        (_MAX_SESSIONS_LIMIT_UPPER, _MAX_SESSIONS_LIMIT_UPPER),
        # Above the cap: silently clamp to the cap.
        (_MAX_SESSIONS_LIMIT_UPPER + 1, _MAX_SESSIONS_LIMIT_UPPER),
        (10**9, _MAX_SESSIONS_LIMIT_UPPER),
    ],
)
async def test_conversations_list_clamps_limit(requested: int, expected: int) -> None:
    client = FakeGatewayClient()
    bridge = AgentOSMCPBridge(gateway_client_factory=lambda: client)

    await bridge.conversations_list(limit=requested)

    method, params = client.calls[0]
    assert method == "sessions.list"
    assert params == {"limit": expected}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (1000, 1000),
        (_MAX_HISTORY_LIMIT_UPPER, _MAX_HISTORY_LIMIT_UPPER),
        (_MAX_HISTORY_LIMIT_UPPER * 10, _MAX_HISTORY_LIMIT_UPPER),
        (10**9, _MAX_HISTORY_LIMIT_UPPER),
    ],
)
async def test_messages_read_clamps_limit(requested: int, expected: int) -> None:
    client = FakeGatewayClient()
    bridge = AgentOSMCPBridge(gateway_client_factory=lambda: client)

    await bridge.messages_read("agent:main:main", limit=requested)

    method, params = client.calls[0]
    assert method == "chat.history"
    assert params == {"sessionKey": "agent:main:main", "limit": expected}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (1000, 1000),
        (_MAX_HISTORY_LIMIT_UPPER, _MAX_HISTORY_LIMIT_UPPER),
        (_MAX_HISTORY_LIMIT_UPPER * 10, _MAX_HISTORY_LIMIT_UPPER),
        (10**9, _MAX_HISTORY_LIMIT_UPPER),
    ],
)
async def test_transcript_jsonl_clamps_limit(requested: int, expected: int) -> None:
    client = FakeGatewayClient()
    bridge = AgentOSMCPBridge(gateway_client_factory=lambda: client)

    await bridge.transcript_jsonl("agent:main:main", limit=requested)

    method, params = client.calls[0]
    assert method == "chat.history"
    assert params == {"sessionKey": "agent:main:main", "limit": expected}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_timeout", "requested_events"),
    [
        # Below the cap: pass through.
        (1000, 1),
        # At the cap: pass through.
        (_MAX_TIMEOUT_MS_UPPER, _MAX_EVENTS_UPPER),
        # Above the cap: clamp.
        (_MAX_TIMEOUT_MS_UPPER * 10, _MAX_EVENTS_UPPER * 10),
        (10**9, 10**9),
    ],
)
async def test_events_wait_returns_for_huge_clamped_arguments(
    requested_timeout: int, requested_events: int
) -> None:
    """A pre-loaded terminal event lets the call return immediately, so a
    several-hour timeout that gets clamped to 5 minutes still resolves on the
    first recv_event tick — the test cares that the loop terminates, not how
    long it would have waited without the clamp."""
    client = FakeGatewayClient()
    await client.events.put(
        {
            "event": "session.event.done",
            "payload": {"session_key": "agent:main:main", "stream_seq": 8},
        }
    )
    bridge = AgentOSMCPBridge(gateway_client_factory=lambda: client)

    result = await bridge.events_wait(
        "agent:main:main",
        since_stream_seq=7,
        timeout_ms=requested_timeout,
        max_events=requested_events,
    )

    assert result["current_stream_seq"] == 8
    assert result["timed_out"] is False


@pytest.mark.asyncio
async def test_events_wait_clamps_max_events_to_upper() -> None:
    """A max_events=10**9 would loop indefinitely waiting for ten billion
    events. After clamping, the bridge returns as soon as a terminal event
    arrives, even though the caller asked for the full ten billion."""
    client = FakeGatewayClient()
    await client.events.put(
        {
            "event": "session.event.done",
            "payload": {"session_key": "agent:main:main", "stream_seq": 1},
        }
    )
    bridge = AgentOSMCPBridge(gateway_client_factory=lambda: client)

    result = await bridge.events_wait(
        "agent:main:main",
        timeout_ms=5000,
        max_events=10**9,
    )

    assert len(result["events"]) == 1


@pytest.mark.asyncio
async def test_events_wait_floors_non_positive_max_events_to_one() -> None:
    """The lower clamp `max(1, max_events)` is what kept the original code
    from looping forever on `max_events=0` or negative values. Pin it so a
    refactor that drops the clamp is caught immediately."""
    client = FakeGatewayClient()
    await client.events.put(
        {
            "event": "session.event.done",
            "payload": {"session_key": "agent:main:main", "stream_seq": 1},
        }
    )
    bridge = AgentOSMCPBridge(gateway_client_factory=lambda: client)

    for bogus in (0, -1, -100):
        if bogus != 0:
            await client.events.put(
                {
                    "event": "session.event.done",
                    "payload": {
                        "session_key": "agent:main:main",
                        "stream_seq": bogus + 2,
                    },
                }
            )
        result = await bridge.events_wait(
            "agent:main:main",
            timeout_ms=1000,
            max_events=bogus,
        )
        assert len(result["events"]) == 1


@pytest.mark.asyncio
async def test_events_wait_clamps_effective_recv_deadline() -> None:
    """A 1-hour timeout_ms must not propagate a 3600-second `timeout` to
    recv_event — the clamp on `timeout_ms` is what keeps the loop from
    blocking on a single recv_event call for an hour.

    The fake is pre-loaded with a terminal event so the call returns
    immediately; we then assert the value the bridge forwarded to the
    underlying recv_event is no larger than the upper clamp."""
    client = FakeGatewayClient()
    await client.events.put(
        {
            "event": "session.event.done",
            "payload": {"session_key": "agent:main:main", "stream_seq": 1},
        }
    )
    bridge = AgentOSMCPBridge(gateway_client_factory=lambda: client)

    await bridge.events_wait(
        "agent:main:main",
        timeout_ms=3_600_000,
        max_events=1,
    )

    assert client.last_recv_timeout is not None
    assert client.last_recv_timeout <= _MAX_TIMEOUT_MS_UPPER / 1000 + 0.1


@pytest.mark.asyncio
async def test_events_wait_floors_non_positive_timeout_to_zero() -> None:
    """`timeout_ms=0` and negative values must produce a deadline at or before
    `now`, so the loop exits before even calling recv_event — the worst case
    is `recv_event` being asked to wait forever, which is what the original
    `max(0, timeout_ms) / 1000` accidentally avoided (it would have produced
    `now + 0`, i.e. an immediate deadline, but only because the sign was
    swallowed)."""
    client = FakeGatewayClient()
    bridge = AgentOSMCPBridge(gateway_client_factory=lambda: client)

    for bogus in (0, -1, -1000):
        result = await bridge.events_wait(
            "agent:main:main",
            timeout_ms=bogus,
            max_events=1,
        )
        # No events were pre-loaded, and the loop exited on the first
        # `remaining <= 0` check, so recv_event was never called.
        assert result["events"] == []
        assert result["timed_out"] is True
        assert client.last_recv_timeout is None
