"""Benign Slack reaction errors must not switch status reactions off for good.

``_post`` raised ``RuntimeError`` for every non-``ok`` response except the two
auth-shaped ones, and ``_add_state``/``_clear_active`` turn any exception into
``_disable(...)`` -- which is permanent, since ``_disabled`` is only ever set
back to ``False`` in ``__init__``. Slack retries an event on timeout, so
``already_reacted`` is a routine response to ``received()``; ``no_reaction``
is what a human removing the emoji produces. Either one silenced ✅/👀/❌ for
every later message the adapter handled.

These tests drive the real ``SlackStatusReactor._post`` through a scripted
HTTP client rather than a hand-rolled fake, and cover the part the reactor is
actually judged on: that a *subsequent* message still gets its reaction, that
``_active`` is left in the right state, and that every error which genuinely
means "the call did not happen" still disables.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentos.channels._reactions import SlackStatusReactor
from agentos.channels.types import IncomingMessage


class _SilentLog:
    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None


def _message(ts: str = "1700000000.000100", channel: str = "C1") -> IncomingMessage:
    return IncomingMessage(
        sender_id="U1", channel_id=channel, content="hi", metadata={"ts": ts}
    )


def _response(body: dict[str, Any], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = body
    return resp


def _reactor(responder: Any) -> tuple[SlackStatusReactor, MagicMock]:
    """Build a reactor whose Slack client answers through *responder*.

    *responder* takes ``(path, payload)`` and returns the JSON body.
    """
    client = MagicMock()

    async def _post(path: str, json: dict[str, str]) -> MagicMock:
        body = responder(path, json)
        return _response(body) if isinstance(body, dict) else body

    client.post = AsyncMock(side_effect=_post)
    channel = MagicMock()
    channel._get_client.return_value = client
    return SlackStatusReactor(channel, _SilentLog()), client


def _always(body: dict[str, Any]) -> Any:
    return lambda path, payload: body


def _added_emoji(client: MagicMock) -> list[str]:
    return [
        call.kwargs["json"]["name"]
        for call in client.post.await_args_list
        if call.args[0] == "/reactions.add"
    ]


def _removed_emoji(client: MagicMock) -> list[str]:
    return [
        call.kwargs["json"]["name"]
        for call in client.post.await_args_list
        if call.args[0] == "/reactions.remove"
    ]


# ── The reported symptom: reactions keep working afterwards ──────────────────


@pytest.mark.asyncio
async def test_a_later_message_still_gets_its_reaction_after_already_reacted() -> None:
    """The headline of #1756 -- not just ``_disabled``, but the next message.

    Slack redelivering one event must not cost every *other* conversation its
    status reactions.
    """
    seen: list[str] = []

    def responder(path: str, payload: dict[str, str]) -> dict[str, Any]:
        seen.append(payload["timestamp"])
        # Only the retried first message is already reacted to.
        if payload["timestamp"] == "111.000":
            return {"ok": False, "error": "already_reacted"}
        return {"ok": True}

    reactor, client = _reactor(responder)

    await reactor.received(_message(ts="111.000"))
    await reactor.received(_message(ts="222.000"))

    assert reactor._disabled is False
    assert _added_emoji(client) == ["white_check_mark", "white_check_mark"]
    assert seen == ["111.000", "222.000"]


@pytest.mark.asyncio
async def test_a_later_message_still_gets_its_reaction_after_no_reaction() -> None:
    """Same, through the remove path: a human un-reacting must cost nothing."""

    def responder(path: str, payload: dict[str, str]) -> dict[str, Any]:
        if path == "/reactions.remove":
            return {"ok": False, "error": "no_reaction"}
        return {"ok": True}

    reactor, client = _reactor(responder)

    first = _message(ts="111.000")
    await reactor.received(first)
    await reactor.completed(first)
    await reactor.received(_message(ts="222.000"))

    assert reactor._disabled is False
    assert _added_emoji(client) == ["white_check_mark", "white_check_mark"]
    assert _removed_emoji(client) == ["white_check_mark"]


# ── Tracked state after a benign no-op ───────────────────────────────────────


@pytest.mark.asyncio
async def test_already_reacted_still_tracks_the_mark_for_later_removal() -> None:
    """The emoji *is* on the message, so the reactor still owes a removal."""
    reactor, client = _reactor(_always({"ok": False, "error": "already_reacted"}))
    message = _message()

    await reactor.received(message)

    key = reactor._message_key(message)
    assert reactor._active[key] == [
        {"channel": "C1", "timestamp": "1700000000.000100", "name": "white_check_mark"}
    ]


@pytest.mark.asyncio
async def test_no_reaction_on_completion_leaves_no_tracked_entry_behind() -> None:
    """A stale ``_active`` key would leak once per message for the adapter's life."""

    def responder(path: str, payload: dict[str, str]) -> dict[str, Any]:
        if path == "/reactions.remove":
            return {"ok": False, "error": "no_reaction"}
        return {"ok": True}

    reactor, _client = _reactor(responder)
    message = _message()

    await reactor.received(message)
    await reactor.completed(message)

    assert reactor._active.get(reactor._message_key(message), []) == []
    assert reactor._disabled is False


@pytest.mark.asyncio
async def test_one_message_hitting_a_benign_error_does_not_touch_another() -> None:
    """Cross-message isolation: the buckets are per-``ts`` and stay that way."""

    def responder(path: str, payload: dict[str, str]) -> dict[str, Any]:
        if payload["timestamp"] == "111.000":
            return {"ok": False, "error": "already_reacted"}
        return {"ok": True}

    reactor, _client = _reactor(responder)
    benign = _message(ts="111.000")
    ordinary = _message(ts="222.000")

    await reactor.received(benign)
    await reactor.received(ordinary)
    await reactor.completed(benign)

    assert reactor._active.get("111.000", []) == []
    assert len(reactor._active["222.000"]) == 1
    assert reactor._disabled is False


# ── Whole lifecycles under benign errors ─────────────────────────────────────


@pytest.mark.asyncio
async def test_the_full_success_lifecycle_survives_benign_errors_throughout() -> None:
    """received → running → completed with every call answering "already so"."""

    def responder(path: str, payload: dict[str, str]) -> dict[str, Any]:
        if path == "/reactions.add":
            return {"ok": False, "error": "already_reacted"}
        return {"ok": False, "error": "no_reaction"}

    reactor, client = _reactor(responder)
    message = _message()

    await reactor.received(message)
    await reactor.running(message)
    await reactor.completed(message)

    assert reactor._disabled is False
    assert _added_emoji(client) == ["white_check_mark", "eyes"]
    assert _removed_emoji(client) == ["white_check_mark", "eyes"]
    assert reactor._active.get(reactor._message_key(message), []) == []


@pytest.mark.asyncio
async def test_failed_still_posts_the_outcome_mark_after_a_benign_clear() -> None:
    """``failed()`` clears progress marks first; that must not cost the ❌."""

    def responder(path: str, payload: dict[str, str]) -> dict[str, Any]:
        if path == "/reactions.remove":
            return {"ok": False, "error": "no_reaction"}
        return {"ok": True}

    reactor, client = _reactor(responder)
    message = _message()

    await reactor.received(message)
    await reactor.failed(message)

    assert reactor._disabled is False
    assert _added_emoji(client) == ["white_check_mark", "x"]


# ── Errors that mean the call did not happen still disable ───────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    ["ratelimited", "channel_not_found", "message_not_found", "invalid_name", "fatal_error"],
)
async def test_an_error_meaning_the_call_failed_still_disables(error: str) -> None:
    """The benign set must stay exactly two entries wide.

    ``ratelimited`` in particular means the reaction was *not* applied, so
    reporting success would leave the message unmarked with nothing to notice.
    """
    reactor, _client = _reactor(_always({"ok": False, "error": error}))

    await reactor.received(_message())

    assert reactor._disabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["missing_scope", "not_allowed_token_type"])
async def test_auth_shaped_errors_still_disable(error: str) -> None:
    reactor, _client = _reactor(_always({"ok": False, "error": error}))

    await reactor.received(_message())

    assert reactor._disabled is True


@pytest.mark.asyncio
async def test_a_403_still_disables() -> None:
    client = MagicMock()
    client.post = AsyncMock(return_value=_response({}, status_code=403))
    channel = MagicMock()
    channel._get_client.return_value = client
    reactor = SlackStatusReactor(channel, _SilentLog())

    await reactor.received(_message())

    assert reactor._disabled is True


@pytest.mark.asyncio
async def test_an_http_error_still_disables() -> None:
    """``raise_for_status`` fires before the body is ever inspected."""
    resp = _response({"ok": False, "error": "already_reacted"})
    resp.raise_for_status.side_effect = RuntimeError("500 Server Error")
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    channel = MagicMock()
    channel._get_client.return_value = client
    reactor = SlackStatusReactor(channel, _SilentLog())

    await reactor.received(_message())

    assert reactor._disabled is True


@pytest.mark.asyncio
async def test_a_benign_error_does_not_rescue_a_reactor_already_disabled() -> None:
    """Once off, the reactor stays off -- this change adds no recovery path."""
    responses = iter(
        [
            _response({"ok": False, "error": "ratelimited"}),
            _response({"ok": False, "error": "already_reacted"}),
        ]
    )
    client = MagicMock()
    client.post = AsyncMock(side_effect=lambda path, json: next(responses))
    channel = MagicMock()
    channel._get_client.return_value = client
    reactor = SlackStatusReactor(channel, _SilentLog())

    await reactor.received(_message(ts="111.000"))
    assert reactor._disabled is True

    await reactor.received(_message(ts="222.000"))

    assert reactor._disabled is True
    assert client.post.await_count == 1, "a disabled reactor must not call Slack again"
