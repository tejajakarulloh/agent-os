"""Transient HTTP retries on Slack outbound calls.

``retry_request`` is used as-is (same helper Discord already wraps). This
file does not change the helper: a consumed file handle is avoided by
sending bytes on the upload POST, not by seeking inside ``retry_request``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from agentos.channels.contract import ChannelSendStatus
from agentos.channels.slack import SlackChannel
from agentos.channels.types import OutgoingMessage


class _FakeResp:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "https://slack.com"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("agentos.channels._util.asyncio.sleep", _sleep)


def _channel() -> SlackChannel:
    channel = SlackChannel(token="xoxb-test", slack_channel_id="C-default")
    return channel


@pytest.mark.asyncio
async def test_slack_send_retries_on_503_then_succeeds() -> None:
    calls = 0

    class FakeClient:
        async def post(self, path: str, **kwargs: Any) -> _FakeResp:
            nonlocal calls
            calls += 1
            if calls < 3:
                return _FakeResp(503, {"ok": False, "error": "server_error"})
            return _FakeResp(200, {"ok": True, "ts": "123.456"})

    channel = _channel()
    channel._client = FakeClient()  # type: ignore[assignment]

    await channel.send(OutgoingMessage(content="Hello", reply_to="C-default"))
    assert calls == 3


@pytest.mark.asyncio
async def test_slack_send_retries_on_429() -> None:
    calls = 0

    class FakeClient:
        async def post(self, path: str, **kwargs: Any) -> _FakeResp:
            nonlocal calls
            calls += 1
            if calls == 1:
                return _FakeResp(429, {"ok": False, "error": "rate_limited"})
            return _FakeResp(200, {"ok": True, "ts": "123.456"})

    channel = _channel()
    channel._client = FakeClient()  # type: ignore[assignment]

    await channel.send(OutgoingMessage(content="Hello", reply_to="C-default"))
    assert calls == 2


@pytest.mark.asyncio
async def test_slack_send_retries_on_read_timeout() -> None:
    calls = 0

    class FakeClient:
        async def post(self, path: str, **kwargs: Any) -> _FakeResp:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ReadTimeout("Timeout")
            return _FakeResp(200, {"ok": True, "ts": "123.456"})

    channel = _channel()
    channel._client = FakeClient()  # type: ignore[assignment]

    await channel.send(OutgoingMessage(content="Hello", reply_to="C-default"))
    assert calls == 2


@pytest.mark.asyncio
async def test_slack_send_does_not_retry_fatal_400() -> None:
    calls = 0

    class FakeClient:
        async def post(self, path: str, **kwargs: Any) -> _FakeResp:
            nonlocal calls
            calls += 1
            return _FakeResp(400, {"ok": False, "error": "invalid_payload"})

    channel = _channel()
    channel._client = FakeClient()  # type: ignore[assignment]

    with pytest.raises(httpx.HTTPStatusError):
        await channel.send(OutgoingMessage(content="Hello", reply_to="C-default"))
    assert calls == 1


@pytest.mark.asyncio
async def test_slack_send_does_not_retry_fatal_401() -> None:
    calls = 0

    class FakeClient:
        async def post(self, path: str, **kwargs: Any) -> _FakeResp:
            nonlocal calls
            calls += 1
            return _FakeResp(401, {"ok": False, "error": "invalid_auth"})

    channel = _channel()
    channel._client = FakeClient()  # type: ignore[assignment]

    with pytest.raises(httpx.HTTPStatusError):
        await channel.send(OutgoingMessage(content="Hello", reply_to="C-default"))
    assert calls == 1


@pytest.mark.asyncio
async def test_slack_edit_retries_on_503() -> None:
    calls = 0

    class FakeClient:
        async def post(self, path: str, **kwargs: Any) -> _FakeResp:
            nonlocal calls
            calls += 1
            if calls == 1:
                return _FakeResp(503, {"ok": False})
            return _FakeResp(200, {"ok": True})

    channel = _channel()
    channel._client = FakeClient()  # type: ignore[assignment]

    await channel.edit("111.222", "updated")
    assert calls == 2


@pytest.mark.asyncio
async def test_slack_send_file_retries_upload_with_bytes_body(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("file content", encoding="utf-8")
    upload_attempts = 0
    seen_bodies: list[bytes] = []

    class FakeClient:
        async def post(self, path: str, **kwargs: Any) -> _FakeResp:
            nonlocal upload_attempts
            if path == "/files.getUploadURLExternal":
                return _FakeResp(
                    200,
                    {"ok": True, "upload_url": "https://upload.test", "file_id": "F1"},
                )
            if path == "https://upload.test":
                upload_attempts += 1
                body = kwargs["files"]["file"][1]
                assert isinstance(body, bytes)
                seen_bodies.append(body)
                if upload_attempts == 1:
                    raise httpx.ReadTimeout("Upload Timeout")
                return _FakeResp(200, {"ok": True})
            if path == "/files.completeUploadExternal":
                return _FakeResp(200, {"ok": True, "files": [{"id": "F1"}]})
            return _FakeResp(400, {"ok": False})

    channel = _channel()
    channel._client = FakeClient()  # type: ignore[assignment]

    result = await channel.send_file("C-target", str(file_path), content="done")
    assert result.status is ChannelSendStatus.SENT
    assert upload_attempts == 2
    assert seen_bodies == [b"file content", b"file content"]
