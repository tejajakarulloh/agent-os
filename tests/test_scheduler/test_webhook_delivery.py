"""Webhook delivery mode.

``DeliveryMode.WEBHOOK`` POSTs the finished-run event payload to
``DeliveryConfig.webhook_url``, optionally with a bearer token. URL is
validated up front and rejected at add time when malformed.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agentos.scheduler.delivery import (
    _WEBHOOK_TIMEOUT_SECONDS,
    DeliveryChain,
    status_detail,
    validate_webhook_url,
)
from agentos.scheduler.ops import SchedulerOps
from agentos.scheduler.payloads import make_agent_turn_payload
from agentos.scheduler.persistence import JobStore
from agentos.scheduler.types import (
    CronJob,
    DeliveryConfig,
    DeliveryMode,
    ScheduleKind,
    SessionTarget,
)
from agentos.tools import ssrf_client

# --- URL validation --------------------------------------------------------


def test_validate_webhook_url_accepts_http_and_https() -> None:
    validate_webhook_url("http://example.com/hook")
    validate_webhook_url("https://example.com/hook?x=1")


def test_validate_webhook_url_rejects_other_schemes() -> None:
    with pytest.raises(ValueError, match="http or https"):
        validate_webhook_url("ftp://example.com/x")
    with pytest.raises(ValueError, match="http or https"):
        validate_webhook_url("file:///tmp/x")


def test_validate_webhook_url_requires_hostname() -> None:
    with pytest.raises(ValueError, match="hostname"):
        validate_webhook_url("https:///nohost")


def test_validate_webhook_url_rejects_empty() -> None:
    with pytest.raises(ValueError, match="required"):
        validate_webhook_url("")


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.169.254/",
        "https://169.254.169.253/",
        "http://169.254.170.2/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://metadata.goog/",
    ],
)
def test_validate_webhook_url_rejects_cloud_metadata(url: str) -> None:
    """Cron webhooks may target localhost; they may not target IMDS."""
    with pytest.raises(ValueError, match="metadata"):
        validate_webhook_url(url)


def test_validate_webhook_url_still_allows_localhost() -> None:
    validate_webhook_url("http://127.0.0.1:5678/webhook")
    validate_webhook_url("http://localhost:8080/hook")


# --- ops.add validates webhook config -------------------------------------


async def test_ops_add_with_webhook_delivery_persists(tmp_path: Path) -> None:
    db = tmp_path / "cron.db"
    store = JobStore(str(db))
    await store.open()
    try:
        ops = SchedulerOps(store)
        delivery = DeliveryConfig(
            mode=DeliveryMode.WEBHOOK,
            webhook_url="https://hooks.example/cron",
            webhook_token="secret-bearer",
            best_effort=True,
        )
        job = await ops.add(
            name="hook",
            schedule_kind=ScheduleKind.CRON,
            schedule_value="*/5 * * * *",
            handler_key="agent_run",
            payload=make_agent_turn_payload("brief"),
            session_target=SessionTarget.ISOLATED,
            delivery=delivery,
        )
        assert job.delivery.mode == DeliveryMode.WEBHOOK
        assert job.delivery.webhook_url == "https://hooks.example/cron"
        assert job.delivery.webhook_token == "secret-bearer"
        assert job.delivery.best_effort is True

        reloaded = await store.get(job.id)
        assert reloaded is not None
        assert reloaded.delivery.mode == DeliveryMode.WEBHOOK
        assert reloaded.delivery.webhook_url == "https://hooks.example/cron"
        assert reloaded.delivery.webhook_token == "secret-bearer"
        assert reloaded.delivery.best_effort is True
    finally:
        await store.close()


async def test_ops_add_rejects_webhook_without_url(tmp_path: Path) -> None:
    db = tmp_path / "cron.db"
    store = JobStore(str(db))
    await store.open()
    try:
        ops = SchedulerOps(store)
        with pytest.raises(ValueError, match="webhook URL is required"):
            await ops.add(
                name="bad",
                schedule_kind=ScheduleKind.CRON,
                schedule_value="*/5 * * * *",
                handler_key="agent_run",
                payload=make_agent_turn_payload("x"),
                session_target=SessionTarget.ISOLATED,
                delivery=DeliveryConfig(mode=DeliveryMode.WEBHOOK, webhook_url=""),
            )
    finally:
        await store.close()


async def test_ops_add_rejects_metadata_webhook_url(tmp_path: Path) -> None:
    db = tmp_path / "cron.db"
    store = JobStore(str(db))
    await store.open()
    try:
        ops = SchedulerOps(store)
        with pytest.raises(ValueError, match="metadata"):
            await ops.add(
                name="imds",
                schedule_kind=ScheduleKind.CRON,
                schedule_value="*/5 * * * *",
                handler_key="agent_run",
                payload=make_agent_turn_payload("x"),
                session_target=SessionTarget.ISOLATED,
                delivery=DeliveryConfig(
                    mode=DeliveryMode.WEBHOOK,
                    webhook_url="http://169.254.169.254/latest/meta-data/",
                ),
            )
    finally:
        await store.close()


async def test_ops_add_rejects_webhook_with_bad_scheme(tmp_path: Path) -> None:
    db = tmp_path / "cron.db"
    store = JobStore(str(db))
    await store.open()
    try:
        ops = SchedulerOps(store)
        with pytest.raises(ValueError, match="http or https"):
            await ops.add(
                name="bad",
                schedule_kind=ScheduleKind.CRON,
                schedule_value="*/5 * * * *",
                handler_key="agent_run",
                payload=make_agent_turn_payload("x"),
                session_target=SessionTarget.ISOLATED,
                delivery=DeliveryConfig(
                    mode=DeliveryMode.WEBHOOK, webhook_url="ftp://example.com/x"
                ),
            )
    finally:
        await store.close()


async def test_ops_add_allows_webhook_on_main_target(tmp_path: Path) -> None:
    """Webhook delivery is permitted for any sessionTarget, including main."""
    db = tmp_path / "cron.db"
    store = JobStore(str(db))
    await store.open()
    try:
        from agentos.scheduler.payloads import make_system_event_payload

        ops = SchedulerOps(store)
        job = await ops.add(
            name="main-hook",
            schedule_kind=ScheduleKind.CRON,
            schedule_value="*/5 * * * *",
            handler_key="system_event",
            payload=make_system_event_payload("reminder"),
            session_target=SessionTarget.MAIN,
            delivery=DeliveryConfig(
                mode=DeliveryMode.WEBHOOK,
                webhook_url="https://hooks.example/main",
            ),
        )
        assert job.delivery.mode == DeliveryMode.WEBHOOK
        reloaded = await store.get(job.id)
        assert reloaded is not None
        assert reloaded.delivery.mode == DeliveryMode.WEBHOOK
        assert reloaded.delivery.webhook_url == "https://hooks.example/main"
    finally:
        await store.close()


# --- DeliveryChain webhook dispatch ---------------------------------------


def _webhook_job(url: str, token: str = "") -> CronJob:
    return CronJob(
        id="job-1",
        name="hook",
        cron_expr="*/5 * * * *",
        handler_key="agent_run",
        payload={"kind": "agent_turn", "task": "x", "agent_id": "main"},
        session_target=SessionTarget.ISOLATED,
        delivery=DeliveryConfig(
            mode=DeliveryMode.WEBHOOK,
            webhook_url=url,
            webhook_token=token,
        ),
    )


class _RecordingAsyncClient:
    """Capture httpx.AsyncClient.post calls for assertion."""

    instances: list[_RecordingAsyncClient] = []

    def __init__(self, *, timeout=None, **_kw) -> None:
        self.timeout = timeout
        self.posts: list[dict] = []
        _RecordingAsyncClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers or {}})

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                return None

        return _Resp()


async def test_deliver_webhook_posts_json_with_bearer(monkeypatch) -> None:
    _RecordingAsyncClient.instances.clear()

    monkeypatch.setattr(httpx, "AsyncClient", _RecordingAsyncClient)

    chain = DeliveryChain()
    status = await chain._deliver_webhook(
        _webhook_job("https://hooks.example/cron", token="abc"),
        text="summary text",
    )
    assert status == "delivered"
    assert _RecordingAsyncClient.instances, "AsyncClient was not constructed"
    inst = _RecordingAsyncClient.instances[-1]
    assert inst.posts, "no POST issued"
    post = inst.posts[-1]
    assert post["url"] == "https://hooks.example/cron"
    assert post["json"]["jobId"] == "job-1"
    assert post["json"]["summary"] == "summary text"
    assert post["headers"]["Content-Type"] == "application/json"
    assert post["headers"]["Authorization"] == "Bearer abc"


async def test_deliver_webhook_omits_authorization_when_no_token(monkeypatch) -> None:
    _RecordingAsyncClient.instances.clear()

    monkeypatch.setattr(httpx, "AsyncClient", _RecordingAsyncClient)

    chain = DeliveryChain()
    status = await chain._deliver_webhook(
        _webhook_job("https://hooks.example/cron"),
        text="x",
    )
    assert status == "delivered"
    inst = _RecordingAsyncClient.instances[-1]
    assert "Authorization" not in inst.posts[-1]["headers"]


async def test_deliver_webhook_returns_failed_on_http_error(monkeypatch, no_backoff) -> None:
    class _ErrorClient(_RecordingAsyncClient):
        async def post(self, url, json=None, headers=None):
            self.posts.append({"url": url, "json": json, "headers": headers or {}})

            class _Resp:
                status_code = 500

                def raise_for_status(self):
                    raise RuntimeError("HTTP 500")

            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _ErrorClient)
    _RecordingAsyncClient.instances.clear()

    chain = DeliveryChain()
    status = await chain._deliver_webhook(
        _webhook_job("https://hooks.example/cron"),
        text="x",
    )
    assert status == "delivery_failed"
    # A 5xx is transient: the initial attempt plus retry_request's max_retries=3.
    assert len(_RecordingAsyncClient.instances[-1].posts) == 4


# --- transient-failure retries (issue #469) --------------------------------


@pytest.fixture
def no_backoff():
    """Collapse ``retry_request``'s sleeps so retry assertions stay fast."""
    with patch("agentos.channels._util.asyncio.sleep", new=AsyncMock()) as sleep:
        yield sleep


def _scripted_httpx(monkeypatch, responses):
    """Install a fake ``httpx`` whose POSTs replay ``responses`` in order."""

    class _ScriptedClient(_RecordingAsyncClient):
        async def post(self, url, json=None, headers=None):
            self.posts.append({"url": url, "json": json, "headers": headers or {}})
            item = responses[min(len(self.posts) - 1, len(responses) - 1)]
            if isinstance(item, Exception):
                raise item
            return item

    monkeypatch.setattr(httpx, "AsyncClient", _ScriptedClient)
    _RecordingAsyncClient.instances.clear()


def _webhook_response(status_code: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers=headers,
        request=httpx.Request("POST", "https://hooks.example/cron"),
    )


async def test_deliver_webhook_retries_transient_5xx_then_succeeds(monkeypatch, no_backoff) -> None:
    _scripted_httpx(monkeypatch, [_webhook_response(503), _webhook_response(200)])

    chain = DeliveryChain()
    status = await chain._deliver_webhook(
        _webhook_job("https://hooks.example/cron"),
        text="x",
    )

    assert status == "delivered"
    assert len(_RecordingAsyncClient.instances[-1].posts) == 2


async def test_deliver_webhook_retries_connect_error_then_succeeds(monkeypatch, no_backoff) -> None:
    _scripted_httpx(monkeypatch, [httpx.ConnectError("refused"), _webhook_response(200)])

    chain = DeliveryChain()
    status = await chain._deliver_webhook(
        _webhook_job("https://hooks.example/cron"),
        text="x",
    )

    assert status == "delivered"
    assert len(_RecordingAsyncClient.instances[-1].posts) == 2


async def test_deliver_webhook_does_not_retry_fatal_status(monkeypatch, no_backoff) -> None:
    """A 400/401 is the receiver's verdict, not a blip — fail on the first try."""
    for status_code in (400, 401):
        _scripted_httpx(monkeypatch, [_webhook_response(status_code)])

        chain = DeliveryChain()
        status = await chain._deliver_webhook(
            _webhook_job("https://hooks.example/cron", token="abc"),
            text="x",
        )

        assert status == "delivery_failed"
        assert len(_RecordingAsyncClient.instances[-1].posts) == 1
    no_backoff.assert_not_awaited()


async def test_deliver_webhook_honours_retry_after_on_429(monkeypatch, no_backoff) -> None:
    _scripted_httpx(
        monkeypatch,
        [_webhook_response(429, {"Retry-After": "2"}), _webhook_response(200)],
    )

    chain = DeliveryChain()
    status = await chain._deliver_webhook(
        _webhook_job("https://hooks.example/cron"),
        text="x",
    )

    assert status == "delivered"
    no_backoff.assert_awaited_once_with(2.0)


# --- connect-time SSRF guard (issue #725) ----------------------------------
#
# ``validate_webhook_url`` resolves the hostname once and checks the answer;
# httpx then resolves it again when it opens the socket. A short-TTL rebinding
# domain can answer with a public address for the check and with
# ``169.254.169.254`` for the connect, so the job payload lands on the cloud
# metadata endpoint. The guarded client dials the address that was validated.

METADATA_IP = "169.254.169.254"


def _sequence_resolver(*ips: str):
    """``getaddrinfo`` double that answers with a different address per call.

    IP literals resolve to themselves so the wrapped backend can still dial the
    literal the guard approved.
    """
    calls: list[str] = []

    def resolver(host, port=None, *args, **kwargs):
        calls.append(host)
        try:
            ipaddress.ip_address(str(host).strip("[]"))
        except ValueError:
            index = min(len([c for c in calls if c == host]) - 1, len(ips) - 1)
            answer = ips[index]
        else:
            answer = str(host)
        family = socket.AF_INET6 if ipaddress.ip_address(answer).version == 6 else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (answer, port or 80))]

    resolver.calls = calls  # type: ignore[attr-defined]
    return resolver


class _LocalWebhookServer:
    """Minimal HTTP/1.1 responder on an ephemeral loopback port.

    Answers every connection it is given rather than just the first, so a
    regression that reintroduces retries fails on the assertion instead of
    hanging on an unanswered socket, and takes a timeout on the accept loop so
    the thread cannot outlive the test on any platform.
    """

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._sock.settimeout(0.5)
        self.port = self._sock.getsockname()[1]
        self.requests: list[bytes] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:  # pragma: no cover - the socket closed under us
                return
            with conn:
                conn.settimeout(5.0)
                try:
                    self.requests.append(conn.recv(65536))
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                        b"Content-Length: 2\r\nConnection: close\r\n\r\nok"
                    )
                    conn.shutdown(socket.SHUT_WR)
                except OSError:  # pragma: no cover - client went away
                    pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._sock.close()


async def test_deliver_webhook_uses_the_metadata_only_connect_guard(monkeypatch) -> None:
    """The POST goes through ``ssrf_guarded_client``, not a bare client."""
    recorded: dict[str, object] = {}

    def _fake_guarded_client(*, validator, **kwargs):
        recorded["validator"] = validator
        recorded["kwargs"] = kwargs
        return _RecordingAsyncClient(**kwargs)

    monkeypatch.setattr(ssrf_client, "ssrf_guarded_client", _fake_guarded_client)
    _RecordingAsyncClient.instances.clear()

    chain = DeliveryChain()
    status = await chain._deliver_webhook(
        _webhook_job("https://hooks.example/cron"),
        text="x",
    )

    assert status == "delivered"
    # Metadata-only, not the full fetch policy: cron webhooks are pointed at
    # n8n on localhost and LAN boxes on purpose, and must keep working.
    assert recorded["validator"] is ssrf_client.validate_metadata_only_address
    assert recorded["kwargs"]["timeout"] == _WEBHOOK_TIMEOUT_SECONDS


async def test_deliver_webhook_blocks_dns_rebinding_to_metadata(monkeypatch, no_backoff) -> None:
    """A name that rebinds to IMDS after URL validation never gets the payload."""
    resolver = _sequence_resolver("127.0.0.1", METADATA_IP)
    monkeypatch.setattr(socket, "getaddrinfo", resolver)

    chain = DeliveryChain()
    status = await chain._deliver_webhook(
        _webhook_job("http://rebind.example/hook"),
        text="secret summary",
    )

    assert status == "delivery_failed"
    assert "metadata" in status_detail(status).lower()
    # The block has to come from the connect guard, not from validate_webhook_url:
    # SSRFBlockedError is a ValueError, so a check-time block would also mention
    # metadata — but it would carry the "invalid webhook URL" prefix, and it would
    # leave the second resolution unmade.
    assert not status_detail(status).startswith("invalid webhook URL")
    assert resolver.calls == ["rebind.example", "rebind.example"]
    # Blocked at connect time, so retry_request never got a transient error to
    # sleep on — a rebinding target must not be retried into.
    no_backoff.assert_not_awaited()


async def test_deliver_webhook_still_reaches_a_loopback_hook(no_backoff) -> None:
    """The metadata floor keeps localhost hooks (n8n and friends) working."""
    server = _LocalWebhookServer()
    try:
        chain = DeliveryChain()
        status = await chain._deliver_webhook(
            _webhook_job(f"http://127.0.0.1:{server.port}/hook"),
            text="summary text",
        )
    finally:
        server.close()

    assert status == "delivered"
    assert server.requests and b"POST /hook" in server.requests[0]
