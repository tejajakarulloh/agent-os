"""Unit tests for ``retry_request`` utility in ``agentos.channels._util``."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agentos.channels._util import retry_request

_REQ = httpx.Request("POST", "https://api.example.test/v1")


def _resp(status_code: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status_code, json={"ok": True}, headers=headers, request=_REQ)


@pytest.fixture
def mock_sleep():
    with patch("agentos.channels._util.asyncio.sleep", new=AsyncMock()) as sleep:
        yield sleep


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timeout_exc",
    [
        httpx.ConnectTimeout("dns/connect timed out", request=_REQ),
        httpx.ReadTimeout("read timed out", request=_REQ),
        httpx.WriteTimeout("write timed out", request=_REQ),
        httpx.PoolTimeout("pool acquired timed out", request=_REQ),
    ],
)
async def test_retry_request_retries_all_timeout_exceptions(
    timeout_exc: httpx.TimeoutException, mock_sleep: AsyncMock
) -> None:
    """ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout are all retried."""
    func = AsyncMock(side_effect=[timeout_exc, _resp(200)])

    res = await retry_request(func, max_retries=2, base_delay=0.1)

    assert res.status_code == 200
    assert func.await_count == 2
    assert mock_sleep.await_count == 1


@pytest.mark.asyncio
async def test_retry_request_retries_connect_error(mock_sleep: AsyncMock) -> None:
    func = AsyncMock(
        side_effect=[httpx.ConnectError("connection refused", request=_REQ), _resp(200)]
    )

    res = await retry_request(func, max_retries=2, base_delay=0.1)

    assert res.status_code == 200
    assert func.await_count == 2
    assert mock_sleep.await_count == 1


@pytest.mark.asyncio
async def test_retry_request_raises_after_timeouts_exhausted(mock_sleep: AsyncMock) -> None:
    exc = httpx.ConnectTimeout("connect timed out", request=_REQ)
    func = AsyncMock(side_effect=[exc, exc, exc, exc])

    with pytest.raises(httpx.ConnectTimeout):
        await retry_request(func, max_retries=3, base_delay=0.1)

    assert func.await_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("header_val", "expected_sleep"),
    [
        ("3.5", 3.5),
        ("5", 5.0),
        ("0", 0.0),
    ],
)
async def test_retry_request_honours_numeric_retry_after_header(
    header_val: str, expected_sleep: float, mock_sleep: AsyncMock
) -> None:
    func = AsyncMock(side_effect=[_resp(429, headers={"Retry-After": header_val}), _resp(200)])

    res = await retry_request(func, max_retries=2, base_delay=1.0)

    assert res.status_code == 200
    assert func.await_count == 2
    mock_sleep.assert_awaited_once_with(expected_sleep)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_header",
    [
        "Wed, 21 Oct 2026 07:28:00 GMT",  # RFC 7231 HTTP-date format
        "not-a-number",
        "",
        "None",
    ],
)
async def test_retry_request_handles_non_numeric_retry_after_without_crashing(
    invalid_header: str, mock_sleep: AsyncMock
) -> None:
    """Non-numeric Retry-After falls back to exponential backoff instead of raising ValueError."""
    func = AsyncMock(side_effect=[_resp(429, headers={"Retry-After": invalid_header}), _resp(200)])

    res = await retry_request(func, max_retries=2, base_delay=1.0)

    assert res.status_code == 200
    assert func.await_count == 2
    mock_sleep.assert_awaited_once_with(1.0)  # base_delay * (2 ** 0)


@pytest.mark.asyncio
async def test_retry_request_handles_missing_retry_after_header(mock_sleep: AsyncMock) -> None:
    func = AsyncMock(side_effect=[_resp(429), _resp(200)])

    res = await retry_request(func, max_retries=2, base_delay=1.0)

    assert res.status_code == 200
    assert func.await_count == 2
    mock_sleep.assert_awaited_once_with(1.0)
