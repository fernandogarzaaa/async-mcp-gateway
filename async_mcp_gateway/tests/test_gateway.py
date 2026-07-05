"""Integration test matrix for async-mcp-gateway."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import pytest
import pytest_asyncio

from app.main import app, limiter
from app.router import ResilientStreamRouter


class DenyingLimiter:
    """Limiter test double that rejects every request."""

    async def is_allowed(
        self,
        tenant_id: str,
        tokens: int,
        max_tpm: int,
        max_rpm: int,
    ) -> bool:
        return False


class AllowingLimiter:
    """Limiter test double that accepts every request."""

    async def is_allowed(
        self,
        tenant_id: str,
        tokens: int,
        max_tpm: int,
        max_rpm: int,
    ) -> bool:
        return True


@pytest.fixture
def tenant_headers() -> dict[str, str]:
    return {
        "X-Tenant-ID": "tenant-alpha",
        "Authorization": "Bearer alpha-secret-token",
    }


@pytest_asyncio.fixture
async def async_client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_rate_limited_tenant_receives_429(
    async_client: httpx.AsyncClient,
    tenant_headers: dict[str, str],
) -> None:
    with mock.patch.object(limiter, "is_allowed", new=DenyingLimiter().is_allowed):
        response = await async_client.post(
            "/v1/gateway/chat/completions",
            headers=tenant_headers,
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 429
    assert response.json() == {"error": {"code": "rate_limit_exceeded"}}


@pytest.mark.asyncio
async def test_primary_503_fails_over_to_backup_without_framework_error() -> None:
    requests_seen: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        requests_seen.append({"url": str(request.url), "payload": payload})
        if "primary.local" in str(request.url):
            return httpx.Response(503, text="primary unavailable")
        return httpx.Response(
            200,
            content=b'data: {"ok": true}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    router = ResilientStreamRouter(timeout_seconds=1.0)
    await router.client.aclose()
    router.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        frames = [
            frame
            async for frame in router.stream_response(
                {
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "hello"},
                    ],
                    "max_tokens": 16,
                    "stream": True,
                },
                {
                    "primary_provider_url": "https://primary.local/v1/chat/completions",
                    "backup_provider_url": "https://backup.local/v1/messages",
                },
            )
        ]
    finally:
        await router.close()

    assert len(requests_seen) == 2
    assert requests_seen[0]["url"] == "https://primary.local/v1/chat/completions"
    assert requests_seen[1]["url"] == "https://backup.local/v1/messages"
    assert requests_seen[1]["payload"]["system"] == "system"
    assert requests_seen[1]["payload"]["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    assert "".join(frames) == 'data: {"ok": true}\n\n'


@pytest.mark.asyncio
async def test_chat_route_streams_backup_after_primary_503(
    async_client: httpx.AsyncClient,
    tenant_headers: dict[str, str],
) -> None:
    async def fake_stream_response(
        payload: dict[str, Any],
        tenant_config: dict[str, Any],
    ) -> AsyncIterator[str]:
        await asyncio.sleep(0)
        yield "data: fallback\n\n"

    with (
        mock.patch.object(limiter, "is_allowed", new=AllowingLimiter().is_allowed),
        mock.patch("app.main.router.stream_response", new=fake_stream_response),
    ):
        response = await async_client.post(
            "/v1/gateway/chat/completions",
            headers=tenant_headers,
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == "data: fallback\n\n"
