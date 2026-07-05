"""Gateway integration tests with mocked Redis and upstream providers."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import ProviderConfig, Settings, TenantConfig
from app.main import app
from app.services.router import LLMRouter
from tests.conftest import FakeRateLimiter, FakeRouter


@pytest.mark.asyncio
async def test_rate_limit_exhaustion_returns_exact_429(
    gateway_client: httpx.AsyncClient,
    tenant_alpha_headers: dict[str, str],
    denied_decision: Any,
) -> None:
    app.state.rate_limiter = FakeRateLimiter(denied_decision)
    app.state.router = FakeRouter()

    response = await gateway_client.post(
        "/v1/chat/completions",
        headers=tenant_alpha_headers,
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "12"
    assert response.headers["X-RateLimit-Remaining-Requests"] == "0"
    assert response.json() == {
        "error": {
            "code": "rate_limit_exceeded",
            "message": "tenant quota exhausted",
        }
    }


@pytest.mark.asyncio
async def test_streaming_route_sets_sse_headers(
    gateway_client: httpx.AsyncClient,
    tenant_alpha_headers: dict[str, str],
) -> None:
    response = await gateway_client.post(
        "/v1/chat/completions",
        headers=tenant_alpha_headers,
        json={"stream": True, "messages": [{"role": "user", "content": "stream"}]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert b"data:" in response.content


@pytest.mark.asyncio
async def test_router_fails_over_from_openai_500_to_anthropic_payload_translation() -> (
    None
):
    captured_requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_requests.append(
            {
                "url": str(request.url),
                "payload": payload,
                "headers": dict(request.headers),
            }
        )
        if "openai.test" in str(request.url):
            return httpx.Response(500, json={"error": {"message": "upstream down"}})
        return httpx.Response(
            200,
            json={"id": "anthropic-ok", "content": [{"type": "text", "text": "ok"}]},
        )

    settings = Settings(
        failover_budget_ms=0,
        openai=ProviderConfig(
            base_url="https://openai.test",
            api_key=SecretStr("openai-key"),
            chat_path="/v1/chat/completions",
            stream_path="/v1/chat/completions",
            default_model="gpt-test",
        ),
        anthropic=ProviderConfig(
            base_url="https://anthropic.test",
            api_key=SecretStr("anthropic-key"),
            chat_path="/v1/messages",
            stream_path="/v1/messages",
            default_model="claude-test",
        ),
    )
    tenant = TenantConfig(
        tenant_id="tenant-failover",
        bearer_token=SecretStr("token"),
        rpm_limit=100,
        tpm_limit=100_000,
        provider_preference=["openai", "anthropic"],
    )
    router = LLMRouter(settings)
    await router._client.aclose()
    router._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=30.0
    )
    try:
        result = await router.create_chat_completion(
            {
                "model": "gpt-test",
                "messages": [
                    {"role": "system", "content": "system contract"},
                    {"role": "user", "content": "hello"},
                ],
                "max_tokens": 32,
            },
            tenant,
            {},
        )
    finally:
        await router.close()

    assert result["id"] == "anthropic-ok"
    assert len(captured_requests) == 2
    anthropic_payload = captured_requests[1]["payload"]
    assert captured_requests[0]["url"] == "https://openai.test/v1/chat/completions"
    assert captured_requests[1]["url"] == "https://anthropic.test/v1/messages"
    assert anthropic_payload["system"] == "system contract"
    assert anthropic_payload["messages"] == [{"role": "user", "content": "hello"}]
    assert anthropic_payload["max_tokens"] == 32
    assert captured_requests[1]["headers"]["x-api-key"] == "anthropic-key"
