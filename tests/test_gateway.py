"""Gateway integration tests with mocked Redis and upstream providers."""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

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


def _router_with(handler: Any) -> tuple[LLMRouter, TenantConfig]:
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
        tenant_id="tenant-4xx",
        bearer_token=SecretStr("token"),
        rpm_limit=100,
        tpm_limit=100_000,
        provider_preference=["openai", "anthropic"],
    )
    router = LLMRouter(settings)
    router._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=30.0
    )
    return router, tenant


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 413, 422])
async def test_router_does_not_fail_over_on_client_errors(status: int) -> None:
    from app.services.router import ProviderClientError

    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(status, json={"error": {"message": "bad max_tokens"}})

    router, tenant = _router_with(handler)
    try:
        with pytest.raises(ProviderClientError) as info:
            await router.create_chat_completion(
                {"messages": [{"role": "user", "content": "hi"}]}, tenant, {}
            )
    finally:
        await router.close()

    assert calls == ["https://openai.test/v1/chat/completions"]
    assert info.value.status_code == status
    assert info.value.to_payload()["error"] == {
        "code": "upstream_rejected",
        "message": "bad max_tokens",
        "provider": "openai",
        "upstream_status": status,
    }


@pytest.mark.asyncio
async def test_router_still_fails_over_on_401() -> None:
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "openai.test" in str(request.url):
            return httpx.Response(401, json={"error": {"message": "bad key"}})
        return httpx.Response(200, json={"id": "anthropic-ok", "content": []})

    router, tenant = _router_with(handler)
    try:
        result = await router.create_chat_completion(
            {"messages": [{"role": "user", "content": "hi"}]}, tenant, {}
        )
    finally:
        await router.close()

    assert result["id"] == "anthropic-ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_stream_client_error_emits_one_error_event_without_failover() -> None:
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    router, tenant = _router_with(handler)
    chunks: list[bytes] = []
    try:
        async for chunk in router.stream_chat_completion(
            {"messages": [{"role": "user", "content": "hi"}]}, tenant, {}
        ):
            chunks.append(chunk)
    finally:
        await router.close()

    assert calls == ["https://openai.test/v1/chat/completions"]
    body = b"".join(chunks)
    assert b"upstream_rejected" in body
    assert b"bad request" in body


@pytest.mark.asyncio
async def test_chat_route_maps_client_error_to_upstream_status(
    gateway_client: httpx.AsyncClient,
    tenant_alpha_headers: dict[str, str],
) -> None:
    from app.services.router import ProviderClientError

    rejecting = FakeRouter()
    rejecting.create_chat_completion = mock.AsyncMock(
        side_effect=ProviderClientError("openai", 422, "messages must not be empty")
    )
    app.state.router = rejecting
    response = await gateway_client.post(
        "/v1/chat/completions",
        headers=tenant_alpha_headers,
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "upstream_rejected"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"error": {"message": "too long"}}', "too long"),
        (b'{"error": "plain string"}', "plain string"),
        (b'{"error": {"type": "x"}, "message": "top level"}', "top level"),
        (b'{"error": ["x"]}', '{"error": ["x"]}'),
        (b"<html>bad gateway</html>", "<html>bad gateway</html>"),
        (b"\xff\xfe", "\ufffd\ufffd"),
        (b"", "upstream rejected the request"),
        (b"x" * 600, "x" * 500),
    ],
)
def test_upstream_error_message_shapes(body: bytes, expected: str) -> None:
    from app.services.router import _upstream_error_message

    assert _upstream_error_message(body) == expected


def test_provider_client_error_str() -> None:
    from app.services.router import ProviderClientError

    err = ProviderClientError("anthropic", 413, "too big")
    assert str(err) == "anthropic rejected the request (413): too big"


@pytest.mark.asyncio
async def test_router_records_every_attempt_when_all_providers_fail() -> None:
    from app.services.router import ProviderRoutingError

    async def handler(request: httpx.Request) -> httpx.Response:
        if "openai.test" in str(request.url):
            return httpx.Response(503, json={"error": {"message": "down"}})
        raise httpx.ConnectError("refused", request=request)

    router, tenant = _router_with(handler)
    try:
        with pytest.raises(ProviderRoutingError) as info:
            await router.create_chat_completion(
                {"messages": [{"role": "user", "content": "hi"}]}, tenant, {}
            )
    finally:
        await router.close()

    attempts = info.value.attempts
    assert [a.provider for a in attempts] == ["openai", "anthropic"]
    assert attempts[0].status_code == 503
    assert attempts[1].status_code is None and "refused" in (attempts[1].error or "")


@pytest.mark.asyncio
async def test_stream_all_providers_down_yields_upstream_unavailable() -> None:
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "openai.test" in str(request.url):
            return httpx.Response(500, json={"error": {"message": "down"}})
        raise httpx.ConnectError("refused", request=request)

    router, tenant = _router_with(handler)
    chunks: list[bytes] = []
    try:
        async for chunk in router.stream_chat_completion(
            {"messages": [{"role": "user", "content": "hi"}]}, tenant, {}
        ):
            chunks.append(chunk)
    finally:
        await router.close()

    assert len(calls) == 2
    body = b"".join(chunks)
    assert b"upstream_unavailable" in body
    assert b"openai:500" in body


@pytest.mark.asyncio
async def test_keyless_local_provider_gets_no_auth_header() -> None:
    """Ollama / vLLM / llama.cpp need no key; an empty "Bearer " header is illegal."""

    seen: list[dict[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.headers))
        if request.headers.get("accept") == "text/event-stream":
            return httpx.Response(
                200,
                content=(
                    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                    b"data: [DONE]\n\n"
                ),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"id": "local-ok", "choices": []})

    settings = Settings(
        failover_budget_ms=0,
        local=ProviderConfig(
            base_url="http://ollama.test",
            chat_path="/v1/chat/completions",
            stream_path="/v1/chat/completions",
            default_model="llama3",
        ),
        anthropic=ProviderConfig(
            base_url="https://anthropic.test",
            chat_path="/v1/messages",
            stream_path="/v1/messages",
            default_model="claude-test",
        ),
    )
    tenant = TenantConfig(
        tenant_id="tenant-local",
        bearer_token=SecretStr("token"),
        rpm_limit=100,
        tpm_limit=100_000,
        provider_preference=["local", "anthropic"],
    )
    router = LLMRouter(settings)
    await router._client.aclose()
    router._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=30.0
    )
    try:
        result = await router.create_chat_completion(
            {"messages": [{"role": "user", "content": "hi"}]}, tenant, {}
        )
        chunks = [
            chunk
            async for chunk in router.stream_chat_completion(
                {"messages": [{"role": "user", "content": "hi"}], "stream": True},
                tenant,
                {},
            )
        ]
    finally:
        await router.close()

    assert result["id"] == "local-ok"
    assert b"[DONE]" in b"".join(chunks)
    assert len(seen) == 2
    for headers in seen:
        assert "authorization" not in headers
        assert "x-api-key" not in headers
    # Keyed providers still authenticate.
    keyed = LLMRouter(
        Settings(
            openai=ProviderConfig(
                base_url="https://openai.test",
                api_key=SecretStr("o-key"),
                chat_path="/v1/chat/completions",
                stream_path="/v1/chat/completions",
                default_model="gpt-test",
            ),
            anthropic=ProviderConfig(
                base_url="https://anthropic.test",
                api_key=SecretStr("a-key"),
                chat_path="/v1/messages",
                stream_path="/v1/messages",
                default_model="claude-test",
            ),
        )
    )
    try:
        assert keyed._headers_for("openai", {}, stream=False)["Authorization"] == (
            "Bearer o-key"
        )
        assert keyed._headers_for("anthropic", {}, stream=False)["x-api-key"] == "a-key"
    finally:
        await keyed.close()
