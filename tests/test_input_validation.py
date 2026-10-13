"""Malformed client input is a 400, never a 500 or an upstream call."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.main import app
from app.services.router import LLMRouter


@pytest.mark.asyncio
async def test_non_ascii_bearer_token_returns_400(
    gateway_client: httpx.AsyncClient,
) -> None:
    response = await gateway_client.post(
        "/v1/chat/completions",
        headers={
            b"X-Tenant-ID": b"tenant-alpha",
            b"Authorization": "Bearer t\u00f6k\u00e9n".encode(),
        },
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_wrong_ascii_token_still_401(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.post(
        "/v1/chat/completions",
        headers={"X-Tenant-ID": "tenant-alpha", "Authorization": "Bearer nope"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["abc", "100", 1.5, True, 0, -3, [], {}])
async def test_invalid_max_tokens_returns_400(
    gateway_client: httpx.AsyncClient,
    tenant_alpha_headers: dict[str, str],
    value: Any,
) -> None:
    response = await gateway_client.post(
        "/v1/chat/completions",
        headers=tenant_alpha_headers,
        json={"max_tokens": value, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    assert "max_tokens" in response.json()["detail"]
    app.state.router.create_chat_completion.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, 1, 256])
async def test_valid_max_tokens_is_forwarded(
    gateway_client: httpx.AsyncClient,
    tenant_alpha_headers: dict[str, str],
    value: Any,
) -> None:
    response = await gateway_client.post(
        "/v1/chat/completions",
        headers=tenant_alpha_headers,
        json={"max_tokens": value, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200


@pytest.mark.parametrize("value", ["abc", None, True, 0, 2.5])
def test_anthropic_translation_never_crashes_on_max_tokens(value: Any) -> None:
    router = LLMRouter(Settings(_env_file=None))  # type: ignore[call-arg]
    out = router._to_anthropic_payload({"max_tokens": value, "messages": []})
    assert out["max_tokens"] == 1024
    assert router._to_anthropic_payload({"max_tokens": 64})["max_tokens"] == 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"messages": "hello"},
        {"messages": []},
        {"messages": {"role": "user"}},
        {"messages": ["hi"]},
        {"messages": [{"content": "no role"}]},
        {"messages": [{"role": "", "content": "x"}]},
        {"messages": [{"role": 5, "content": "x"}]},
        {"messages": [{"role": "user", "content": 42}]},
        {"messages": [{"role": "user", "content": {"a": 1}}]},
        {"messages": None, "prompt": 3},
    ],
)
async def test_malformed_messages_return_400_without_upstream_call(
    gateway_client: httpx.AsyncClient,
    tenant_alpha_headers: dict[str, str],
    body: dict[str, Any],
) -> None:
    response = await gateway_client.post(
        "/v1/chat/completions", headers=tenant_alpha_headers, json=body
    )
    assert response.status_code == 400
    assert "messages" in response.json()["detail"]
    app.state.router.create_chat_completion.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "hello"},
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}]}]},
        {"messages": [{"role": "assistant", "content": None, "tool_calls": []}]},
    ],
)
async def test_well_formed_messages_are_forwarded(
    gateway_client: httpx.AsyncClient,
    tenant_alpha_headers: dict[str, str],
    body: dict[str, Any],
) -> None:
    response = await gateway_client.post(
        "/v1/chat/completions", headers=tenant_alpha_headers, json=body
    )
    assert response.status_code == 200
    app.state.router.create_chat_completion.assert_called_once()
