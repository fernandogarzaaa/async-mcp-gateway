"""Shared async pytest fixtures for the AI gateway test suite."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any
from unittest import mock

import httpx
import pytest
import pytest_asyncio

from app.core.config import TenantConfig
from app.main import app
from app.services.rate_limiter import RateLimitDecision


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


class MockRedis:
    """Small async Redis substitute for isolated tests."""

    def __init__(self) -> None:
        self.closed = False
        self.loaded_scripts: list[str] = []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def script_load(self, script: str) -> str:
        self.loaded_scripts.append(script)
        return "mock-sha"

    async def evalsha(self, sha: str, keys: int, *args: object) -> list[object]:
        self.calls.append(("evalsha", (sha, keys, *args)))
        return [1, 999, 99999, 0]

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        self.closed = True


@dataclass(slots=True)
class FakeRateLimiter:
    decision: RateLimitDecision

    async def check(
        self, tenant_id: str, tenant: TenantConfig, estimated_tokens: int
    ) -> RateLimitDecision:
        return self.decision

    async def ping(self) -> bool:
        return True


class FakeRouter:
    def __init__(self) -> None:
        self.create_chat_completion = mock.AsyncMock(
            return_value={"id": "test", "choices": []}
        )
        self.stream_calls: list[dict[str, Any]] = []

    async def stream_chat_completion(
        self,
        payload: dict[str, Any],
        tenant: TenantConfig,
        inbound_headers: httpx.Headers,
    ) -> AsyncIterator[bytes]:
        self.stream_calls.append(
            {"payload": payload, "tenant": tenant, "headers": inbound_headers}
        )
        yield b'data: {"delta":"hello"}\n\n'
        yield b"data: [DONE]\n\n"


@pytest.fixture
def mock_redis() -> MockRedis:
    return MockRedis()


@pytest.fixture
def tenant_alpha_headers() -> dict[str, str]:
    return {
        "X-Tenant-ID": "tenant-alpha",
        "Authorization": "Bearer alpha-secret-token",
    }


@pytest.fixture
def allowed_decision() -> RateLimitDecision:
    return RateLimitDecision(
        allowed=True,
        rpm_remaining=119,
        tpm_remaining=119_900,
        retry_after_seconds=0,
    )


@pytest.fixture
def denied_decision() -> RateLimitDecision:
    return RateLimitDecision(
        allowed=False,
        rpm_remaining=0,
        tpm_remaining=119_900,
        retry_after_seconds=12,
    )


@pytest_asyncio.fixture
async def gateway_client(
    allowed_decision: RateLimitDecision,
) -> AsyncIterator[httpx.AsyncClient]:
    app.state.rate_limiter = FakeRateLimiter(allowed_decision)
    app.state.router = FakeRouter()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client
