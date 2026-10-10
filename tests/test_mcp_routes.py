"""HTTP-level tests for the /v1/mcp routes backed by the real stdio supervisor."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from app.main import app, load_mcp_config
from app.models.schemas import (
    MCPHealthCheckConfig,
    MCPServerConfig,
    MCPSupervisorConfig,
    TenantMCPPolicy,
)
from app.services.mcp_supervisor import MCPProcessPoolManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCK_MCP = PROJECT_ROOT / "mock_mcp_server.py"


@pytest_asyncio.fixture
async def mcp_client(
    gateway_client: httpx.AsyncClient,
) -> AsyncIterator[httpx.AsyncClient]:
    server = MCPServerConfig(
        name="mock",
        command=[sys.executable, "-u", str(MOCK_MCP)],
        request_timeout_seconds=10.0,
        health_check=MCPHealthCheckConfig(enabled=False),
    )
    manager = MCPProcessPoolManager(
        MCPSupervisorConfig(
            servers={"mock": server},
            tenant_policies={
                "tenant-beta": TenantMCPPolicy(
                    tenant_id="tenant-beta", allowed_servers=set()
                )
            },
        )
    )
    await manager.start()
    app.state.mcp = manager
    try:
        yield gateway_client
    finally:
        app.state.mcp = None
        await manager.close()


@pytest.mark.asyncio
async def test_mcp_round_trip_and_notification(
    mcp_client: httpx.AsyncClient, tenant_alpha_headers: dict[str, str]
) -> None:
    listed = await mcp_client.get("/v1/mcp/servers", headers=tenant_alpha_headers)
    assert listed.json() == {"servers": ["mock"]}

    init = await mcp_client.post(
        "/v1/mcp/mock",
        headers=tenant_alpha_headers,
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    assert init.status_code == 200
    assert init.json()["result"]["serverInfo"]["name"] == "mock-mcp"

    note = await mcp_client.post(
        "/v1/mcp/mock",
        headers=tenant_alpha_headers,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert note.status_code == 202

    call = await mcp_client.post(
        "/v1/mcp/mock",
        headers=tenant_alpha_headers,
        json={
            "jsonrpc": "2.0",
            "id": "c1",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"x": 1}},
        },
    )
    assert call.status_code == 200
    text = json.loads(call.json()["result"]["content"][0]["text"])
    assert text == {"arguments": {"x": 1}, "tenant_id": "tenant-alpha", "tool": "echo"}


@pytest.mark.asyncio
async def test_mcp_errors_map_to_http_status(
    mcp_client: httpx.AsyncClient, tenant_alpha_headers: dict[str, str]
) -> None:
    unknown = await mcp_client.post(
        "/v1/mcp/nope",
        headers=tenant_alpha_headers,
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
    )
    assert unknown.status_code == 404

    missing_id = await mcp_client.post(
        "/v1/mcp/mock",
        headers=tenant_alpha_headers,
        json={"jsonrpc": "2.0", "method": "tools/list"},
    )
    assert missing_id.status_code == 400

    beta = await mcp_client.post(
        "/v1/mcp/mock",
        headers={
            "X-Tenant-ID": "tenant-beta",
            "Authorization": "Bearer beta-secret-token",
        },
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
    )
    assert beta.status_code == 403

    crash = await mcp_client.post(
        "/v1/mcp/mock",
        headers=tenant_alpha_headers,
        json={
            "jsonrpc": "2.0",
            "id": "boom",
            "method": "tools/call",
            "params": {"name": "crash", "arguments": {}},
        },
    )
    assert crash.status_code == 502

    recovered = await mcp_client.post(
        "/v1/mcp/mock",
        headers=tenant_alpha_headers,
        json={"jsonrpc": "2.0", "id": "after", "method": "ping"},
    )
    assert recovered.status_code == 200
    assert recovered.json()["result"]["status"] == "ok"


@pytest.mark.asyncio
async def test_mcp_routes_404_when_not_configured(
    gateway_client: httpx.AsyncClient, tenant_alpha_headers: dict[str, str]
) -> None:
    app.state.mcp = None
    response = await gateway_client.get("/v1/mcp/servers", headers=tenant_alpha_headers)
    assert response.status_code == 404


def test_example_config_is_valid() -> None:
    config = load_mcp_config(str(PROJECT_ROOT / "mcp_servers.example.json"))
    assert {"mock", "notes"} <= set(config.servers)


@pytest.mark.asyncio
async def test_reaper_does_not_kill_process_with_in_flight_request() -> None:
    server = MCPServerConfig(
        name="slow",
        command=[sys.executable, "-u", str(MOCK_MCP)],
        request_timeout_seconds=5.0,
        idle_ttl_seconds=0.1,
        health_check=MCPHealthCheckConfig(enabled=False),
    )
    manager = MCPProcessPoolManager(
        MCPSupervisorConfig(servers={"slow": server}, reap_interval_seconds=0.05)
    )
    await manager.start()
    try:
        response = await manager.invoke(
            "tenant-slow",
            "slow",
            {
                "jsonrpc": "2.0",
                "id": "d1",
                "method": "tools/call",
                "params": {"name": "delay", "arguments": {"seconds": 0.5}},
            },
        )
        assert response["result"]["content"][0]["text"] == "delayed"
        await asyncio.sleep(0.4)
        assert await manager.snapshot() == []
    finally:
        await manager.close()
