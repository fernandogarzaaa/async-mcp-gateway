"""Unit tests for the asynchronous MCP subprocess supervisor."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from app.models.schemas import (
    MCPHealthCheckConfig,
    MCPServerConfig,
    MCPSupervisorConfig,
    TenantMCPPolicy,
)
from app.services.mcp_supervisor import (
    MCPAccessDenied,
    MCPProcessPoolManager,
    MCPRequestTimeout,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCK_MCP = PROJECT_ROOT / "mock_mcp_server.py"


def make_server_config(
    name: str = "mock-mcp",
    *,
    request_timeout_seconds: float = 1.0,
    idle_ttl_seconds: float = 0.2,
) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        command=[sys.executable, "-u", str(MOCK_MCP)],
        request_timeout_seconds=request_timeout_seconds,
        idle_ttl_seconds=idle_ttl_seconds,
        health_check=MCPHealthCheckConfig(enabled=False),
    )


@pytest.mark.asyncio
async def test_supervisor_round_trips_tool_call_with_tenant_isolation() -> None:
    server = make_server_config()
    manager = MCPProcessPoolManager(
        MCPSupervisorConfig(servers={server.name: server}, reap_interval_seconds=0.05)
    )
    await manager.start()
    try:
        response = await manager.invoke(
            "tenant-a",
            server.name,
            {
                "jsonrpc": "2.0",
                "id": "call-1",
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"value": 42}},
            },
        )
        snapshot = await manager.snapshot()
    finally:
        await manager.close()

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "call-1"
    assert response["result"]["content"][0]["type"] == "text"
    assert '"tenant_id": "tenant-a"' in response["result"]["content"][0]["text"]
    assert snapshot[0]["requests_total"] == 1
    assert snapshot[0]["failures_total"] == 0


@pytest.mark.asyncio
async def test_hung_mcp_subprocess_times_out_and_is_reaped_by_ttl() -> None:
    server = make_server_config(request_timeout_seconds=0.05, idle_ttl_seconds=0.1)
    manager = MCPProcessPoolManager(
        MCPSupervisorConfig(
            servers={server.name: server},
            reap_interval_seconds=0.05,
            shutdown_grace_seconds=0.05,
        )
    )
    await manager.start()
    try:
        with pytest.raises(MCPRequestTimeout):
            await manager.invoke(
                "tenant-hang",
                server.name,
                {
                    "jsonrpc": "2.0",
                    "id": "hang-1",
                    "method": "tools/call",
                    "params": {"name": "hang", "arguments": {}},
                },
            )

        await asyncio.sleep(0.3)
        assert await manager.snapshot() == []
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_tenant_process_capacity_prevents_cross_server_memory_growth() -> None:
    first = make_server_config("first")
    second = make_server_config("second")
    policy = TenantMCPPolicy(
        tenant_id="tenant-cap", allowed_servers={"first", "second"}, max_processes=1
    )
    manager = MCPProcessPoolManager(
        MCPSupervisorConfig(
            servers={"first": first, "second": second},
            tenant_policies={"tenant-cap": policy},
            reap_interval_seconds=0.05,
        )
    )
    await manager.start()
    try:
        response = await manager.invoke(
            "tenant-cap",
            "first",
            {"jsonrpc": "2.0", "id": "one", "method": "ping", "params": {}},
        )
        assert response["result"]["status"] == "ok"
        with pytest.raises(MCPAccessDenied, match="exceeded MCP process limit"):
            await manager.invoke(
                "tenant-cap",
                "second",
                {"jsonrpc": "2.0", "id": "two", "method": "ping", "params": {}},
            )
    finally:
        await manager.close()
