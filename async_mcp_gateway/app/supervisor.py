"""Tenant-isolated stdio MCP process supervision."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MCPSupervisor:
    """Manage persistent stdio-based MCP server processes per tenant."""

    def __init__(self, request_timeout_seconds: float = 30.0) -> None:
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self.commands: dict[str, list[str]] = {}
        self.request_timeout_seconds = request_timeout_seconds
        self._lock = asyncio.Lock()

    async def execute_tool(
        self,
        tenant_id: str,
        command: list[str],
        json_rpc_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one JSON-RPC payload against a tenant-specific MCP process."""

        process = await self._get_or_start_process(tenant_id, command)
        try:
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("mcp process stdio pipes are unavailable")
            payload = json.dumps(json_rpc_payload, separators=(",", ":")) + "\n"
            process.stdin.write(payload.encode("utf-8"))
            flush = getattr(process.stdin, "flush", None)
            if callable(flush):
                flush()
            await process.stdin.drain()
            raw_line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=self.request_timeout_seconds,
            )
            if not raw_line:
                raise RuntimeError("mcp process closed stdout without a response")
            response = json.loads(raw_line.decode("utf-8"))
            if not isinstance(response, dict):
                raise RuntimeError("mcp process returned a non-object JSON frame")
            return response
        except Exception as exc:
            logger.exception(
                "mcp tool execution failed; recycling process",
                extra={"tenant_id": tenant_id, "error": str(exc)},
            )
            await self._recycle_process(tenant_id)
            return {
                "jsonrpc": "2.0",
                "id": json_rpc_payload.get("id"),
                "error": {
                    "code": -32000,
                    "message": "mcp process execution failed",
                    "data": str(exc),
                },
            }

    async def shutdown(self) -> None:
        """Terminate every supervised process."""

        async with self._lock:
            tenant_ids = list(self.processes)
        for tenant_id in tenant_ids:
            await self._recycle_process(tenant_id)

    async def _get_or_start_process(
        self, tenant_id: str, command: list[str]
    ) -> asyncio.subprocess.Process:
        async with self._lock:
            process = self.processes.get(tenant_id)
            previous_command = self.commands.get(tenant_id)
            if (
                process is not None
                and process.returncode is None
                and previous_command == command
            ):
                return process
            if process is not None:
                await self._terminate_process(process)
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self.processes[tenant_id] = process
            self.commands[tenant_id] = list(command)
            logger.info(
                "started mcp process",
                extra={"tenant_id": tenant_id, "pid": process.pid},
            )
            return process

    async def _recycle_process(self, tenant_id: str) -> None:
        async with self._lock:
            process = self.processes.pop(tenant_id, None)
            self.commands.pop(tenant_id, None)
        if process is not None:
            await self._terminate_process(process)

    @staticmethod
    async def _terminate_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError:
            process.kill()
            await process.wait()
