"""Asynchronous MCP subprocess supervisor for tenant-isolated stdio tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import (
    JsonRpcRequest,
    JsonRpcResponse,
    MCPServerConfig,
    MCPSupervisorConfig,
)

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]


class MCPSupervisorError(RuntimeError):
    """Base exception for MCP supervisor failures."""


class MCPProcessUnavailable(MCPSupervisorError):
    """Raised when a supervised MCP process is not available."""


class MCPRequestTimeout(MCPSupervisorError):
    """Raised when a JSON-RPC request does not complete before its deadline."""


class MCPAccessDenied(MCPSupervisorError):
    """Raised when a tenant is not authorized for an MCP server."""


@dataclass(frozen=True, slots=True)
class MCPPoolKey:
    """Pool key isolating MCP process instances by tenant and server name."""

    tenant_id: str
    server_name: str


@dataclass(slots=True)
class MCPProcessStats:
    """Operational counters for one supervised MCP process."""

    started_at: float
    last_used_at: float
    requests_total: int = 0
    failures_total: int = 0
    restarts_total: int = 0
    stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=500))


class ManagedMCPProcess:
    """Persistent stdio JSON-RPC bridge for one tenant/server subprocess."""

    def __init__(
        self,
        tenant_id: str,
        config: MCPServerConfig,
        *,
        shutdown_grace_seconds: float,
    ) -> None:
        self.tenant_id = tenant_id
        self.config = config
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[str, asyncio.Future[JsonDict]] = {}
        self._pending_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._wait_task: asyncio.Task[None] | None = None
        now = time.monotonic()
        self.stats = MCPProcessStats(
            started_at=now,
            last_used_at=now,
            stderr_tail=deque(maxlen=config.max_stderr_lines),
        )

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None else None

    @property
    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.returncode is None

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.stats.last_used_at

    async def ensure_started(self) -> None:
        """Start or recycle the underlying subprocess."""

        async with self._lifecycle_lock:
            if self.is_running:
                return
            await self._cleanup_process()
            await self._start_process()

    async def request(
        self, payload: Mapping[str, Any], timeout_seconds: float | None = None
    ) -> JsonDict:
        """Send one line-delimited JSON-RPC request and await its response."""

        await self.ensure_started()
        request = JsonRpcRequest.model_validate(dict(payload))
        request_id = str(request.id)
        timeout = timeout_seconds or self.config.request_timeout_seconds
        future: asyncio.Future[JsonDict] = asyncio.get_running_loop().create_future()

        async with self._pending_lock:
            if request_id in self._pending:
                raise MCPSupervisorError(
                    f"duplicate in-flight MCP request id {request_id!r}"
                )
            self._pending[request_id] = future

        try:
            await self._write_json_line(request.model_dump(exclude_none=True))
            self.stats.requests_total += 1
            self.stats.last_used_at = time.monotonic()
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as exc:
            self.stats.failures_total += 1
            async with self._pending_lock:
                self._pending.pop(request_id, None)
            raise MCPRequestTimeout(
                f"MCP request {request_id!r} timed out after {timeout:.2f}s"
            ) from exc
        except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
            self.stats.failures_total += 1
            async with self._pending_lock:
                self._pending.pop(request_id, None)
            await self.restart()
            raise MCPProcessUnavailable(
                f"MCP process {self.config.name!r} write failed"
            ) from exc

    async def health_check(self) -> bool:
        """Return whether the process is alive and optionally answers a health RPC."""

        if not self.is_running:
            return False
        if not self.config.health_check.enabled:
            return True

        health_payload = self.config.health_check.request.model_copy(
            update={"id": f"health-{uuid.uuid4().hex}"}
        )
        try:
            await self.request(
                health_payload.model_dump(exclude_none=True),
                timeout_seconds=self.config.health_check.timeout_seconds,
            )
            return True
        except MCPSupervisorError:
            logger.warning(
                "mcp health check failed",
                extra={
                    "tenant_id": self.tenant_id,
                    "server": self.config.name,
                    "pid": self.pid,
                },
            )
            return False

    async def restart(self) -> None:
        """Recycle the subprocess and fail all currently pending requests."""

        async with self._lifecycle_lock:
            self.stats.restarts_total += 1
            await self._cleanup_process()
            await self._start_process()

    async def close(self) -> None:
        """Terminate the subprocess and release all supervisor resources."""

        async with self._lifecycle_lock:
            await self._cleanup_process()

    async def _start_process(self) -> None:
        env = os.environ.copy()
        env.update(self.config.tenant_environment(self.tenant_id))
        for key in tuple(env):
            if key.startswith("COV_CORE_") or key in {
                "COVERAGE_PROCESS_START",
                "PYTEST_CURRENT_TEST",
            }:
                env.pop(key, None)
        started = time.monotonic()
        try:
            self._process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *self.config.command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.config.cwd,
                    env=env,
                    limit=1024 * 1024,
                ),
                timeout=self.config.startup_timeout_seconds,
            )
        except (OSError, TimeoutError) as exc:
            raise MCPProcessUnavailable(
                f"failed to start MCP server {self.config.name!r}"
            ) from exc

        self.stats.started_at = started
        self.stats.last_used_at = started
        self._reader_task = asyncio.create_task(
            self._stdout_loop(), name=f"mcp-stdout-{self.config.name}-{self.tenant_id}"
        )
        self._stderr_task = asyncio.create_task(
            self._stderr_loop(), name=f"mcp-stderr-{self.config.name}-{self.tenant_id}"
        )
        self._wait_task = asyncio.create_task(
            self._wait_loop(), name=f"mcp-wait-{self.config.name}-{self.tenant_id}"
        )
        logger.info(
            "started mcp process",
            extra={
                "tenant_id": self.tenant_id,
                "server": self.config.name,
                "pid": self.pid,
            },
        )

    async def _cleanup_process(self) -> None:
        await self._fail_pending(
            MCPProcessUnavailable(f"MCP server {self.config.name!r} stopped")
        )

        for task in (self._reader_task, self._stderr_task, self._wait_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(
                task
                for task in (self._reader_task, self._stderr_task, self._wait_task)
                if task is not None
            ),
            return_exceptions=True,
        )
        self._reader_task = None
        self._stderr_task = None
        self._wait_task = None

        process = self._process
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=self._shutdown_grace_seconds
                )
            except TimeoutError:
                process.kill()
                await process.wait()
        self._process = None

    async def _write_json_line(self, payload: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise MCPProcessUnavailable(
                f"MCP server {self.config.name!r} is not running"
            )
        line = (
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        async with self._write_lock:
            process.stdin.write(line)
            await process.stdin.drain()

    async def _stdout_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                await self._handle_stdout_line(line)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "mcp stdout loop failed",
                extra={"tenant_id": self.tenant_id, "server": self.config.name},
            )
            await self._fail_pending(exc)

    async def _stderr_loop(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                self.stats.stderr_tail.append(decoded)
                logger.debug(
                    "mcp stderr",
                    extra={
                        "tenant_id": self.tenant_id,
                        "server": self.config.name,
                        "stderr": decoded,
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "mcp stderr loop failed",
                extra={"tenant_id": self.tenant_id, "server": self.config.name},
            )

    async def _wait_loop(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            return_code = await process.wait()
            logger.warning(
                "mcp process exited",
                extra={
                    "tenant_id": self.tenant_id,
                    "server": self.config.name,
                    "pid": process.pid,
                    "return_code": return_code,
                },
            )
            await self._fail_pending(
                MCPProcessUnavailable(
                    f"MCP server {self.config.name!r} exited with {return_code}"
                )
            )
        except asyncio.CancelledError:
            raise

    async def _handle_stdout_line(self, line: bytes) -> None:
        try:
            payload = json.loads(line.decode("utf-8"))
            response = JsonRpcResponse.model_validate(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "discarding invalid mcp stdout line",
                extra={
                    "tenant_id": self.tenant_id,
                    "server": self.config.name,
                    "error": str(exc),
                },
            )
            return

        if response.id is None:
            logger.debug(
                "received mcp notification",
                extra={"tenant_id": self.tenant_id, "server": self.config.name},
            )
            return

        request_id = str(response.id)
        async with self._pending_lock:
            future = self._pending.pop(request_id, None)
        if future is None:
            logger.warning(
                "received mcp response for unknown request",
                extra={
                    "tenant_id": self.tenant_id,
                    "server": self.config.name,
                    "request_id": request_id,
                },
            )
            return
        if not future.done():
            future.set_result(response.model_dump(exclude_none=True))

    async def _fail_pending(self, exc: BaseException) -> None:
        async with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(exc)


class MCPProcessPoolManager:
    """Tenant-isolated pool manager for supervised MCP subprocesses."""

    def __init__(self, config: MCPSupervisorConfig) -> None:
        self._config = config
        self._processes: dict[MCPPoolKey, ManagedMCPProcess] = {}
        self._lock = asyncio.Lock()
        self._reaper_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        """Start the background idle/crash reaper."""

        if self._reaper_task is None or self._reaper_task.done():
            self._closed = False
            self._reaper_task = asyncio.create_task(
                self._reaper_loop(), name="mcp-process-reaper"
            )

    async def close(self) -> None:
        """Terminate all managed processes and stop background tasks."""

        self._closed = True
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            await asyncio.gather(self._reaper_task, return_exceptions=True)
            self._reaper_task = None

        async with self._lock:
            processes = list(self._processes.values())
            self._processes.clear()
        await asyncio.gather(
            *(process.close() for process in processes), return_exceptions=True
        )

    async def invoke(
        self, tenant_id: str, server_name: str, payload: Mapping[str, Any]
    ) -> JsonDict:
        """Route one JSON-RPC payload to a tenant-isolated MCP process."""

        process = await self._get_or_create_process(tenant_id, server_name)
        return await process.request(payload)

    async def health_check(self, tenant_id: str, server_name: str) -> bool:
        """Run a health check for one tenant/server process if it exists."""

        key = MCPPoolKey(tenant_id=tenant_id, server_name=server_name)
        async with self._lock:
            process = self._processes.get(key)
        if process is None:
            return False
        healthy = await process.health_check()
        if not healthy:
            await self._recycle_process(key, process)
        return healthy

    async def snapshot(self) -> list[dict[str, Any]]:
        """Return operational state for every managed MCP process."""

        async with self._lock:
            items = list(self._processes.items())
        return [
            {
                "tenant_id": key.tenant_id,
                "server_name": key.server_name,
                "pid": process.pid,
                "running": process.is_running,
                "idle_seconds": round(process.idle_seconds, 3),
                "requests_total": process.stats.requests_total,
                "failures_total": process.stats.failures_total,
                "restarts_total": process.stats.restarts_total,
            }
            for key, process in items
        ]

    async def _get_or_create_process(
        self, tenant_id: str, server_name: str
    ) -> ManagedMCPProcess:
        self._authorize(tenant_id, server_name)
        key = MCPPoolKey(tenant_id=tenant_id, server_name=server_name)
        async with self._lock:
            process = self._processes.get(key)
            if process is None:
                self._enforce_tenant_capacity_locked(tenant_id)
                server_config = self._config.servers[server_name]
                process = ManagedMCPProcess(
                    tenant_id,
                    server_config,
                    shutdown_grace_seconds=self._config.shutdown_grace_seconds,
                )
                self._processes[key] = process
        await process.ensure_started()
        return process

    def _authorize(self, tenant_id: str, server_name: str) -> None:
        if server_name not in self._config.servers:
            raise MCPAccessDenied(f"unknown MCP server {server_name!r}")
        policy = self._config.tenant_policies.get(tenant_id)
        if policy is not None and server_name not in policy.allowed_servers:
            raise MCPAccessDenied(
                f"tenant {tenant_id!r} is not allowed to use MCP server {server_name!r}"
            )

    def _enforce_tenant_capacity_locked(self, tenant_id: str) -> None:
        policy = self._config.tenant_policies.get(tenant_id)
        if policy is None:
            return
        active = sum(1 for key in self._processes if key.tenant_id == tenant_id)
        if active >= policy.max_processes:
            raise MCPAccessDenied(
                f"tenant {tenant_id!r} exceeded MCP process limit "
                f"{policy.max_processes}"
            )

    async def _reaper_loop(self) -> None:
        while not self._closed:
            try:
                await asyncio.sleep(self._config.reap_interval_seconds)
                await self._reap_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("mcp reaper loop failed")

    async def _reap_once(self) -> None:
        async with self._lock:
            items = list(self._processes.items())

        for key, process in items:
            ttl = (
                process.config.idle_ttl_seconds or self._config.default_idle_ttl_seconds
            )
            if process.idle_seconds >= ttl:
                logger.info(
                    "reaping idle mcp process",
                    extra={
                        "tenant_id": key.tenant_id,
                        "server": key.server_name,
                        "pid": process.pid,
                    },
                )
                await self._remove_process(key, process)
                continue
            if not process.is_running:
                logger.warning(
                    "recycling crashed mcp process",
                    extra={
                        "tenant_id": key.tenant_id,
                        "server": key.server_name,
                        "pid": process.pid,
                    },
                )
                await self._recycle_process(key, process)
                continue
            if process.config.health_check.enabled:
                healthy = await process.health_check()
                if not healthy:
                    await self._recycle_process(key, process)

    async def _remove_process(
        self, key: MCPPoolKey, process: ManagedMCPProcess
    ) -> None:
        async with self._lock:
            current = self._processes.get(key)
            if current is not process:
                return
            self._processes.pop(key, None)
        await process.close()

    async def _recycle_process(
        self, key: MCPPoolKey, process: ManagedMCPProcess
    ) -> None:
        async with self._lock:
            if self._processes.get(key) is not process:
                return
        try:
            await process.restart()
        except MCPSupervisorError:
            logger.exception(
                "failed to recycle mcp process",
                extra={"tenant_id": key.tenant_id, "server": key.server_name},
            )
            await self._remove_process(key, process)
