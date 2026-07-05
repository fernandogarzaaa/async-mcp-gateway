"""FastAPI ingress for the async MCP gateway."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.config import settings
from app.limiter import RedisRateLimiter
from app.router import ResilientStreamRouter
from app.supervisor import MCPSupervisor

logger = logging.getLogger(__name__)

TENANT_REGISTRY: dict[str, dict[str, Any]] = {
    "tenant-alpha": {
        "token": "alpha-secret-token",
        "max_tpm": 120_000,
        "max_rpm": 120,
        "scopes": {"chat", "mcp"},
    },
    "tenant-beta": {
        "token": "beta-secret-token",
        "max_tpm": 60_000,
        "max_rpm": 60,
        "scopes": {"chat"},
    },
}

limiter = RedisRateLimiter()
router = ResilientStreamRouter()
supervisor = MCPSupervisor(request_timeout_seconds=settings.GATEWAY_TIMEOUT)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and tear down gateway resources."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    await limiter.connect()
    try:
        yield
    finally:
        await router.close()
        await supervisor.shutdown()
        await limiter.close()


app = FastAPI(
    title="async-mcp-gateway",
    version="0.1.0",
    lifespan=lifespan,
)


async def authenticate_tenant(
    x_tenant_id: str = Header(alias="X-Tenant-ID"),
    authorization: str = Header(alias="Authorization"),
) -> dict[str, Any]:
    """Validate tenant headers against the hardcoded test registry."""

    tenant_config = TENANT_REGISTRY.get(x_tenant_id)
    if tenant_config is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="unknown tenant",
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != tenant_config["token"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
        )
    return {"tenant_id": x_tenant_id, **tenant_config}


@app.post("/v1/gateway/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    tenant_config: dict[str, Any] = Depends(authenticate_tenant),
) -> Response:
    """Stream chat completions through the resilient provider router."""

    payload = await _json_body(request)
    tokens = _estimate_tokens(payload)
    allowed = await limiter.is_allowed(
        tenant_id=str(tenant_config["tenant_id"]),
        tokens=tokens,
        max_tpm=int(tenant_config["max_tpm"]),
        max_rpm=int(tenant_config["max_rpm"]),
    )
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": {"code": "rate_limit_exceeded"}},
        )
    return StreamingResponse(
        router.stream_response(payload, tenant_config),
        media_type="text/event-stream",
    )


@app.post("/v1/gateway/mcp/execute")
async def execute_mcp(
    request: Request,
    tenant_config: dict[str, Any] = Depends(authenticate_tenant),
) -> dict[str, Any]:
    """Execute a JSON-RPC MCP payload through a supervised process."""

    if "mcp" not in tenant_config["scopes"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="scope denied",
        )
    body = await _json_body(request)
    command = body.get("command")
    payload = body.get("payload")
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid command",
        )
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid payload",
        )
    return await supervisor.execute_tool(
        tenant_id=str(tenant_config["tenant_id"]),
        command=command,
        json_rpc_payload=payload,
    )


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return body


def _estimate_tokens(payload: dict[str, Any]) -> int:
    return max(1, len(str(payload)) // 4)
