"""FastAPI application entrypoint for the asynchronous AI gateway."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError
from redis.exceptions import RedisError

from app.core.config import Settings, get_settings
from app.core.security import TenantAuthMiddleware, TenantContext
from app.models.schemas import MCPSupervisorConfig
from app.services.mcp_supervisor import (
    MCPAccessDenied,
    MCPProcessPoolManager,
    MCPRequestTimeout,
    MCPSupervisorError,
)
from app.services.rate_limiter import (
    RedisTokenBucketRateLimiter,
    estimate_prompt_tokens,
)
from app.services.router import LLMRouter, ProviderRoutingError

logger = logging.getLogger(__name__)


def configure_logging(settings: Settings) -> None:
    """Configure standard-library structured logging."""

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def load_mcp_config(path: str) -> MCPSupervisorConfig:
    """Load and validate an MCP supervisor config from a JSON file."""

    with open(path, encoding="utf-8") as handle:
        return MCPSupervisorConfig.model_validate(json.load(handle))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)

    redis_from_url = cast(Callable[..., redis.Redis], redis.from_url)
    redis_client = redis_from_url(settings.redis_uri, decode_responses=True)
    rate_limiter = RedisTokenBucketRateLimiter(redis_client)
    router = LLMRouter(settings)

    mcp_manager: MCPProcessPoolManager | None = None
    if settings.mcp_config_file:
        mcp_manager = MCPProcessPoolManager(load_mcp_config(settings.mcp_config_file))
        await mcp_manager.start()

    await rate_limiter.initialize()
    app.state.settings = settings
    app.state.rate_limiter = rate_limiter
    app.state.router = router
    app.state.mcp = mcp_manager

    logger.info(
        "ai gateway started",
        extra={"app_name": settings.app_name, "mcp_enabled": mcp_manager is not None},
    )
    try:
        yield
    finally:
        if mcp_manager is not None:
            await mcp_manager.close()
        await router.close()
        await rate_limiter.close()
        logger.info("ai gateway stopped")


settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(TenantAuthMiddleware, settings=settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> dict[str, str]:
    limiter: RedisTokenBucketRateLimiter = request.app.state.rate_limiter
    try:
        await limiter.ping()
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="redis unavailable") from exc
    return {"status": "ready"}


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Response:
    payload = await _read_json_payload(request)
    tenant = _tenant_context(request)
    estimated_tokens = estimate_prompt_tokens(payload)

    limiter: RedisTokenBucketRateLimiter = request.app.state.rate_limiter
    try:
        decision = await limiter.check(
            tenant.tenant_id, tenant.config, estimated_tokens
        )
    except RedisError as exc:
        logger.exception(
            "rate limiter unavailable", extra={"tenant_id": tenant.tenant_id}
        )
        raise HTTPException(status_code=503, detail="rate limiter unavailable") from exc

    if not decision.allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "tenant quota exhausted",
                }
            },
            headers=decision.headers(tenant.config),
        )

    router: LLMRouter = request.app.state.router
    if bool(payload.get("stream", False)):
        stream = router.stream_chat_completion(payload, tenant.config, request.headers)
        return StreamingResponse(
            stream,
            status_code=200,
            media_type="text/event-stream",
            headers=decision.headers(tenant.config)
            | {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response_payload = await router.create_chat_completion(
            payload, tenant.config, request.headers
        )
    except ProviderRoutingError as exc:
        logger.error(
            "provider routing failed",
            extra={"tenant_id": tenant.tenant_id, "error": str(exc)},
        )
        return JSONResponse(
            status_code=502,
            content={"error": {"code": "upstream_unavailable", "message": str(exc)}},
            headers=decision.headers(tenant.config),
        )

    return JSONResponse(
        content=response_payload, headers=decision.headers(tenant.config)
    )


@app.get("/v1/mcp/servers")
async def mcp_servers(request: Request) -> dict[str, list[str]]:
    tenant = _tenant_context(request)
    manager = _mcp_manager(request)
    return {"servers": manager.allowed_servers(tenant.tenant_id)}


@app.post("/v1/mcp/{server_name}", response_model=None)
async def mcp_rpc(server_name: str, request: Request) -> Response:
    """Forward one JSON-RPC message to the tenant's stdio MCP server process."""

    payload = await _read_json_payload(request)
    tenant = _tenant_context(request)
    manager = _mcp_manager(request)
    if not manager.has_server(server_name):
        raise HTTPException(
            status_code=404, detail=f"unknown MCP server {server_name!r}"
        )

    limiter: RedisTokenBucketRateLimiter = request.app.state.rate_limiter
    try:
        decision = await limiter.check(tenant.tenant_id, tenant.config, 1)
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="rate limiter unavailable") from exc
    headers = decision.headers(tenant.config)
    if not decision.allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "tenant quota exhausted",
                }
            },
            headers=headers,
        )

    try:
        if payload.get("id") is None:
            await manager.notify(tenant.tenant_id, server_name, payload)
            return Response(status_code=202, headers=headers)
        result = await manager.invoke(tenant.tenant_id, server_name, payload)
    except ValidationError as exc:
        reasons = "; ".join(str(error["msg"]) for error in exc.errors())
        raise HTTPException(
            status_code=400, detail=f"invalid JSON-RPC message: {reasons}"
        ) from exc
    except MCPAccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except MCPRequestTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except MCPSupervisorError as exc:
        logger.warning(
            "mcp request failed",
            extra={
                "tenant_id": tenant.tenant_id,
                "server": server_name,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(content=result, headers=headers)


def _mcp_manager(request: Request) -> MCPProcessPoolManager:
    manager = getattr(request.app.state, "mcp", None)
    if not isinstance(manager, MCPProcessPoolManager):
        raise HTTPException(
            status_code=404,
            detail="MCP is not configured; set MCP_CONFIG_FILE",
        )
    return manager


async def _read_json_payload(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return payload


def _tenant_context(request: Request) -> TenantContext:
    tenant = getattr(request.state, "tenant", None)
    if not isinstance(tenant, TenantContext):
        raise HTTPException(status_code=401, detail="tenant context missing")
    return tenant
