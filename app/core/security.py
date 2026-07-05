"""Multi-tenant bearer authentication middleware."""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.config import Settings, TenantConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Authenticated tenant metadata attached to request.state."""

    tenant_id: str
    config: TenantConfig


class TenantAuthMiddleware(BaseHTTPMiddleware):
    """Validate X-Tenant-ID and Authorization bearer tokens for every request."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in {"/healthz", "/readyz"}:
            return await call_next(request)

        tenant_id = request.headers.get("X-Tenant-ID", "").strip()
        authorization = request.headers.get("Authorization", "").strip()

        if not tenant_id:
            logger.warning("missing tenant id", extra={"path": request.url.path})
            return self._reject(401, "missing_tenant_id", "X-Tenant-ID is required")

        tenant = self._settings.tenants.get(tenant_id)
        if tenant is None:
            logger.warning("unknown tenant", extra={"tenant_id": tenant_id})
            return self._reject(403, "unknown_tenant", "tenant is not configured")

        token = self._extract_bearer_token(authorization)
        expected = tenant.bearer_token.get_secret_value()
        if token is None or not hmac.compare_digest(token, expected):
            logger.warning("invalid tenant token", extra={"tenant_id": tenant_id})
            return self._reject(401, "invalid_token", "invalid bearer token")

        request.state.tenant = TenantContext(tenant_id=tenant_id, config=tenant)
        return await call_next(request)

    @staticmethod
    def _extract_bearer_token(authorization: str) -> str | None:
        scheme, separator, token = authorization.partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not token:
            return None
        return token.strip()

    @staticmethod
    def _reject(status_code: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"error": {"code": code, "message": message}},
            headers={"WWW-Authenticate": "Bearer"},
        )
