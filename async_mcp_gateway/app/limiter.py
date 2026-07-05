"""Atomic Redis-backed rate limiting for gateway tenants."""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import redis.asyncio as redis
from redis.exceptions import RedisError

from app.config import settings

logger = logging.getLogger(__name__)


class RedisRateLimiter:
    """Minute-window RPM and TPM limiter backed by Redis transactions."""

    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url or settings.REDIS_URL
        self.client: redis.Redis | None = None
        self.available = False

    async def connect(self) -> None:
        """Initialize the Redis connection and fail open when unavailable."""

        try:
            client = cast(Any, redis.from_url)(
                self.redis_url,
                decode_responses=True,
            )
            await client.ping()
            self.client = client
            self.available = True
            logger.info("redis limiter connected", extra={"redis_url": self.redis_url})
        except RedisError as exc:
            self.client = None
            self.available = False
            logger.warning(
                "redis limiter unavailable; failing open",
                extra={"redis_url": self.redis_url, "error": str(exc)},
            )

    async def close(self) -> None:
        """Close the Redis client if it was opened."""

        if self.client is not None:
            try:
                await self.client.aclose()
            except RedisError as exc:
                logger.warning("redis limiter close failed", extra={"error": str(exc)})

    async def is_allowed(
        self,
        tenant_id: str,
        tokens: int,
        max_tpm: int,
        max_rpm: int,
    ) -> bool:
        """Return whether a request fits within tenant RPM and TPM limits."""

        if self.client is None or not self.available:
            logger.debug("rate limiter bypassed because redis is unavailable")
            return True

        safe_tokens = max(0, tokens)
        window = int(time.time() // 60)
        rpm_key = f"rate:{tenant_id}:rpm:{window}"
        tpm_key = f"rate:{tenant_id}:tpm:{window}"

        try:
            pipe = self.client.pipeline(transaction=True)
            pipe.incr(rpm_key, 1)
            pipe.incrby(tpm_key, safe_tokens)
            pipe.expire(rpm_key, 120)
            pipe.expire(tpm_key, 120)
            result = await pipe.execute()
            request_count = self._to_int(result[0])
            token_count = self._to_int(result[1])
            allowed = request_count <= max_rpm and token_count <= max_tpm
            logger.info(
                "rate limit evaluated",
                extra={
                    "tenant_id": tenant_id,
                    "window": window,
                    "request_count": request_count,
                    "token_count": token_count,
                    "max_rpm": max_rpm,
                    "max_tpm": max_tpm,
                    "allowed": allowed,
                },
            )
            return allowed
        except RedisError as exc:
            self.available = False
            logger.warning(
                "redis limiter failed during transaction; failing open",
                extra={"tenant_id": tenant_id, "error": str(exc)},
            )
            return True

    @staticmethod
    def _to_int(value: object) -> int:
        if isinstance(value, (bytes, str, int)):
            return int(value)
        return cast(int, value)
