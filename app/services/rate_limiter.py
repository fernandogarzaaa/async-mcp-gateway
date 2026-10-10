"""Redis-backed async token-bucket limiter for tenant RPM and TPM quotas."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Final, cast

import redis.asyncio as redis
from redis.exceptions import NoScriptError, RedisError

from app.core.config import TenantConfig

logger = logging.getLogger(__name__)

TOKEN_BUCKET_LUA: Final[
    str
] = """
local now = tonumber(ARGV[1])
local rpm_capacity = tonumber(ARGV[2])
local rpm_refill = tonumber(ARGV[3])
local rpm_cost = tonumber(ARGV[4])
local tpm_capacity = tonumber(ARGV[5])
local tpm_refill = tonumber(ARGV[6])
local tpm_cost = tonumber(ARGV[7])
local ttl_ms = tonumber(ARGV[8])

local function load_bucket(key, capacity, refill)
  local values = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(values[1])
  local ts = tonumber(values[2])
  if tokens == nil then
    tokens = capacity
  end
  if ts == nil then
    ts = now
  end
  local elapsed = math.max(0, now - ts)
  tokens = math.min(capacity, tokens + (elapsed * refill))
  return tokens
end

local rpm_tokens = load_bucket(KEYS[1], rpm_capacity, rpm_refill)
local tpm_tokens = load_bucket(KEYS[2], tpm_capacity, tpm_refill)
local allowed = 0

if rpm_tokens >= rpm_cost and tpm_tokens >= tpm_cost then
  rpm_tokens = rpm_tokens - rpm_cost
  tpm_tokens = tpm_tokens - tpm_cost
  allowed = 1
end

redis.call('HSET', KEYS[1], 'tokens', rpm_tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[1], ttl_ms)
redis.call('HSET', KEYS[2], 'tokens', tpm_tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[2], ttl_ms)

local rpm_missing = math.max(0, rpm_cost - rpm_tokens)
local tpm_missing = math.max(0, tpm_cost - tpm_tokens)
local rpm_reset_ms = 0
local tpm_reset_ms = 0
if allowed == 0 and rpm_refill > 0 then
  rpm_reset_ms = math.ceil(rpm_missing / rpm_refill)
end
if allowed == 0 and tpm_refill > 0 then
  tpm_reset_ms = math.ceil(tpm_missing / tpm_refill)
end

return {allowed, rpm_tokens, tpm_tokens, math.max(rpm_reset_ms, tpm_reset_ms)}
"""


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Outcome from a tenant rate-limit check."""

    allowed: bool
    rpm_remaining: int
    tpm_remaining: int
    retry_after_seconds: int

    def headers(self, tenant: TenantConfig) -> dict[str, str]:
        """Return standard rate-limit headers for HTTP responses."""

        headers = {
            "X-RateLimit-Limit-Requests": str(tenant.rpm_limit),
            "X-RateLimit-Remaining-Requests": str(max(0, self.rpm_remaining)),
            "X-RateLimit-Limit-Tokens": str(tenant.tpm_limit),
            "X-RateLimit-Remaining-Tokens": str(max(0, self.tpm_remaining)),
        }
        if not self.allowed:
            headers["Retry-After"] = str(max(1, self.retry_after_seconds))
        return headers


class RedisTokenBucketRateLimiter:
    """Atomic Redis Lua token bucket limiter for RPM and TPM quotas."""

    def __init__(self, client: redis.Redis, key_prefix: str = "ai-gateway") -> None:
        self._client = client
        self._key_prefix = key_prefix
        self._script_sha: str | None = None

    async def initialize(self) -> None:
        """Load the Lua script into Redis so hot-path checks can use EVALSHA."""

        try:
            self._script_sha = await self._client.script_load(TOKEN_BUCKET_LUA)
            logger.info(
                "loaded rate limiter lua script", extra={"sha": self._script_sha}
            )
        except RedisError:
            logger.exception("failed to load rate limiter lua script")
            raise

    async def close(self) -> None:
        """Close the Redis connection pool."""

        try:
            await self._client.aclose()
        except RedisError:
            logger.exception("error while closing redis client")

    async def ping(self) -> bool:
        """Return whether Redis responds to a health-check ping."""

        return bool(await self._client.ping())

    async def check(
        self,
        tenant_id: str,
        tenant: TenantConfig,
        estimated_tokens: int,
    ) -> RateLimitDecision:
        """Consume one request token and estimated prompt tokens atomically."""

        now_ms = int(time.time() * 1000)
        rpm_key = f"{self._key_prefix}:{{{tenant_id}}}:rpm"
        tpm_key = f"{self._key_prefix}:{{{tenant_id}}}:tpm"
        token_cost = max(1, estimated_tokens)

        try:
            result = await self._run_check(rpm_key, tpm_key, now_ms, tenant, token_cost)
        except RedisError:
            logger.exception(
                "redis rate limit check failed", extra={"tenant_id": tenant_id}
            )
            raise

        allowed, rpm_remaining, tpm_remaining, reset_ms = self._parse_result(result)
        return RateLimitDecision(
            allowed=allowed,
            rpm_remaining=math.floor(rpm_remaining),
            tpm_remaining=math.floor(tpm_remaining),
            retry_after_seconds=math.ceil(reset_ms / 1000),
        )

    async def _run_check(
        self,
        rpm_key: str,
        tpm_key: str,
        now_ms: int,
        tenant: TenantConfig,
        token_cost: int,
    ) -> list[object]:
        ttl_ms = 120_000
        args = [
            str(now_ms),
            str(tenant.rpm_limit),
            str(tenant.rpm_limit / 60_000),
            "1",
            str(tenant.tpm_limit),
            str(tenant.tpm_limit / 60_000),
            str(token_cost),
            str(ttl_ms),
        ]
        if self._script_sha is None:
            await self.initialize()
        assert self._script_sha is not None

        try:
            result = await cast(Any, self._client).evalsha(
                self._script_sha, 2, rpm_key, tpm_key, *args
            )
            return cast(list[object], result)
        except NoScriptError:
            await self.initialize()
            assert self._script_sha is not None
            result = await cast(Any, self._client).evalsha(
                self._script_sha, 2, rpm_key, tpm_key, *args
            )
            return cast(list[object], result)

    @staticmethod
    def _parse_result(raw: list[object]) -> tuple[bool, float, float, int]:
        allowed = bool(RedisTokenBucketRateLimiter._as_int(raw[0]))
        rpm_remaining = RedisTokenBucketRateLimiter._as_float(raw[1])
        tpm_remaining = RedisTokenBucketRateLimiter._as_float(raw[2])
        reset_ms = RedisTokenBucketRateLimiter._as_int(raw[3])
        return allowed, rpm_remaining, tpm_remaining, reset_ms

    @staticmethod
    def _as_float(value: object) -> float:
        if isinstance(value, (bytes, str, int, float)):
            return float(value)
        raise TypeError(f"cannot convert Redis value {value!r} to float")

    @staticmethod
    def _as_int(value: object) -> int:
        if isinstance(value, (bytes, str, int, float)):
            return int(value)
        raise TypeError(f"cannot convert Redis value {value!r} to int")


def estimate_prompt_tokens(payload: dict[str, object]) -> int:
    """Cheap deterministic token estimate used before provider-specific tokenizers."""

    def value_chars(value: object) -> int:
        if isinstance(value, str):
            return len(value)
        if isinstance(value, list):
            return sum(value_chars(item) for item in value)
        if isinstance(value, dict):
            return sum(value_chars(item) for item in value.values())
        return len(str(value))

    messages = payload.get("messages")
    prompt = payload.get("prompt")
    source = (
        messages if messages is not None else prompt if prompt is not None else payload
    )
    return max(1, math.ceil(value_chars(source) / 4))
