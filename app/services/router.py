"""Resilient asynchronous provider router with streaming failover."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

import httpx

from app.core.config import ProviderConfig, ProviderName, Settings, TenantConfig

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)<\s*/?\s*(tool|function|system|developer)[^>]*>"),
    re.compile(r"(?i)\bignore\s+(all\s+)?previous\s+(instructions|messages)\b"),
    re.compile(r"(?i)\bBEGIN_(TOOL|SYSTEM|DEVELOPER)_CALL\b"),
    re.compile(r"(?i)\btool_execution\s*[:=]"),
)


def _max_tokens(value: object, default: int = 1024) -> int:
    """Anthropic requires max_tokens; fall back to the default when unusable."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


@dataclass(frozen=True, slots=True)
class ProviderAttempt:
    """Provider request attempt metadata."""

    provider: ProviderName
    status_code: int | None
    elapsed_ms: float
    error: str | None = None


class ProviderRoutingError(RuntimeError):
    """Raised when every provider route fails."""

    def __init__(self, attempts: list[ProviderAttempt]) -> None:
        self.attempts = attempts
        details = [f"{a.provider}:{a.status_code or a.error}" for a in attempts]
        super().__init__(f"all provider attempts failed: {', '.join(details)}")


# Upstream statuses that describe the request itself (malformed body, too large,
# unprocessable). Retrying them on the next provider only multiplies cost and turns
# a client error into a misleading 502, so they are returned to the caller as-is.
NON_RETRYABLE_STATUSES = frozenset({400, 413, 422})


class ProviderClientError(RuntimeError):
    """Raised when a provider rejects the request itself (no failover)."""

    def __init__(self, provider: ProviderName, status_code: int, message: str) -> None:
        self.provider = provider
        self.status_code = status_code
        self.message = message
        super().__init__(f"{provider} rejected the request ({status_code}): {message}")

    def to_payload(self) -> JsonDict:
        return {
            "error": {
                "code": "upstream_rejected",
                "message": self.message,
                "provider": self.provider,
                "upstream_status": self.status_code,
            }
        }


def _upstream_error_message(body: bytes) -> str:
    """Best-effort message from an upstream error body (OpenAI/Anthropic shapes)."""

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body[:500].decode("utf-8", "replace") or "upstream rejected the request"
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return cast(str, err["message"])[:500]
        if isinstance(err, str):
            return err[:500]
        if isinstance(data.get("message"), str):
            return cast(str, data["message"])[:500]
    return json.dumps(data)[:500]


class LLMRouter:
    """Proxy requests to upstream providers and fail over on transient exhaustion."""

    def __init__(self, settings: Settings) -> None:
        timeout = httpx.Timeout(
            settings.request_timeout_seconds,
            connect=settings.stream_connect_timeout_seconds,
        )
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        """Close the shared HTTPX client."""

        await self._client.aclose()

    async def create_chat_completion(
        self,
        payload: JsonDict,
        tenant: TenantConfig,
        inbound_headers: Mapping[str, str],
    ) -> JsonDict:
        """Send a non-streaming chat completion request with provider failover."""

        attempts: list[ProviderAttempt] = []
        for provider in tenant.provider_preference:
            started = time.perf_counter()
            provider_payload = self._payload_for_provider(provider, payload, tenant)
            try:
                response = await self._client.post(
                    self._url_for(provider, stream=False),
                    json=provider_payload,
                    headers=self._headers_for(provider, inbound_headers, stream=False),
                )
            except httpx.HTTPError as exc:
                attempts.append(self._attempt(provider, started, error=str(exc)))
                logger.warning(
                    "provider request error",
                    extra={"provider": provider, "error": str(exc)},
                )
                await self._failover_pause(started)
                continue

            if self._should_failover(response):
                attempts.append(
                    self._attempt(provider, started, status_code=response.status_code)
                )
                logger.warning(
                    "provider failed over",
                    extra={"provider": provider, "status_code": response.status_code},
                )
                await response.aread()
                await self._failover_pause(started)
                continue

            if response.status_code in NON_RETRYABLE_STATUSES:
                body = await response.aread()
                attempts.append(
                    self._attempt(provider, started, status_code=response.status_code)
                )
                logger.warning(
                    "provider rejected request; not failing over",
                    extra={"provider": provider, "status_code": response.status_code},
                )
                raise ProviderClientError(
                    provider, response.status_code, _upstream_error_message(body)
                )

            try:
                response.raise_for_status()
                return cast(JsonDict, response.json())
            except (httpx.HTTPStatusError, json.JSONDecodeError) as exc:
                attempts.append(
                    self._attempt(
                        provider,
                        started,
                        status_code=response.status_code,
                        error=str(exc),
                    )
                )
                logger.exception(
                    "provider response parse failed", extra={"provider": provider}
                )
                await self._failover_pause(started)

        raise ProviderRoutingError(attempts)

    async def stream_chat_completion(
        self,
        payload: JsonDict,
        tenant: TenantConfig,
        inbound_headers: Mapping[str, str],
    ) -> AsyncIterator[bytes]:
        """Stream SSE chunks from the first healthy provider."""

        attempts: list[ProviderAttempt] = []
        for provider in tenant.provider_preference:
            started = time.perf_counter()
            provider_payload = self._payload_for_provider(
                provider, payload | {"stream": True}, tenant
            )

            try:
                async with self._client.stream(
                    "POST",
                    self._url_for(provider, stream=True),
                    json=provider_payload,
                    headers=self._headers_for(provider, inbound_headers, stream=True),
                ) as response:
                    if self._should_failover(response):
                        attempts.append(
                            self._attempt(
                                provider, started, status_code=response.status_code
                            )
                        )
                        await response.aread()
                        await self._failover_pause(started)
                        continue
                    if response.status_code in NON_RETRYABLE_STATUSES:
                        body = await response.aread()
                        rejected = ProviderClientError(
                            provider,
                            response.status_code,
                            _upstream_error_message(body),
                        )
                        logger.warning(
                            "stream provider rejected request; not failing over",
                            extra={
                                "provider": provider,
                                "status_code": response.status_code,
                            },
                        )
                        yield self._sse_error("upstream_rejected", str(rejected))
                        return
                    response.raise_for_status()
                    async for chunk in self._intercept_sse(
                        response.aiter_bytes(), provider
                    ):
                        yield chunk
                    return
            except httpx.HTTPError as exc:
                attempts.append(self._attempt(provider, started, error=str(exc)))
                logger.warning(
                    "stream provider error",
                    extra={"provider": provider, "error": str(exc)},
                )
                await self._failover_pause(started)
                continue

        error = ProviderRoutingError(attempts)
        logger.error(
            "all stream providers failed",
            extra={"attempts": [asdict(a) for a in attempts]},
        )
        yield self._sse_error("upstream_unavailable", str(error))

    async def _intercept_sse(
        self,
        chunks: AsyncIterator[bytes],
        provider: ProviderName,
    ) -> AsyncIterator[bytes]:
        """Relay provider SSE bytes, terminating the stream on an injection marker.

        Bytes are released one complete SSE event (``\n\n``) at a time, after
        the event plus the trailing ``sse_scan_tail_bytes`` of already-released
        text has been scanned, so a marker split across network chunks inside
        one event is never forwarded. On a match the offending event is
        dropped, a single ``gateway_error`` event is sent and the stream ends.
        """
        released_tail = ""
        pending = b""
        max_tail = self._settings.sse_scan_tail_bytes
        max_pending = 65_536
        try:
            async for chunk in chunks:
                started = time.perf_counter_ns()
                pending += chunk
                boundary = pending.rfind(b"\n\n")
                if boundary < 0 and len(pending) < max_pending:
                    continue
                cut = boundary + 2 if boundary >= 0 else len(pending)
                ready, pending = pending[:cut], pending[cut:]
                text = ready.decode("utf-8", errors="ignore")
                if self._scan_for_markers(released_tail + text, provider):
                    yield self._sse_error(
                        "stream_marker_detected",
                        "unsafe marker detected in provider stream; stream terminated",
                    )
                    return
                released_tail = (released_tail + text)[-max_tail:]
                self._warn_scan_overhead(started, provider)
                yield ready
            if pending:
                text = pending.decode("utf-8", errors="ignore")
                if self._scan_for_markers(released_tail + text, provider):
                    yield self._sse_error(
                        "stream_marker_detected",
                        "unsafe marker detected in provider stream; stream terminated",
                    )
                    return
                yield pending
        finally:
            aclose = getattr(chunks, "aclose", None)
            if aclose is not None:
                await aclose()

    @staticmethod
    def _scan_for_markers(window: str, provider: ProviderName) -> bool:
        matches = [p.pattern for p in INJECTION_PATTERNS if p.search(window)]
        if matches:
            logger.warning(
                "sse injection marker detected; terminating stream",
                extra={"provider": provider, "patterns": matches[:3]},
            )
        return bool(matches)

    @staticmethod
    def _warn_scan_overhead(started_ns: int, provider: ProviderName) -> None:
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        if elapsed_ms > 10:
            logger.warning(
                "sse scan overhead exceeded budget",
                extra={"provider": provider, "elapsed_ms": elapsed_ms},
            )

    def _payload_for_provider(
        self,
        provider: ProviderName,
        payload: JsonDict,
        tenant: TenantConfig,
    ) -> JsonDict:
        compacted = self._compact_payload(payload, tenant.max_input_tokens)
        if provider == "anthropic":
            return self._to_anthropic_payload(compacted)
        if provider == "openai":
            return compacted | {
                "model": compacted.get("model") or self._settings.openai.default_model
            }
        return compacted | {
            "model": compacted.get("model") or self._settings.local.default_model
        }

    @staticmethod
    def _compact_payload(payload: JsonDict, max_tokens: int) -> JsonDict:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        budget_chars = max_tokens * 4
        system_messages: list[object] = []
        recent_messages: list[object] = []
        used = 0

        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                system_messages.append(message)
                used += len(json.dumps(message, separators=(",", ":")))

        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "system":
                continue
            size = len(json.dumps(message, separators=(",", ":")))
            if used + size > budget_chars and recent_messages:
                break
            recent_messages.append(message)
            used += size

        return payload | {"messages": system_messages + list(reversed(recent_messages))}

    def _to_anthropic_payload(self, payload: JsonDict) -> JsonDict:
        messages = payload.get("messages")
        system_parts: list[str] = []
        anthropic_messages: list[JsonDict] = []

        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "user"))
                content = self._stringify_content(message.get("content", ""))
                if role == "system":
                    system_parts.append(content)
                elif role == "assistant":
                    anthropic_messages.append({"role": "assistant", "content": content})
                else:
                    anthropic_messages.append({"role": "user", "content": content})

        return {
            "model": payload.get("model") or self._settings.anthropic.default_model,
            "system": "\n\n".join(part for part in system_parts if part),
            "messages": anthropic_messages
            or [
                {
                    "role": "user",
                    "content": self._stringify_content(payload.get("prompt", "")),
                }
            ],
            "max_tokens": _max_tokens(payload.get("max_tokens")),
            "temperature": payload.get("temperature", 0.2),
            "stream": bool(payload.get("stream", False)),
        }

    @staticmethod
    def _stringify_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _url_for(self, provider: ProviderName, *, stream: bool) -> str:
        config = self._config_for(provider)
        path = config.stream_path if stream else config.chat_path
        return f"{config.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _headers_for(
        self,
        provider: ProviderName,
        inbound_headers: Mapping[str, str],
        *,
        stream: bool,
    ) -> dict[str, str]:
        config = self._config_for(provider)
        headers = {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            "User-Agent": "multi-tenant-ai-gateway/1.0",
        }
        api_key = config.api_key.get_secret_value().strip()
        if provider == "anthropic":
            if api_key:
                headers["x-api-key"] = api_key
            headers["anthropic-version"] = inbound_headers.get(
                "anthropic-version", "2023-06-01"
            )
        elif api_key:
            # Keyless upstreams (Ollama, vLLM, llama.cpp server) get no auth header:
            # "Bearer " with an empty token is an illegal header value in httpx, which
            # made every request to a keyless provider fail before it was sent.
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _config_for(self, provider: ProviderName) -> ProviderConfig:
        return cast(ProviderConfig, getattr(self._settings, provider))

    @staticmethod
    def _should_failover(response: httpx.Response) -> bool:
        return response.status_code == 429 or 500 <= response.status_code <= 599

    @staticmethod
    def _attempt(
        provider: ProviderName,
        started: float,
        *,
        status_code: int | None = None,
        error: str | None = None,
    ) -> ProviderAttempt:
        return ProviderAttempt(
            provider=provider,
            status_code=status_code,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=error,
        )

    async def _failover_pause(self, started: float) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        remaining_ms = max(0, self._settings.failover_budget_ms - int(elapsed_ms))
        if remaining_ms:
            await asyncio.sleep(remaining_ms / 1000)

    @staticmethod
    def _sse_error(code: str, message: str) -> bytes:
        payload = json.dumps(
            {"error": {"code": code, "message": message}}, separators=(",", ":")
        )
        return f"event: gateway_error\ndata: {payload}\n\n".encode("utf-8")
