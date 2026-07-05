"""Streaming LLM proxy with provider failover and SSE inspection."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

INJECTION_PATTERN = re.compile(
    r"(?i)(ignore\s+previous\s+instructions|<\s*system\s*>|sudo\s+|rm\s+-rf|"
    r"curl\s+http|wget\s+http|powershell\s+-|cmd\.exe)"
)


class ResilientStreamRouter:
    """Stream from a primary provider and fail over to a backup provider."""

    def __init__(self, timeout_seconds: float | None = None) -> None:
        timeout = timeout_seconds or settings.GATEWAY_TIMEOUT
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        """Close the underlying HTTP client."""

        await self.client.aclose()

    async def stream_response(
        self,
        payload: dict[str, Any],
        tenant_config: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Yield inspected SSE frames from the active provider."""

        primary_url = str(
            tenant_config.get("primary_provider_url") or settings.PRIMARY_PROVIDER_URL
        )
        backup_url = str(
            tenant_config.get("backup_provider_url") or settings.BACKUP_PROVIDER_URL
        )
        headers = self._provider_headers(tenant_config)

        try:
            async for frame in self._stream_provider(primary_url, payload, headers):
                yield frame
            return
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            ProviderFailoverRequired,
        ) as exc:
            logger.warning(
                "primary provider failed; using backup",
                extra={
                    "primary_url": primary_url,
                    "backup_url": backup_url,
                    "error": str(exc),
                },
            )

        backup_payload = self._translate_to_backup_schema(payload)
        async for frame in self._stream_provider(backup_url, backup_payload, headers):
            yield frame

    async def _stream_provider(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncGenerator[str, None]:
        async with self.client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code in {429, 502, 503}:
                await response.aread()
                raise ProviderFailoverRequired(
                    f"provider returned {response.status_code}"
                )
            response.raise_for_status()
            async for line in response.aiter_text():
                if not line:
                    continue
                if INJECTION_PATTERN.search(line):
                    logger.warning("blocked unsafe provider stream frame")
                    yield self._sse_error(
                        "unsafe_stream_frame",
                        "blocked unsafe stream frame",
                    )
                    return
                yield line

    @staticmethod
    def _translate_to_backup_schema(payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        system_parts: list[str] = []
        translated_messages: list[dict[str, str]] = []
        if isinstance(messages, list):
            for item in messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "user"))
                content = ResilientStreamRouter._stringify_content(
                    item.get("content", "")
                )
                if role == "system":
                    system_parts.append(content)
                elif role == "assistant":
                    translated_messages.append(
                        {"role": "assistant", "content": content}
                    )
                else:
                    translated_messages.append({"role": "user", "content": content})
        return {
            "model": payload.get("model", "backup-model"),
            "system": "\n\n".join(system_parts),
            "messages": translated_messages or [{"role": "user", "content": ""}],
            "max_tokens": int(payload.get("max_tokens", 1024)),
            "stream": True,
        }

    @staticmethod
    def _stringify_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    @staticmethod
    def _provider_headers(tenant_config: dict[str, Any]) -> dict[str, str]:
        headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
        provider_token = tenant_config.get("provider_token")
        if isinstance(provider_token, str) and provider_token:
            headers["Authorization"] = f"Bearer {provider_token}"
        return headers

    @staticmethod
    def _sse_error(code: str, message: str) -> str:
        payload = json.dumps({"error": {"code": code, "message": message}})
        return f"event: gateway_error\ndata: {payload}\n\n"


class ProviderFailoverRequired(RuntimeError):
    """Raised when the primary provider response should trigger failover."""
