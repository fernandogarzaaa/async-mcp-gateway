"""Runtime configuration for the multi-tenant AI gateway."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openai", "anthropic", "local"]


class ProviderConfig(BaseSettings):
    """Configuration for a single upstream model provider."""

    base_url: str
    api_key: SecretStr = SecretStr("")
    chat_path: str
    stream_path: str
    default_model: str


class TenantConfig(BaseSettings):
    """Mock tenant registry entry used by the authentication middleware."""

    tenant_id: str
    bearer_token: SecretStr
    rpm_limit: int = Field(gt=0)
    tpm_limit: int = Field(gt=0)
    provider_preference: list[ProviderName]
    max_input_tokens: int = Field(default=24_000, gt=0)


class Settings(BaseSettings):
    """Application settings loaded from environment and sane local defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "multi-tenant-ai-gateway"
    log_level: str = "INFO"
    redis_uri: str = "redis://127.0.0.1:6379/0"
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    stream_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    failover_budget_ms: int = Field(default=100, ge=0, le=1_000)
    sse_scan_tail_bytes: int = Field(default=512, ge=64, le=4096)

    openai: ProviderConfig = ProviderConfig(
        base_url="https://api.openai.com",
        chat_path="/v1/chat/completions",
        stream_path="/v1/chat/completions",
        default_model="gpt-4o-mini",
    )
    anthropic: ProviderConfig = ProviderConfig(
        base_url="https://api.anthropic.com",
        chat_path="/v1/messages",
        stream_path="/v1/messages",
        default_model="claude-3-5-haiku-latest",
    )
    local: ProviderConfig = ProviderConfig(
        base_url="http://127.0.0.1:3000",
        chat_path="/v1/chat/completions",
        stream_path="/v1/chat/completions",
        default_model="local-fallback",
    )

    tenants: dict[str, TenantConfig] = {
        "tenant-alpha": TenantConfig(
            tenant_id="tenant-alpha",
            bearer_token=SecretStr("alpha-secret-token"),
            rpm_limit=120,
            tpm_limit=120_000,
            provider_preference=["openai", "anthropic", "local"],
            max_input_tokens=32_000,
        ),
        "tenant-beta": TenantConfig(
            tenant_id="tenant-beta",
            bearer_token=SecretStr("beta-secret-token"),
            rpm_limit=60,
            tpm_limit=60_000,
            provider_preference=["anthropic", "openai", "local"],
            max_input_tokens=16_000,
        ),
    }

    allowed_origins: list[str | AnyHttpUrl] = ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object."""

    return Settings()
