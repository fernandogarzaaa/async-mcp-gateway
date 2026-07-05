"""Environment-backed settings for the async MCP gateway."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings parsed from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    PRIMARY_PROVIDER_URL: str = Field(
        default="https://api.openai.com/v1/chat/completions"
    )
    BACKUP_PROVIDER_URL: str = Field(default="https://api.anthropic.com/v1/messages")
    GATEWAY_TIMEOUT: float = Field(default=30.0, gt=0)
    DEFAULT_MCP_TTL: int = Field(default=300, gt=0)


settings = Settings()
