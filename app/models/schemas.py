"""Pydantic V2 schemas for gateway, MCP, and benchmark configuration."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None
JsonObject = dict[str, JsonValue]


class ProviderName(StrEnum):
    """Supported upstream model provider names."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class TransportKind(StrEnum):
    """Supported downstream MCP server transports."""

    STDIO = "stdio"
    SSE = "sse"


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request accepted by downstream MCP tools."""

    model_config = ConfigDict(extra="allow")

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    method: str = Field(min_length=1, max_length=256)
    params: JsonValue = None

    @model_validator(mode="after")
    def require_id_for_request_response(self) -> "JsonRpcRequest":
        if self.id is None and self.method != "notifications/initialized":
            raise ValueError(
                "JSON-RPC requests routed through the supervisor require an id"
            )
        return self


class JsonRpcErrorObject(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: JsonValue = None


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response emitted by an MCP process."""

    model_config = ConfigDict(extra="allow")

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: JsonValue = None
    error: JsonRpcErrorObject | None = None

    @model_validator(mode="after")
    def require_result_or_error(self) -> "JsonRpcResponse":
        if self.result is None and self.error is None:
            raise ValueError("JSON-RPC response must include result or error")
        if self.result is not None and self.error is not None:
            raise ValueError("JSON-RPC response cannot include both result and error")
        return self


class MCPHealthCheckConfig(BaseModel):
    """Health-check behavior for a supervised MCP server process."""

    enabled: bool = True
    interval_seconds: Annotated[float, Field(gt=0)] = 30.0
    timeout_seconds: Annotated[float, Field(gt=0)] = 5.0
    request: JsonRpcRequest = Field(
        default_factory=lambda: JsonRpcRequest(
            id="health",
            method="ping",
            params={},
        ),
    )


class MCPServerConfig(BaseModel):
    """Validated configuration for one MCP server definition."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_.-]+$")
    transport: TransportKind = TransportKind.STDIO
    command: list[str] = Field(min_length=1)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    per_tenant_env: dict[str, dict[str, str]] = Field(default_factory=dict)
    startup_timeout_seconds: Annotated[float, Field(gt=0)] = 10.0
    request_timeout_seconds: Annotated[float, Field(gt=0)] = 30.0
    idle_ttl_seconds: Annotated[float, Field(gt=0)] = 300.0
    max_stderr_lines: Annotated[int, Field(ge=10, le=10_000)] = 500
    health_check: MCPHealthCheckConfig = Field(default_factory=MCPHealthCheckConfig)

    @field_validator("command")
    @classmethod
    def validate_command_parts(cls, command: list[str]) -> list[str]:
        if any(not part.strip() for part in command):
            raise ValueError("command entries must be non-empty strings")
        return command

    def tenant_environment(self, tenant_id: str) -> dict[str, str]:
        tenant_overrides = self.per_tenant_env.get(tenant_id, {})
        return self.env | tenant_overrides | {"TENANT_ID": tenant_id}


class TenantMCPPolicy(BaseModel):
    """Per-tenant authorization and capacity policy for MCP process pools."""

    tenant_id: str = Field(min_length=1, max_length=128)
    allowed_servers: set[str] = Field(default_factory=set)
    max_processes: Annotated[int, Field(ge=1, le=64)] = 8


class MCPSupervisorConfig(BaseModel):
    """Top-level MCP supervisor configuration."""

    model_config = ConfigDict(extra="forbid")

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    tenant_policies: dict[str, TenantMCPPolicy] = Field(default_factory=dict)
    default_idle_ttl_seconds: Annotated[float, Field(gt=0)] = 300.0
    reap_interval_seconds: Annotated[float, Field(gt=0)] = 15.0
    shutdown_grace_seconds: Annotated[float, Field(gt=0)] = 5.0

    @model_validator(mode="after")
    def ensure_server_keys_match_names(self) -> "MCPSupervisorConfig":
        for key, server in self.servers.items():
            if key != server.name:
                raise ValueError(
                    f"server mapping key {key!r} must match server.name {server.name!r}"
                )
        for policy in self.tenant_policies.values():
            unknown = policy.allowed_servers.difference(self.servers)
            if unknown:
                servers = sorted(unknown)
                raise ValueError(
                    f"tenant {policy.tenant_id} references unknown MCP servers: "
                    f"{servers}"
                )
        return self


class GatewayProviderConfig(BaseModel):
    """Provider schema variant for external configuration validation."""

    base_url: str = Field(min_length=1)
    api_key: SecretStr = SecretStr("")
    chat_path: str = Field(min_length=1)
    stream_path: str = Field(min_length=1)
    default_model: str = Field(min_length=1)


class GatewayTenantConfig(BaseModel):
    """Tenant schema variant for external configuration validation."""

    tenant_id: str = Field(min_length=1, max_length=128)
    bearer_token: SecretStr
    rpm_limit: Annotated[int, Field(gt=0)]
    tpm_limit: Annotated[int, Field(gt=0)]
    provider_preference: list[ProviderName] = Field(min_length=1)
    max_input_tokens: Annotated[int, Field(gt=0)] = 24_000


class GatewayRuntimeConfig(BaseModel):
    """Complete gateway configuration payload."""

    model_config = ConfigDict(extra="forbid")

    app_name: str = "multi-tenant-ai-gateway"
    redis_uri: str = "redis://redis:6379/0"
    request_timeout_seconds: Annotated[float, Field(gt=0)] = 30.0
    stream_connect_timeout_seconds: Annotated[float, Field(gt=0)] = 10.0
    failover_budget_ms: Annotated[int, Field(ge=0, le=1_000)] = 100
    tenants: dict[str, GatewayTenantConfig] = Field(default_factory=dict)
    providers: dict[ProviderName, GatewayProviderConfig] = Field(default_factory=dict)
    mcp: MCPSupervisorConfig = Field(default_factory=MCPSupervisorConfig)


class BenchmarkConfig(BaseModel):
    """Validated options for the asyncio benchmark script."""

    base_url: str = "http://127.0.0.1:8000"
    tenants: Annotated[int, Field(ge=1, le=10_000)] = 100
    concurrency: Annotated[int, Field(ge=1, le=10_000)] = 100
    requests_per_tenant: Annotated[int, Field(ge=1)] = 10
    timeout_seconds: Annotated[float, Field(gt=0)] = 45.0
    stream: bool = True
    tenant_token_prefix: str = "tenant-token"
