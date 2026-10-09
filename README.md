# async-mcp-gateway

An asynchronous, multi-tenant gateway for LLM chat completions and Model Context Protocol (MCP) tools, built on FastAPI and Redis.

- **Tenant authentication**: every request (except health checks) must carry an `X-Tenant-ID` header and a matching `Authorization: Bearer <token>`.
- **Rate limiting**: an atomic Redis token bucket per tenant, enforcing requests-per-minute and tokens-per-minute quotas. Responses carry rate-limit headers; exhausted quotas return `429`.
- **Provider failover**: each tenant has an ordered provider preference (`openai`, `anthropic`, `local`). The router tries them in order and fails over on upstream errors within a configurable budget.
- **Streaming**: `stream: true` requests are proxied as Server-Sent Events, with incremental scanning of the stream across chunk boundaries.
- **MCP process supervision** (`app/services/mcp_supervisor.py`): a pool of stdio MCP server processes isolated per tenant and server, with health checks, restarts and idle reaping.

> Status: beta. Tenants are an in-memory mock registry defined in `app/core/config.py` (`tenant-alpha`, `tenant-beta`). Replace it with a real store before production use.

## Endpoints

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/healthz` | none | Liveness: returns `{"status": "ok"}`. |
| `GET` | `/readyz` | none | Readiness: pings Redis; `503` if Redis is unreachable. |
| `POST` | `/v1/chat/completions` | tenant | OpenAI-style chat completion. Set `"stream": true` for SSE. |

Error responses: `400` invalid JSON, `401`/`403` missing or bad tenant credentials, `429` quota exhausted, `502` all providers failed, `503` rate limiter unavailable.

Example:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "X-Tenant-ID: tenant-alpha" \
  -H "Authorization: Bearer alpha-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Quick start

### Docker Compose

```bash
cp .env.example .env   # then fill in provider API keys
docker compose up --build
```

This starts the gateway on `http://127.0.0.1:8000` and Redis on port 6379.

### Local (uv)

Requires Python 3.11–3.13, [uv](https://docs.astral.sh/uv/) and a running Redis.

```bash
uv sync                      # installs runtime + dev dependencies from uv.lock
cp .env.example .env         # fill in provider API keys
docker run -d -p 6379:6379 redis:7-alpine   # if you don't have Redis locally
uv run uvicorn app.main:app --reload
```

## Configuration

Settings are read from environment variables and `.env` via Pydantic Settings (`app/core/config.py`). Nested provider settings use a double underscore. See `.env.example` for the full list.

| Variable | Default | Meaning |
| --- | --- | --- |
| `APP_NAME` | `multi-tenant-ai-gateway` | FastAPI title / log name |
| `LOG_LEVEL` | `INFO` | Logging threshold |
| `REDIS_URI` | `redis://127.0.0.1:6379/0` | Redis used by the rate limiter |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Upstream provider request timeout |
| `STREAM_CONNECT_TIMEOUT_SECONDS` | `10` | Upstream streaming connect timeout |
| `FAILOVER_BUDGET_MS` | `100` | Failover grace budget (0–1000 ms) |
| `SSE_SCAN_TAIL_BYTES` | `512` | Bytes kept between SSE chunks for scanning (64–4096) |
| `OPENAI__API_KEY`, `OPENAI__BASE_URL`, `OPENAI__CHAT_PATH`, `OPENAI__STREAM_PATH`, `OPENAI__DEFAULT_MODEL` | OpenAI defaults | OpenAI provider |
| `ANTHROPIC__API_KEY`, `ANTHROPIC__BASE_URL`, … | Anthropic defaults | Anthropic provider |
| `LOCAL__API_KEY`, `LOCAL__BASE_URL`, … | `http://127.0.0.1:3000` | Local fallback provider |

Never commit a real `.env`; it is ignored by git.

## Development

```bash
uv sync
uv run pytest                     # tests (coverage gate: 70%)
uv run ruff check app tests scripts mock_mcp_server.py
uv run black --check app tests scripts mock_mcp_server.py
uv run mypy --strict --explicit-package-bases app tests scripts mock_mcp_server.py
```

Tests use an in-memory Redis mock, so no Redis server is needed to run them. `mock_mcp_server.py` is a zero-dependency stdio JSON-RPC server used to exercise the MCP supervisor.

CI (`.github/workflows/ci.yml`) runs lint/format, mypy and pytest (with a Redis service) on pushes to `main` and on pull requests.

## Load testing

`scripts/benchmark.py` fires concurrent chat-completion requests across many tenants and reports latency statistics.

```bash
uv run python scripts/benchmark.py --base-url http://127.0.0.1:8000 \
  --tenants 100 --concurrency 100 --requests-per-tenant 10 [--non-stream]
```

Options can also be set with `BENCHMARK_BASE_URL`, `BENCHMARK_TENANTS`, `BENCHMARK_CONCURRENCY`, `BENCHMARK_REQUESTS_PER_TENANT` and `BENCHMARK_TIMEOUT_SECONDS`. With Docker Compose, the `benchmark` service runs it against the gateway:

```bash
docker compose --profile test up --build benchmark
```

Notes: the benchmark sends tenant IDs `tenant-000`, `tenant-001`, … with tokens `tenant-token-000`, …, so those tenants must exist in the registry (the default registry only has `tenant-alpha` and `tenant-beta`, so other requests are rejected with `403`). It also opens an HTTP/2 client, which needs the `h2` package (`pip install 'httpx[http2]'`).

## License

MIT, as declared in `pyproject.toml` (no LICENSE file is committed yet).
