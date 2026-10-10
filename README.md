# async-mcp-gateway

An asynchronous, multi-tenant gateway for LLM chat completions and Model Context Protocol (MCP) tools, built on FastAPI and Redis.

- **Tenant authentication**: every request (except health checks) must carry an `X-Tenant-ID` header and a matching `Authorization: Bearer <token>`.
- **Rate limiting**: an atomic Redis token bucket per tenant, enforcing requests-per-minute and tokens-per-minute quotas. Responses carry rate-limit headers; exhausted quotas return `429`.
- **Provider failover**: each tenant has an ordered provider preference (`openai`, `anthropic`, `local`). The router tries them in order and fails over on upstream errors within a configurable budget.
- **Streaming**: `stream: true` requests are proxied as Server-Sent Events, with incremental scanning of the stream across chunk boundaries.
- **MCP over HTTP** (`POST /v1/mcp/{server}`): forwards JSON-RPC messages to stdio MCP servers, each running as its own process per tenant and server (`app/services/mcp_supervisor.py`), with per-tenant allow-lists, health checks, crash recovery and idle reaping.

> Status: beta. Tenants are an in-memory mock registry defined in `app/core/config.py` (`tenant-alpha`, `tenant-beta`). Replace it with a real store before production use.

## Endpoints

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/healthz` | none | Liveness: returns `{"status": "ok"}`. |
| `GET` | `/readyz` | none | Readiness: pings Redis; `503` if Redis is unreachable. |
| `POST` | `/v1/chat/completions` | tenant | OpenAI-style chat completion. Set `"stream": true` for SSE. |
| `GET` | `/v1/mcp/servers` | tenant | MCP servers this tenant may use (needs `MCP_CONFIG_FILE`). |
| `POST` | `/v1/mcp/{server}` | tenant | Send one JSON-RPC message to the tenant's MCP server process. Requests (with `id`) return the server's response; notifications (no `id`) return `202`. |

Error responses: `400` invalid JSON / JSON-RPC, `401`/`403` missing or bad tenant credentials (or MCP server not allowed for the tenant), `404` unknown MCP server or MCP not configured, `429` quota exhausted, `502` all providers failed or the MCP process died, `503` rate limiter unavailable, `504` MCP request timed out.

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

No provider keys? Tenants fail over in order, so pointing `LOCAL__BASE_URL` at any OpenAI-compatible server (llama.cpp, Ollama's `/v1`, LM Studio) is enough; requests to providers without a key fail with `401` and fall through to `local`.

## Using MCP servers through the gateway

1. Describe your stdio MCP servers in a JSON file. `mcp_servers.example.json` ships two: `mock` (zero-dependency `mock_mcp_server.py`) and `notes` (`examples/notes_mcp_server.py`, a real server built with the official `mcp` SDK). Optional `tenant_policies` restrict which servers each tenant may use; tenants without a policy may use every server.
2. Start the gateway with it:

   ```bash
   MCP_CONFIG_FILE=mcp_servers.example.json uv run uvicorn app.main:app
   ```

3. Talk MCP over HTTP (the included client does initialize → tools/list → tools/call):

   ```bash
   uv run python scripts/mcp_http_client.py --base-url http://127.0.0.1:8000 \
     --server notes --call add_note '{"text": "hello"}' --call list_notes '{}'
   ```

   or with curl:

   ```bash
   curl http://127.0.0.1:8000/v1/mcp/mock \
     -H "X-Tenant-ID: tenant-alpha" -H "Authorization: Bearer alpha-secret-token" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"x":1}}}'
   ```

Each tenant gets its own process per server (it receives `TENANT_ID` in its environment), so state never leaks between tenants. Processes persist between HTTP calls, so a client initializes once; if a process is idle longer than `idle_ttl_seconds` it is reaped and the next call starts a fresh one (re-run `initialize`). MCP calls count one request against the tenant's RPM quota.

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
| `MCP_CONFIG_FILE` | unset | JSON file with MCP server definitions; enables `/v1/mcp/*` |
| `ALLOWED_ORIGINS` | `["*"]` | CORS origins, as a JSON list |
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
