# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- HTTP routes for MCP: `GET /v1/mcp/servers` and `POST /v1/mcp/{server}`, backed by the existing stdio supervisor and enabled with `MCP_CONFIG_FILE`
- `mcp_servers.example.json`, `examples/notes_mcp_server.py` (official `mcp` SDK, 1.x and 2.x) and `scripts/mcp_http_client.py`
- `initialize` support in `mock_mcp_server.py`; JSON-RPC notifications are forwarded without waiting for a reply

### Fixed
- The idle reaper killed MCP processes that still had a request in flight (any call longer than `idle_ttl_seconds`, and a flaky supervisor test on slow machines)
- A request failing because its process stopped no longer triggers a second, redundant restart
- `.env.example` set `ALLOWED_ORIGINS=*`, which crashed startup after `cp .env.example .env`; it is now `["*"]`
- Formatting for black 26

### Added / Changed
- README rewritten from the code: endpoints, quick start, configuration, development commands, load testing
- uv.lock for reproducible installs
- Dependabot config for uv, Docker and GitHub Actions (weekly)
- SECURITY.md with private reporting contact
