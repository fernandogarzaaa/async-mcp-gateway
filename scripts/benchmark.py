"""High-concurrency asyncio benchmark for the streaming AI gateway."""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    base_url: str
    tenants: int
    concurrency: int
    requests_per_tenant: int
    timeout_seconds: float
    stream: bool


@dataclass(slots=True)
class RequestResult:
    tenant_id: str
    status_code: int | None
    latency_ms: float
    bytes_read: int
    error: str | None = None


def parse_options() -> BenchmarkOptions:
    parser = argparse.ArgumentParser(
        description="Benchmark the multi-tenant AI gateway."
    )
    parser.add_argument(
        "--base-url", default=os.getenv("BENCHMARK_BASE_URL", "http://127.0.0.1:8000")
    )
    parser.add_argument(
        "--tenants", type=int, default=int(os.getenv("BENCHMARK_TENANTS", "100"))
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("BENCHMARK_CONCURRENCY", "100")),
    )
    parser.add_argument(
        "--requests-per-tenant",
        type=int,
        default=int(os.getenv("BENCHMARK_REQUESTS_PER_TENANT", "10")),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("BENCHMARK_TIMEOUT_SECONDS", "45")),
    )
    parser.add_argument("--non-stream", action="store_true")
    args = parser.parse_args()
    return BenchmarkOptions(
        base_url=args.base_url.rstrip("/"),
        tenants=max(1, args.tenants),
        concurrency=max(1, args.concurrency),
        requests_per_tenant=max(1, args.requests_per_tenant),
        timeout_seconds=max(1.0, args.timeout_seconds),
        stream=not args.non_stream,
    )


async def run_benchmark(options: BenchmarkOptions) -> list[RequestResult]:
    timeout = httpx.Timeout(options.timeout_seconds, connect=10.0)
    limits = httpx.Limits(
        max_connections=options.concurrency * 2,
        max_keepalive_connections=options.concurrency,
    )
    semaphore = asyncio.Semaphore(options.concurrency)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=True) as client:
        tasks = [
            asyncio.create_task(
                run_one_request(client, semaphore, options, tenant_index, request_index)
            )
            for tenant_index in range(options.tenants)
            for request_index in range(options.requests_per_tenant)
        ]
        return await gather_with_progress(tasks)


async def gather_with_progress(
    tasks: list[asyncio.Task[RequestResult]],
) -> list[RequestResult]:
    results: list[RequestResult] = []
    total = len(tasks)
    completed = 0
    started = time.perf_counter()
    for task in asyncio.as_completed(tasks):
        results.append(await task)
        completed += 1
        if completed % max(1, total // 10) == 0 or completed == total:
            elapsed = time.perf_counter() - started
            print(f"completed={completed}/{total} elapsed={elapsed:.2f}s")
    return results


async def run_one_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    options: BenchmarkOptions,
    tenant_index: int,
    request_index: int,
) -> RequestResult:
    tenant_id = f"tenant-{tenant_index:03d}"
    headers = {
        "X-Tenant-ID": tenant_id,
        "Authorization": f"Bearer tenant-token-{tenant_index:03d}",
        "Accept": "text/event-stream" if options.stream else "application/json",
    }
    payload: dict[str, Any] = {
        "model": "benchmark-model",
        "stream": options.stream,
        "messages": [
            {"role": "system", "content": "Return concise benchmark output."},
            {
                "role": "user",
                "content": (
                    f"tenant={tenant_id} request={request_index} produce a "
                    "short streamed response"
                ),
            },
        ],
        "max_tokens": 64,
        "temperature": 0.1,
    }

    async with semaphore:
        started = time.perf_counter()
        bytes_read = 0
        try:
            if options.stream:
                async with client.stream(
                    "POST",
                    f"{options.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    async for chunk in response.aiter_bytes():
                        bytes_read += len(chunk)
                    latency_ms = (time.perf_counter() - started) * 1000
                    return RequestResult(
                        tenant_id, response.status_code, latency_ms, bytes_read
                    )

            response = await client.post(
                f"{options.base_url}/v1/chat/completions", json=payload, headers=headers
            )
            bytes_read = len(response.content)
            latency_ms = (time.perf_counter() - started) * 1000
            return RequestResult(
                tenant_id, response.status_code, latency_ms, bytes_read
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            return RequestResult(
                tenant_id,
                None,
                latency_ms,
                bytes_read,
                error=f"{type(exc).__name__}: {exc}",
            )


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percent / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def report(results: list[RequestResult]) -> None:
    latencies = [result.latency_ms for result in results]
    successes = [
        result
        for result in results
        if result.status_code is not None and 200 <= result.status_code < 300
    ]
    errors = [
        result
        for result in results
        if result.status_code is None or result.status_code >= 400
    ]
    status_counts: dict[str, int] = {}
    for result in results:
        key = str(result.status_code) if result.status_code is not None else "exception"
        status_counts[key] = status_counts.get(key, 0) + 1

    print("")
    print("benchmark summary")
    print(f"requests_total={len(results)}")
    print(f"successes={len(successes)}")
    print(f"errors={len(errors)}")
    print(f"error_drop_rate={len(errors) / max(1, len(results)):.4f}")
    print(f"latency_avg_ms={statistics.fmean(latencies) if latencies else 0.0:.2f}")
    print(f"latency_p50_ms={percentile(latencies, 50):.2f}")
    print(f"latency_p95_ms={percentile(latencies, 95):.2f}")
    print(f"latency_p99_ms={percentile(latencies, 99):.2f}")
    print(f"bytes_total={sum(result.bytes_read for result in results)}")
    print(f"status_counts={status_counts}")

    sample_errors = [result for result in errors if result.error][:5]
    for index, result in enumerate(sample_errors, start=1):
        print(
            f"sample_error_{index} tenant={result.tenant_id} "
            f"latency_ms={result.latency_ms:.2f} error={result.error}"
        )


async def main() -> None:
    options = parse_options()
    started = time.perf_counter()
    results = await run_benchmark(options)
    elapsed = time.perf_counter() - started
    report(results)
    print(f"wall_clock_seconds={elapsed:.2f}")
    print(f"throughput_rps={len(results) / elapsed if elapsed > 0 else 0.0:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
