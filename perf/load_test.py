"""
Concurrent load generator for ollama performance profiling.

Sends short and long prompts at varying concurrency levels, collecting
TTFT, tokens-per-second, and latency percentiles.  Results are written
to perf/metrics.csv.
"""

import asyncio
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
OUT_CSV = os.path.join(os.path.dirname(__file__), "metrics.csv")

SHORT_PROMPTS = [
    "What is 2+2?",
    "Name a primary color.",
    "Is water wet?",
    "Capital of Japan?",
]

LONG_PROMPTS = [
    (
        "Write a detailed explanation of how a neural network learns through "
        "backpropagation. Cover the forward pass, loss computation, gradient "
        "calculation, and weight updates. Include an example with a simple "
        "two-layer network."
    ),
    (
        "Explain the history of the Internet from ARPANET to the modern World "
        "Wide Web. Discuss key milestones, protocols, and the people who were "
        "instrumental in its development."
    ),
    (
        "Describe the process of photosynthesis in detail. Include the light-"
        "dependent reactions and the Calvin cycle, and explain where each step "
        "occurs within the chloroplast."
    ),
    (
        "Compare and contrast TCP and UDP protocols. Discuss their headers, "
        "connection setup, reliability guarantees, flow control mechanisms, "
        "and typical use cases for each protocol."
    ),
]

MAX_TOKENS = 128
CONCURRENCY_LEVELS = [1, 2, 4, 8]
REPEATS = 3


@dataclass
class RequestResult:
    prompt_type: str
    concurrency: int
    ttft_ms: float
    total_latency_ms: float
    tokens_generated: int
    tokens_per_sec: float
    cache_warm: bool
    stop_seq: bool


async def send_request(
    session: aiohttp.ClientSession,
    prompt: str,
    prompt_type: str,
    concurrency: int,
    cache_warm: bool,
    stop_seq: bool,
) -> RequestResult:
    """Send a single streaming request and measure TTFT + throughput."""
    body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": 0,
            "seed": 42,
        },
    }
    if stop_seq:
        body["options"]["stop"] = ["\n\n", ".\n"]

    t_start = time.perf_counter()
    ttft = None
    full_response = ""
    tokens = 0
    tps = 0.0

    async with session.post(
        f"{OLLAMA_URL}/api/generate", json=body, timeout=aiohttp.ClientTimeout(total=300)
    ) as resp:
        resp.raise_for_status()
        async for line in resp.content:
            chunk = line.decode("utf-8").strip()
            if not chunk:
                continue
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue

            if ttft is None and obj.get("response"):
                ttft = (time.perf_counter() - t_start) * 1000

            full_response += obj.get("response", "")

            if obj.get("done"):
                tokens = obj.get("eval_count", 0)
                eval_dur = obj.get("eval_duration", 0)
                if eval_dur > 0 and tokens > 0:
                    tps = tokens / (eval_dur / 1e9)
                else:
                    tps = 0
                break

    total_ms = (time.perf_counter() - t_start) * 1000
    if ttft is None:
        ttft = total_ms

    if tokens == 0:
        tokens = max(1, len(full_response.split()))
    if tps == 0 and total_ms > 0:
        tps = tokens / (total_ms / 1000)

    return RequestResult(
        prompt_type=prompt_type,
        concurrency=concurrency,
        ttft_ms=ttft,
        total_latency_ms=total_ms,
        tokens_generated=tokens,
        tokens_per_sec=tps,
        cache_warm=cache_warm,
        stop_seq=stop_seq,
    )


async def run_batch(
    prompts: list[str],
    prompt_type: str,
    concurrency: int,
    cache_warm: bool,
    stop_seq: bool,
) -> list[RequestResult]:
    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [
            send_request(session, p, prompt_type, concurrency, cache_warm, stop_seq)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)


def get_gpu_util() -> str:
    """Best-effort GPU utilisation from ollama ps."""
    try:
        out = subprocess.check_output(
            ["ollama", "ps"], timeout=5, text=True, stderr=subprocess.DEVNULL
        )
        return out.strip().replace("\n", " | ")
    except Exception:
        return "N/A"


def percentile(vals: list[float], p: int) -> float:
    return float(np.percentile(vals, p)) if vals else 0.0


def main():
    all_results: list[RequestResult] = []
    configs = [
        ("short", SHORT_PROMPTS),
        ("long", LONG_PROMPTS),
    ]

    print(f"Model:        {MODEL}")
    print(f"Endpoint:     {OLLAMA_URL}")
    print(f"Max tokens:   {MAX_TOKENS}")
    print(f"Concurrency:  {CONCURRENCY_LEVELS}")
    print(f"Repeats:      {REPEATS}")
    print(f"GPU info:     {get_gpu_util()}\n")

    for conc in CONCURRENCY_LEVELS:
        for p_type, prompts in configs:
            for stop_seq in [False, True]:
                for cache_pass in range(REPEATS):
                    cache_warm = cache_pass > 0
                    tag = (
                        f"conc={conc}  type={p_type:5s}  "
                        f"stop={stop_seq}  warm={cache_warm}"
                    )
                    print(f"  [{tag}] running …", end="", flush=True)

                    batch = (prompts * ((conc // len(prompts)) + 1))[:conc]
                    results = asyncio.run(
                        run_batch(batch, p_type, conc, cache_warm, stop_seq)
                    )
                    all_results.extend(results)

                    lats = [r.total_latency_ms for r in results]
                    print(
                        f"  p50={percentile(lats,50):.0f}ms  "
                        f"p95={percentile(lats,95):.0f}ms"
                    )

    # ── write CSV ────────────────────────────────────────────────────
    rows: list[dict] = []
    groups: dict[tuple, list[RequestResult]] = {}
    for r in all_results:
        key = (r.concurrency, r.prompt_type, r.cache_warm, r.stop_seq)
        groups.setdefault(key, []).append(r)

    gpu = get_gpu_util()
    for (conc, ptype, warm, stop), items in sorted(groups.items()):
        lats = [i.total_latency_ms for i in items]
        ttfts = [i.ttft_ms for i in items]
        tps_vals = [i.tokens_per_sec for i in items]
        rows.append({
            "concurrency": conc,
            "prompt_type": ptype,
            "cache": "warm" if warm else "cold",
            "stop_seq": stop,
            "ttft_ms": f"{percentile(ttfts, 50):.1f}",
            "tpot": f"{np.mean(tps_vals):.1f}",
            "latency_p50": f"{percentile(lats, 50):.1f}",
            "latency_p95": f"{percentile(lats, 95):.1f}",
            "latency_p99": f"{percentile(lats, 99):.1f}",
            "gpu_util": gpu,
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetrics written to {OUT_CSV}")
    print(f"Total requests: {len(all_results)}")


if __name__ == "__main__":
    main()
