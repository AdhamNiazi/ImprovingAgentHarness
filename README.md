# LLM Evaluation Pipeline

End-to-end pipeline for serving, evaluating, and profiling a local LLM via
[ollama](https://ollama.com) and the
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Model: **qwen2.5:1.5b** (configurable via `OLLAMA_MODEL` env var).

## Prerequisites

- Python 3.10+
- Node.js 18+
- [ollama](https://ollama.com/download) installed and on PATH
- ~2 GB disk for the model weights

## Quick Start

### Option 1: Web Dashboard (recommended)

```bash
python -m venv .venv                          # create virtual environment
.venv\Scripts\activate                        # Windows
# source .venv/bin/activate                   # macOS/Linux
pip install -r requirements.txt               # install Python deps
cd web && npm install && cd ..                # install frontend deps

# Terminal 1: Start the backend API
python -m uvicorn api.server:app --port 8000

# Terminal 2: Start the frontend
cd web && npm run dev
```

Open http://localhost:5173 to access the dashboard.

### Option 2: CLI

```bash
pip install -r requirements.txt
python serve/serve.py             # start ollama + pull model
python serve/client.py            # run sample generations
```

### With Make

```bash
make install      # create venv + install all deps
make api          # start backend API on :8000
make web          # start frontend dev server on :5173
make serve        # start ollama
make eval         # run benchmarks
make perf         # load test
make guardrails   # determinism checks
```

## Project Structure

```
api/
  server.py         — FastAPI backend (REST + WebSocket for live logs)

web/                — React + Vite + Tailwind frontend dashboard
  src/
    App.tsx         — Main app with sidebar navigation
    api.ts          — API client + WebSocket connection
    panels/         — StatusPanel, PlaygroundPanel, EvalPanel,
                      PerfPanel, GuardrailsPanel, LogPanel

serve/
  serve.py          — Start ollama, pull model, verify health
  client.py         — Sample prompt generations with timing

eval_runner/
  model.py          — Custom lm-eval-harness LM wrapper for ollama
  run_eval.py       — Benchmark runner (MMLU, HellaSwag, custom task)
  custom_task/      — python_logic benchmark (JSONL + YAML config)
  results/          — Benchmark outputs and summary table

perf/
  load_test.py      — Async concurrent load generator
  metrics.csv       — Collected latency / throughput data
  analysis.ipynb    — Plots and commentary

guardrails/
  validate.py       — Determinism tests + output validation
  README.md         — What was tested and where nondeterminism persists
```

## Web Dashboard

The dashboard provides a unified interface to control and visualise the entire pipeline:

| Tab | Features |
|-----|----------|
| **Status** | Ollama health, model availability, results/metrics status |
| **Playground** | Interactive prompt-to-response with model metadata |
| **Evaluation** | Select tasks, set limits, run benchmarks, view results table |
| **Performance** | Run load tests, latency bar charts, full metrics table |
| **Guardrails** | Quick determinism check, full validation suite |

All operations stream real-time logs via WebSocket to the Live Logs sidebar.

## Part A — Serving

`serve.py` locates the ollama binary, starts the server if needed, pulls
`qwen2.5:1.5b`, and confirms the API is healthy.  `client.py` sends five
diverse prompts (factual, reasoning, code, creative, extraction) and prints
responses with timing.

## Part B — Evaluation

`model.py` implements a custom `lm_eval.api.model.LM` subclass registered as
`"ollama"`.  It supports:

- **generate_until** — via `/api/generate` with stop sequences
- **loglikelihood** — via `/v1/completions` echo mode (fast) with fallback to
  token-by-token forced decoding via `/api/generate`
- **loglikelihood_rolling** — delegates to the same loglikelihood engine

Caching is provided by wrapping the model with `lm_eval.api.model.CachingLM`
(SQLite-backed), so repeated runs skip API calls for identical prompts.

Benchmarks evaluated:

| Task              | Type               | Source                  |
|-------------------|--------------------|-------------------------|
| mmlu_abstract_algebra | multiple_choice | lm-eval-harness built-in |
| hellaswag         | multiple_choice    | lm-eval-harness built-in |
| python_logic      | multiple_choice    | Custom (25 questions)    |

## Part C — Performance

`load_test.py` fires concurrent streaming requests at ollama and collects:

- Time-to-first-token (TTFT)
- Tokens per second
- P50 / P95 / P99 end-to-end latency
- GPU utilisation (if available)

Configurations sweep: concurrency (1/2/4/8), prompt length (short/long),
cache state (cold/warm), and stop-sequence presence.

`analysis.ipynb` produces four plots with commentary.

## Part D — Guardrails

`validate.py` verifies:

1. **Determinism** — identical prompts with `temperature=0, seed=42, top_k=1`
   produce byte-identical outputs across three trials.
2. **Output validation** — regex and JSON-schema validators for benchmark
   answers.
3. **Nondeterminism report** — character-level diffs and likely root causes.

See `guardrails/README.md` for detailed findings.
