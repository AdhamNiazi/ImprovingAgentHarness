"""
FastAPI backend that exposes the entire LLM evaluation pipeline as HTTP endpoints
and persists every run's metrics into a local SQLite database.

Endpoints:
  GET  /api/status              - ollama health + model info
  POST /api/serve/start         - start ollama & pull model
  POST /api/client/run          - run sample prompts
  POST /api/client/generate     - single prompt generation
  POST /api/eval/run            - run benchmarks
  GET  /api/eval/results        - fetch latest results JSON
  GET  /api/eval/samples        - fetch latest samples JSON
  POST /api/perf/run            - run load test
  GET  /api/perf/metrics        - fetch latest metrics CSV as JSON
  POST /api/guardrails/run      - run determinism + validation tests
  POST /api/guardrails/determinism - quick inline determinism check

  GET  /api/runs                - list all historical runs (paginated)
  GET  /api/runs/{id}           - single run detail + scores
  GET  /api/runs/{id}/samples   - paginated per-question results
  GET  /api/runs/{id}/perf      - perf metrics for a run
  GET  /api/runs/{id}/guardrails - guardrail checks for a run
  GET  /api/history/scores      - all eval scores across all runs

  WS   /ws/logs                 - stream real-time log output
"""

import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests as http_requests
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import db

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

app = FastAPI(title="LLM Eval Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    db.init_db()


# ── WebSocket log streaming ──────────────────────────────────────────
_ws_clients: list[WebSocket] = []


async def broadcast(msg: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


_TQDM_RE = re.compile(
    r"(?P<phase>[^:]+):\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<total>\d+)"
    r"\s*\[(?P<elapsed>[^\]<,]+)<(?P<eta>[^\],]+),\s*(?P<speed>[^\]]+)\]"
)


def _parse_progress(text: str, tag: str) -> dict | None:
    m = _TQDM_RE.search(text)
    if not m:
        return None
    return {
        "type": "progress",
        "tag": tag,
        "phase": m.group("phase").strip(),
        "pct": int(m.group("pct")),
        "current": int(m.group("cur")),
        "total": int(m.group("total")),
        "elapsed": m.group("elapsed").strip(),
        "eta": m.group("eta").strip(),
        "speed": m.group("speed").strip(),
    }


async def run_script(cmd: list[str], tag: str) -> dict:
    await broadcast({"type": "start", "tag": tag})
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    lines: list[str] = []
    buf = b""
    last_progress_time = 0.0

    while True:
        chunk = await proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf or b"\r" in buf:
            idx_n = buf.find(b"\n")
            idx_r = buf.find(b"\r")
            if idx_n == -1:
                idx = idx_r
            elif idx_r == -1:
                idx = idx_n
            else:
                idx = min(idx_n, idx_r)

            segment = buf[:idx]
            buf = buf[idx + 1:]
            if not segment:
                continue

            text = segment.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue

            progress = _parse_progress(text, tag)
            if progress:
                now = time.monotonic()
                if now - last_progress_time >= 0.5:
                    await broadcast(progress)
                    last_progress_time = now
            else:
                lines.append(text)
                await broadcast({"type": "log", "tag": tag, "line": text})

    if buf.strip():
        text = buf.decode("utf-8", errors="replace").rstrip()
        progress = _parse_progress(text, tag)
        if progress:
            await broadcast(progress)
        elif text:
            lines.append(text)
            await broadcast({"type": "log", "tag": tag, "line": text})

    await proc.wait()
    await broadcast({"type": "progress", "tag": tag, "phase": "", "pct": 100,
                     "current": 0, "total": 0, "elapsed": "", "eta": "", "speed": ""})
    await broadcast({"type": "done", "tag": tag, "exit_code": proc.returncode})
    return {"exit_code": proc.returncode, "output": lines}


@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ── Status ───────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    try:
        r = http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        running = True
    except Exception:
        models = []
        running = False

    return {
        "ollama_running": running,
        "ollama_url": OLLAMA_URL,
        "models": models,
        "target_model": MODEL,
        "model_ready": MODEL in " ".join(models) if models else False,
        "has_eval_results": (ROOT / "eval_runner" / "results" / "results.json").exists(),
        "has_perf_metrics": (ROOT / "perf" / "metrics.csv").exists(),
        "total_runs": db.count_runs(),
    }


# ── Serve ────────────────────────────────────────────────────────────

@app.post("/api/serve/start")
async def serve_start():
    return await run_script(
        [sys.executable, str(ROOT / "serve" / "serve.py")], "serve"
    )


# ── Client demo ──────────────────────────────────────────────────────

@app.post("/api/client/run")
async def client_run():
    return await run_script(
        [sys.executable, str(ROOT / "serve" / "client.py")], "client"
    )


@app.post("/api/client/generate")
async def client_generate(body: dict):
    prompt = body.get("prompt", "Hello")
    max_tokens = body.get("max_tokens", 256)
    t0 = time.perf_counter()
    try:
        r = http_requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return {
            "response": data.get("response", ""),
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "eval_count": data.get("eval_count", 0),
            "model": MODEL,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Evaluation ───────────────────────────────────────────────────────

@app.post("/api/eval/run")
async def eval_run(body: dict | None = None):
    body = body or {}
    tasks = body.get("tasks", ["python_logic"])
    limit = body.get("limit", 20)

    run_id = db.create_run("eval", MODEL, {"tasks": tasks, "limit": limit})

    cmd = [
        sys.executable,
        str(ROOT / "eval_runner" / "run_eval.py"),
        "--tasks", *tasks,
    ]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]

    result = await run_script(cmd, "eval")
    db.finish_run(run_id, result["exit_code"])

    if result["exit_code"] == 0:
        _persist_eval(run_id)

    return {**result, "run_id": run_id}


def _persist_eval(run_id: int) -> None:
    results_path = ROOT / "eval_runner" / "results" / "results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        db.save_eval_scores(run_id, data)

    samples_path = ROOT / "eval_runner" / "results" / "samples.json"
    if samples_path.exists():
        with open(samples_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        db.save_eval_samples(run_id, samples)


@app.get("/api/eval/results")
async def eval_results():
    results_path = ROOT / "eval_runner" / "results" / "results.json"
    if not results_path.exists():
        return {"results": None, "message": "No results yet. Run an evaluation first."}
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"results": data}


@app.get("/api/eval/samples")
async def eval_samples():
    samples_path = ROOT / "eval_runner" / "results" / "samples.json"
    if not samples_path.exists():
        return {"samples": None}
    with open(samples_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"samples": data}


# ── Performance ──────────────────────────────────────────────────────

@app.post("/api/perf/run")
async def perf_run():
    run_id = db.create_run("perf", MODEL)

    result = await run_script(
        [sys.executable, str(ROOT / "perf" / "load_test.py")], "perf"
    )
    db.finish_run(run_id, result["exit_code"])

    if result["exit_code"] == 0:
        _persist_perf(run_id)

    return {**result, "run_id": run_id}


def _persist_perf(run_id: int) -> None:
    metrics_path = ROOT / "perf" / "metrics.csv"
    if not metrics_path.exists():
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    db.save_perf_metrics(run_id, rows)


@app.get("/api/perf/metrics")
async def perf_metrics():
    metrics_path = ROOT / "perf" / "metrics.csv"
    if not metrics_path.exists():
        return {"metrics": None, "message": "No metrics yet. Run the load test first."}
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {"metrics": rows}


# ── Guardrails ───────────────────────────────────────────────────────

@app.post("/api/guardrails/run")
async def guardrails_run():
    run_id = db.create_run("guardrails", MODEL)

    result = await run_script(
        [sys.executable, str(ROOT / "guardrails" / "validate.py")], "guardrails"
    )
    db.finish_run(run_id, result["exit_code"])

    for line in result.get("output", []):
        if "[PASS]" in line or "[FAIL]" in line:
            passed = "[PASS]" in line
            db.save_guardrail_check(run_id, "determinism", line, passed)
        elif "[OK]" in line:
            db.save_guardrail_check(run_id, "validation", line, True)

    return {**result, "run_id": run_id}


@app.post("/api/guardrails/determinism")
async def guardrails_determinism_quick(body: dict | None = None):
    body = body or {}
    prompt = body.get("prompt", "What is 2+2?")
    trials = body.get("trials", 3)
    responses = []
    for _ in range(trials):
        try:
            r = http_requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "raw": True,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "top_p": 1,
                        "top_k": 1,
                        "seed": 42,
                        "num_predict": 128,
                    },
                },
                timeout=60,
            )
            r.raise_for_status()
            responses.append(r.json().get("response", ""))
        except Exception as e:
            responses.append(f"ERROR: {e}")

    identical = len(set(responses)) == 1

    run_id = db.create_run("guardrails", MODEL, {"prompt": prompt, "trials": trials})
    db.finish_run(run_id, 0)
    db.save_guardrail_check(run_id, "determinism", prompt, identical,
                            {"trials": trials, "responses": responses})

    return {
        "prompt": prompt,
        "trials": trials,
        "identical": identical,
        "responses": responses,
        "run_id": run_id,
    }


# ── Historical data endpoints (paginated) ────────────────────────────

@app.get("/api/runs")
async def list_runs(
    kind: str | None = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    offset = (page - 1) * per_page
    total = db.count_runs(kind)
    runs = db.list_runs(kind, limit=per_page, offset=offset)
    for r in runs:
        if r.get("config"):
            r["config"] = json.loads(r["config"])
    return {
        "runs": runs,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
    }


@app.get("/api/runs/{run_id}")
async def get_run(run_id: int):
    run = db.get_run(run_id)
    if not run:
        return {"error": "Run not found"}
    if run.get("config"):
        run["config"] = json.loads(run["config"])
    scores = db.get_eval_scores(run_id) if run["kind"] == "eval" else []
    return {"run": run, "scores": scores}


@app.get("/api/runs/{run_id}/samples")
async def get_run_samples(
    run_id: int,
    task: str | None = None,
    filter: str | None = Query(None, description="all, correct, incorrect"),
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
):
    correct_only = None
    if filter == "correct":
        correct_only = True
    elif filter == "incorrect":
        correct_only = False

    offset = (page - 1) * per_page
    total = db.count_eval_samples(run_id, task, correct_only)
    samples = db.get_eval_samples(run_id, task, limit=per_page, offset=offset,
                                  correct_only=correct_only)

    for s in samples:
        if s.get("choices"):
            s["choices"] = json.loads(s["choices"])
        if s.get("log_probs"):
            s["log_probs"] = json.loads(s["log_probs"])

    return {
        "samples": samples,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
    }


@app.get("/api/runs/{run_id}/perf")
async def get_run_perf(run_id: int):
    metrics = db.get_perf_metrics_for_run(run_id)
    return {"metrics": metrics}


@app.get("/api/runs/{run_id}/guardrails")
async def get_run_guardrails(run_id: int):
    checks = db.get_guardrail_checks(run_id)
    for c in checks:
        if c.get("details"):
            c["details"] = json.loads(c["details"])
    return {"checks": checks}


@app.get("/api/history/scores")
async def history_scores():
    all_scores = db.get_all_eval_scores()
    for s in all_scores:
        if s.get("config"):
            s["config"] = json.loads(s["config"])
    return {"scores": all_scores}


# ── Part E: Benchmark Improvement ────────────────────────────────────

@app.post("/api/improve/prepare")
async def improve_prepare():
    """Download HellaSwag data and build few-shot index."""
    return await run_script(
        [sys.executable, str(ROOT / "improve" / "prepare_data.py")], "improve-prepare"
    )


@app.post("/api/improve/run")
async def improve_run(body: dict | None = None):
    body = body or {}
    config = body.get("config", "baseline")
    limit = body.get("limit", 20)

    run_id = db.create_run("improve", MODEL, {"config": config, "limit": limit})

    cmd = [
        sys.executable, str(ROOT / "improve" / "infer.py"),
        "--config", config,
        "--limit", str(limit),
    ]
    result = await run_script(cmd, f"improve-{config}")
    db.finish_run(run_id, result["exit_code"])

    if result["exit_code"] == 0:
        _persist_improve(run_id, config)

    return {**result, "run_id": run_id}


@app.post("/api/improve/run-all")
async def improve_run_all(body: dict | None = None):
    body = body or {}
    limit = body.get("limit", 20)
    configs = body.get("configs", ["baseline", "template_only", "fewshot_3", "full"])

    run_id = db.create_run("improve", MODEL, {"configs": configs, "limit": limit})

    cmd = [
        sys.executable, str(ROOT / "improve" / "infer.py"),
        "--all", "--limit", str(limit),
    ]
    result = await run_script(cmd, "improve-all")
    db.finish_run(run_id, result["exit_code"])

    if result["exit_code"] == 0:
        for cfg_name in configs:
            _persist_improve(run_id, cfg_name)

    return {**result, "run_id": run_id}


def _persist_improve(run_id: int, config_name: str) -> None:
    result_path = ROOT / "improve" / "results" / f"{config_name}.json"
    if not result_path.exists():
        return
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stats = data.get("stats", {})
    db.save_eval_scores(run_id, {
        f"hellaswag_{config_name}": {
            "acc,none": stats.get("accuracy", 0),
            "ci_95_low": stats.get("ci_95_low", 0),
            "ci_95_high": stats.get("ci_95_high", 0),
            "n": stats.get("n", 0),
            "elapsed_s": stats.get("elapsed_s", 0),
        }
    })


@app.get("/api/improve/results")
async def improve_results():
    results_dir = ROOT / "improve" / "results"
    if not results_dir.exists():
        return {"results": None, "message": "No improvement results yet."}

    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return {"results": json.load(f)}

    all_results = {}
    for fp in sorted(results_dir.glob("*.json")):
        if fp.name == "summary.json":
            continue
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = fp.stem
        all_results[name] = data.get("stats", data)

    return {"results": all_results if all_results else None}


@app.get("/api/improve/predictions/{config}")
async def improve_predictions(config: str, page: int = Query(1, ge=1), per_page: int = Query(25, ge=1, le=100)):
    result_path = ROOT / "improve" / "results" / f"{config}.json"
    if not result_path.exists():
        return {"predictions": None}
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds = data.get("predictions", [])
    offset = (page - 1) * per_page
    total = len(preds)
    page_data = preds[offset:offset + per_page]
    return {
        "predictions": page_data,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
        "stats": data.get("stats", {}),
    }


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000)
