"""
SQLite persistence layer for all pipeline run metrics.

Tables:
  runs           – one row per execution (eval, perf, guardrails)
  eval_scores    – aggregate scores per task per run
  eval_samples   – individual question-level results per run
  perf_metrics   – one row per config combination per perf run
  guardrail_checks – one row per determinism/validation check per run
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "pipeline.db"

_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db() -> None:
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS runs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        kind        TEXT NOT NULL,   -- 'eval', 'perf', 'guardrails'
        model       TEXT NOT NULL,
        started_at  TEXT NOT NULL,
        finished_at TEXT,
        exit_code   INTEGER,
        config      TEXT             -- JSON blob of run params
    );

    CREATE TABLE IF NOT EXISTS eval_scores (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id  INTEGER NOT NULL REFERENCES runs(id),
        task    TEXT NOT NULL,
        metric  TEXT NOT NULL,
        value   REAL NOT NULL,
        stderr  REAL
    );

    CREATE TABLE IF NOT EXISTS eval_samples (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id       INTEGER NOT NULL REFERENCES runs(id),
        task         TEXT NOT NULL,
        doc_id       INTEGER,
        question     TEXT,
        choices      TEXT,        -- JSON array
        correct_idx  INTEGER,
        model_idx    INTEGER,
        correct      INTEGER,     -- 1 or 0
        acc          REAL,
        acc_norm     REAL,
        log_probs    TEXT         -- JSON array of per-choice logprobs
    );

    CREATE TABLE IF NOT EXISTS perf_metrics (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      INTEGER NOT NULL REFERENCES runs(id),
        concurrency INTEGER,
        prompt_type TEXT,
        cache       TEXT,
        stop_seq    TEXT,
        ttft_ms     REAL,
        tps         REAL,
        latency_p50 REAL,
        latency_p95 REAL,
        latency_p99 REAL,
        gpu_util    TEXT
    );

    CREATE TABLE IF NOT EXISTS guardrail_checks (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      INTEGER NOT NULL REFERENCES runs(id),
        check_type  TEXT NOT NULL,  -- 'determinism' or 'validation'
        prompt      TEXT,
        passed      INTEGER,       -- 1 or 0
        details     TEXT           -- JSON blob
    );

    CREATE INDEX IF NOT EXISTS idx_eval_scores_run ON eval_scores(run_id);
    CREATE INDEX IF NOT EXISTS idx_eval_samples_run ON eval_samples(run_id);
    CREATE INDEX IF NOT EXISTS idx_perf_metrics_run ON perf_metrics(run_id);
    CREATE INDEX IF NOT EXISTS idx_guardrail_checks_run ON guardrail_checks(run_id);
    """)
    c.commit()


# ── Run lifecycle ────────────────────────────────────────────────────

def create_run(kind: str, model: str, config: dict | None = None) -> int:
    c = _conn()
    cur = c.execute(
        "INSERT INTO runs (kind, model, started_at, config) VALUES (?, ?, ?, ?)",
        (kind, model, _now(), json.dumps(config) if config else None),
    )
    c.commit()
    return cur.lastrowid  # type: ignore[return-value]


def finish_run(run_id: int, exit_code: int) -> None:
    c = _conn()
    c.execute(
        "UPDATE runs SET finished_at = ?, exit_code = ? WHERE id = ?",
        (_now(), exit_code, run_id),
    )
    c.commit()


# ── Eval persistence ─────────────────────────────────────────────────

def save_eval_scores(run_id: int, results: dict) -> None:
    """Save aggregate task scores. `results` is {task: {metric: value}}."""
    c = _conn()
    rows = []
    for task, metrics in results.items():
        for key, val in metrics.items():
            if key == "alias" or not isinstance(val, (int, float)):
                continue
            if "stderr" in key:
                continue
            stderr_key = key.replace(",none", "_stderr,none")
            stderr = metrics.get(stderr_key)
            rows.append((run_id, task, key.replace(",none", ""), float(val),
                         float(stderr) if stderr is not None else None))
    c.executemany(
        "INSERT INTO eval_scores (run_id, task, metric, value, stderr) VALUES (?,?,?,?,?)",
        rows,
    )
    c.commit()


def save_eval_samples(run_id: int, samples: dict) -> None:
    """Save per-question results. `samples` is {task: [sample, ...]}."""
    c = _conn()
    rows = []
    for task, items in samples.items():
        for s in items:
            doc = s.get("doc", {})
            choices = doc.get("choices", [])
            correct_idx = s.get("target")
            if isinstance(correct_idx, str):
                try:
                    correct_idx = int(correct_idx)
                except ValueError:
                    correct_idx = None

            resps = s.get("filtered_resps", [])
            log_probs = []
            for r in resps:
                if isinstance(r, (list, tuple)) and len(r) > 0:
                    log_probs.append(r[0] if isinstance(r[0], (int, float)) else None)
                else:
                    log_probs.append(None)

            model_idx = None
            if log_probs:
                valid = [(i, lp) for i, lp in enumerate(log_probs) if lp is not None]
                if valid:
                    model_idx = max(valid, key=lambda x: x[1])[0]

            correct = 1 if (model_idx is not None and model_idx == correct_idx) else 0

            rows.append((
                run_id, task, s.get("doc_id"),
                doc.get("question", ""),
                json.dumps(choices),
                correct_idx, model_idx, correct,
                s.get("acc"), s.get("acc_norm"),
                json.dumps(log_probs),
            ))
    c.executemany(
        """INSERT INTO eval_samples
           (run_id, task, doc_id, question, choices, correct_idx, model_idx,
            correct, acc, acc_norm, log_probs)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    c.commit()


# ── Perf persistence ─────────────────────────────────────────────────

def save_perf_metrics(run_id: int, rows_data: list[dict]) -> None:
    c = _conn()
    rows = []
    for r in rows_data:
        rows.append((
            run_id,
            _int_or_none(r.get("concurrency")),
            r.get("prompt_type", ""),
            r.get("cache", ""),
            r.get("stop_seq", ""),
            _float_or_none(r.get("ttft_ms")),
            _float_or_none(r.get("tpot") or r.get("tps")),
            _float_or_none(r.get("latency_p50")),
            _float_or_none(r.get("latency_p95")),
            _float_or_none(r.get("latency_p99")),
            r.get("gpu_util", ""),
        ))
    c.executemany(
        """INSERT INTO perf_metrics
           (run_id, concurrency, prompt_type, cache, stop_seq,
            ttft_ms, tps, latency_p50, latency_p95, latency_p99, gpu_util)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    c.commit()


# ── Guardrails persistence ───────────────────────────────────────────

def save_guardrail_check(run_id: int, check_type: str, prompt: str,
                         passed: bool, details: dict | None = None) -> None:
    c = _conn()
    c.execute(
        """INSERT INTO guardrail_checks (run_id, check_type, prompt, passed, details)
           VALUES (?,?,?,?,?)""",
        (run_id, check_type, prompt, 1 if passed else 0,
         json.dumps(details) if details else None),
    )
    c.commit()


# ── Query helpers ────────────────────────────────────────────────────

def list_runs(kind: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    c = _conn()
    if kind:
        cur = c.execute(
            "SELECT * FROM runs WHERE kind = ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (kind, limit, offset),
        )
    else:
        cur = c.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
    return [dict(r) for r in cur.fetchall()]


def count_runs(kind: str | None = None) -> int:
    c = _conn()
    if kind:
        cur = c.execute("SELECT COUNT(*) FROM runs WHERE kind = ?", (kind,))
    else:
        cur = c.execute("SELECT COUNT(*) FROM runs")
    return cur.fetchone()[0]


def get_run(run_id: int) -> dict | None:
    c = _conn()
    cur = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def get_eval_scores(run_id: int) -> list[dict]:
    c = _conn()
    cur = c.execute(
        "SELECT * FROM eval_scores WHERE run_id = ? ORDER BY task, metric",
        (run_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def get_eval_samples(run_id: int, task: str | None = None,
                     limit: int = 25, offset: int = 0,
                     correct_only: bool | None = None) -> list[dict]:
    c = _conn()
    clauses = ["run_id = ?"]
    params: list[Any] = [run_id]
    if task:
        clauses.append("task = ?")
        params.append(task)
    if correct_only is True:
        clauses.append("correct = 1")
    elif correct_only is False:
        clauses.append("correct = 0")
    where = " AND ".join(clauses)
    params.extend([limit, offset])
    cur = c.execute(
        f"SELECT * FROM eval_samples WHERE {where} ORDER BY doc_id LIMIT ? OFFSET ?",
        params,
    )
    return [dict(r) for r in cur.fetchall()]


def count_eval_samples(run_id: int, task: str | None = None,
                       correct_only: bool | None = None) -> int:
    c = _conn()
    clauses = ["run_id = ?"]
    params: list[Any] = [run_id]
    if task:
        clauses.append("task = ?")
        params.append(task)
    if correct_only is True:
        clauses.append("correct = 1")
    elif correct_only is False:
        clauses.append("correct = 0")
    where = " AND ".join(clauses)
    cur = c.execute(f"SELECT COUNT(*) FROM eval_samples WHERE {where}", params)
    return cur.fetchone()[0]


def get_perf_metrics_for_run(run_id: int) -> list[dict]:
    c = _conn()
    cur = c.execute("SELECT * FROM perf_metrics WHERE run_id = ?", (run_id,))
    return [dict(r) for r in cur.fetchall()]


def get_guardrail_checks(run_id: int) -> list[dict]:
    c = _conn()
    cur = c.execute("SELECT * FROM guardrail_checks WHERE run_id = ?", (run_id,))
    return [dict(r) for r in cur.fetchall()]


def get_all_eval_scores() -> list[dict]:
    """All scores across all runs, for trend charts."""
    c = _conn()
    cur = c.execute("""
        SELECT es.*, r.started_at, r.config, r.model
        FROM eval_scores es
        JOIN runs r ON r.id = es.run_id
        ORDER BY r.id DESC, es.task, es.metric
    """)
    return [dict(r) for r in cur.fetchall()]


# ── Utilities ────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_or_none(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _int_or_none(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None
