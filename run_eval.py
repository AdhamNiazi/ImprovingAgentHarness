"""
Evaluation runner.

Runs MMLU (abstract_algebra subset), HellaSwag, and the custom python_logic
benchmark via lm-evaluation-harness, using the OllamaLM wrapper with
SQLite-backed caching.  Results are saved to eval_runner/results/.
"""

import json
import os
import sys
import argparse
import datetime

import lm_eval
from lm_eval.tasks import TaskManager

# Ensure our custom model is registered before lm_eval uses the registry
sys.path.insert(0, os.path.dirname(__file__))
import model as _model_reg  # noqa: F401  — triggers @register_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CACHE_DB = os.path.join(SCRIPT_DIR, ".cache.db")
CUSTOM_TASK_DIR = os.path.join(SCRIPT_DIR, "custom_task")

DEFAULT_TASKS = [
    "mmlu_abstract_algebra",
    "hellaswag",
    "python_logic",
]


def build_model(model_name: str, base_url: str):
    return _model_reg.OllamaLM(model=model_name, base_url=base_url)


def write_summary(all_results: dict, out_dir: str):
    lines = [
        "# Evaluation Summary",
        "",
        f"Date: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        "",
        "| Task | Metric | Value |",
        "|------|--------|-------|",
    ]
    for task_name, task_data in sorted(all_results.items()):
        metrics = task_data if isinstance(task_data, dict) else {}
        for metric, val in sorted(metrics.items()):
            if isinstance(val, (int, float)):
                lines.append(f"| {task_name} | {metric} | {val:.4f} |")
    lines.append("")

    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSummary written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluations")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Tasks to evaluate (default: mmlu_abstract_algebra hellaswag python_logic)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples per task (None = full dataset)")
    args = parser.parse_args()

    tasks = args.tasks or DEFAULT_TASKS
    limit = args.limit if args.limit and args.limit > 0 else None

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Model:  {args.model}")
    print(f"URL:    {args.base_url}")
    print(f"Tasks:  {tasks}")
    print(f"Limit:  {limit or 'full dataset'}")
    print(f"Cache:  {CACHE_DB}\n")

    lm = build_model(args.model, args.base_url)

    task_manager = TaskManager(include_path=CUSTOM_TASK_DIR)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        task_manager=task_manager,
        log_samples=True,
        use_cache=CACHE_DB,
    )

    # Persist full results JSON
    results_path = os.path.join(RESULTS_DIR, "results.json")
    serialisable = results.get("results", results)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, default=str)
    print(f"Full results -> {results_path}")

    # Print + write summary table
    task_results = results.get("results", {})
    write_summary(task_results, RESULTS_DIR)

    # Also dump per-sample logs if available
    samples = results.get("samples", {})
    if samples:
        samples_path = os.path.join(RESULTS_DIR, "samples.json")
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, default=str)


if __name__ == "__main__":
    main()
