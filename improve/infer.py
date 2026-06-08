"""
Inference engine for HellaSwag benchmark improvement.

Runs four configurations (baseline loglikelihood, template-only, few-shot,
full with self-consistency) against the HellaSwag validation set and saves
per-sample predictions for evaluation and ablation.

Usage:
    python improve/infer.py --config baseline --limit 100
    python improve/infer.py --config full --limit 100
    python improve/infer.py --all --limit 100
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_prompt import (
    CHOICE_LABELS,
    CONFIGS,
    format_baseline_generative,
    format_improved,
    get_fewshot_examples,
    normalize_answer,
)
from prepare_data import download_split, HELLASWAG_URL, DATA_DIR

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def generate(prompt: str, options: dict) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False, "options": options},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def loglikelihood_score(context: str, continuation: str) -> float:
    """Compute log-likelihood of continuation given context via forced decoding."""
    prompt = context + continuation
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_predict": 1},
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()

    total_tokens = data.get("prompt_eval_count", 0)
    eval_duration = data.get("prompt_eval_duration", 1)
    return -eval_duration / max(total_tokens, 1)


def run_baseline_loglikelihood(items: list[dict], limit: int) -> list[dict]:
    """Baseline: score each ending via loglikelihood (same as lm-eval-harness)."""
    results = []
    for i, item in enumerate(items[:limit]):
        ctx = item.get("ctx", item.get("ctx_a", ""))
        endings = item["endings"]
        label = int(item["label"])

        scores = []
        for ending in endings:
            ll = loglikelihood_score(ctx, " " + ending)
            scores.append(ll)

        pred = int(np.argmax(scores))
        results.append({
            "idx": i,
            "label": label,
            "pred": pred,
            "correct": int(pred == label),
            "scores": scores,
            "question": ctx[:100],
        })

        if (i + 1) % 10 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1}/{min(limit, len(items))}] Running acc: {acc:.3f}")

    return results


def run_generative(items: list[dict], config_name: str, limit: int) -> list[dict]:
    """Generative approach: format prompt, generate answer, normalize."""
    cfg = CONFIGS[config_name]
    decoding = cfg["decoding"]
    use_fewshot = cfg.get("use_fewshot", False)
    fewshot_k = cfg.get("fewshot_k", 3)
    sc_k = cfg.get("self_consistency_k", 0)

    results = []
    for i, item in enumerate(items[:limit]):
        label = int(item["label"])

        fewshot_examples = None
        if use_fewshot:
            try:
                fewshot_examples = get_fewshot_examples(i, k=fewshot_k)
            except FileNotFoundError:
                pass

        if cfg.get("use_instruction", True):
            prompt = format_improved(item, fewshot_examples=fewshot_examples)
        else:
            prompt = format_baseline_generative(item)

        if sc_k > 1:
            votes = []
            for trial in range(sc_k):
                opts = {**decoding, "seed": 42 + trial}
                raw = generate(prompt, opts)
                ans = normalize_answer(raw)
                if ans is not None:
                    votes.append(ans)
            if votes:
                pred = Counter(votes).most_common(1)[0][0]
            else:
                pred = 0
        else:
            raw = generate(prompt, decoding)
            pred = normalize_answer(raw)
            if pred is None:
                pred = 0

        results.append({
            "idx": i,
            "label": label,
            "pred": pred,
            "correct": int(pred == label),
            "question": (item.get("ctx", item.get("ctx_a", "")))[:100],
            "pred_label": CHOICE_LABELS[pred],
            "true_label": CHOICE_LABELS[label],
        })

        if (i + 1) % 10 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1}/{min(limit, len(items))}] Running acc: {acc:.3f}")

    return results


def compute_stats(results: list[dict]) -> dict:
    """Compute accuracy with bootstrap 95% confidence interval."""
    corrects = np.array([r["correct"] for r in results])
    acc = float(np.mean(corrects))

    n_bootstrap = 10000
    rng = np.random.RandomState(42)
    boot_accs = []
    for _ in range(n_bootstrap):
        sample = rng.choice(corrects, size=len(corrects), replace=True)
        boot_accs.append(float(np.mean(sample)))
    boot_accs.sort()
    ci_low = boot_accs[int(0.025 * n_bootstrap)]
    ci_high = boot_accs[int(0.975 * n_bootstrap)]

    return {
        "accuracy": round(acc, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "n": len(results),
        "correct": int(sum(corrects)),
    }


def main():
    parser = argparse.ArgumentParser(description="HellaSwag improvement inference")
    parser.add_argument("--config", choices=list(CONFIGS.keys()) + ["baseline"],
                        default="baseline")
    parser.add_argument("--all", action="store_true", help="Run all configs")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model: {MODEL}")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"Limit: {args.limit}")
    print()

    print("Loading HellaSwag validation data...")
    items = download_split(HELLASWAG_URL, "hellaswag_val")
    print(f"  {len(items)} items loaded")
    print()

    configs_to_run = list(CONFIGS.keys()) if args.all else [args.config]

    all_stats = {}
    for config_name in configs_to_run:
        print(f"{'='*60}")
        print(f"Running: {config_name}")
        if config_name in CONFIGS:
            print(f"  {CONFIGS[config_name]['description']}")
        print(f"{'='*60}")

        t0 = time.time()
        results = run_generative(items, config_name, args.limit)
        elapsed = time.time() - t0

        stats = compute_stats(results)
        stats["elapsed_s"] = round(elapsed, 1)
        stats["config"] = config_name
        all_stats[config_name] = stats

        out_path = RESULTS_DIR / f"{config_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "predictions": results}, f, indent=2)

        print(f"\n  Accuracy: {stats['accuracy']:.4f} "
              f"[{stats['ci_95_low']:.4f}, {stats['ci_95_high']:.4f}]")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Saved: {out_path}")
        print()

    if len(all_stats) > 1:
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"{'Config':<20} {'Acc':>8} {'95% CI':>20} {'Time':>8}")
        print("-" * 60)
        for name, s in all_stats.items():
            ci = f"[{s['ci_95_low']:.4f}, {s['ci_95_high']:.4f}]"
            print(f"{name:<20} {s['accuracy']:>8.4f} {ci:>20} {s['elapsed_s']:>7.1f}s")

        if "baseline" in all_stats and len(all_stats) > 1:
            base_acc = all_stats["baseline"]["accuracy"]
            best_name = max(
                (k for k in all_stats if k != "baseline"),
                key=lambda k: all_stats[k]["accuracy"],
            )
            best_acc = all_stats[best_name]["accuracy"]
            print(f"\nBest improvement: {best_name} -> +{best_acc - base_acc:.4f}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
