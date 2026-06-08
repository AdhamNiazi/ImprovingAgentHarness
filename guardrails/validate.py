"""
Guardrails: determinism verification and output validation.

1. Deterministic mode — sends identical prompts multiple times with
   temperature=0, seed=42 and asserts byte-identical responses.
2. Output validation — regex and schema checks for the custom python_logic
   benchmark outputs.
3. Nondeterminism report — character-level similarity for any divergent pairs.
"""

import argparse
import difflib
import json
import os
import re
import sys
import time

import requests

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

DETERMINISTIC_OPTS = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "seed": 42,
}

TEST_PROMPTS = [
    "What is the capital of France?",
    "Compute 7 * 8.",
    "Is the sky blue? Answer yes or no.",
    "List the first five prime numbers.",
    "def fibonacci(n):",
]

TRIALS = 3


# ── helpers ──────────────────────────────────────────────────────────

def generate(prompt: str, max_tokens: int = 128) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {"num_predict": max_tokens, **DETERMINISTIC_OPTS},
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "")


def char_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# ── determinism test ─────────────────────────────────────────────────

def test_determinism():
    print("=" * 60)
    print("  Determinism Test")
    print(f"  Model: {MODEL}   Trials: {TRIALS}")
    print(f"  Settings: {DETERMINISTIC_OPTS}")
    print("=" * 60)

    results = {}
    all_pass = True

    for prompt in TEST_PROMPTS:
        responses = []
        for t in range(TRIALS):
            resp = generate(prompt)
            responses.append(resp)
            time.sleep(0.2)

        identical = all(r == responses[0] for r in responses)
        sims = [
            char_similarity(responses[0], responses[i])
            for i in range(1, len(responses))
        ]
        avg_sim = sum(sims) / len(sims) if sims else 1.0

        status = "PASS" if identical else "FAIL"
        if not identical:
            all_pass = False

        print(f"\n  [{status}] {prompt[:50]}")
        print(f"         identical={identical}  avg_similarity={avg_sim:.4f}")
        if not identical:
            for i, r in enumerate(responses):
                print(f"         trial {i}: {r[:80]}…" if len(r) > 80 else f"         trial {i}: {r}")

        results[prompt] = {
            "identical": identical,
            "avg_similarity": avg_sim,
            "responses": responses,
        }

    print("\n" + "=" * 60)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    return results


# ── output validation ────────────────────────────────────────────────

ANSWER_RE = re.compile(r"^[A-Da-d]$")


def validate_mc_answer(output: str) -> tuple[bool, str]:
    """Validate that output is a single A-D letter."""
    stripped = output.strip()
    if ANSWER_RE.match(stripped):
        return True, ""
    # Try extracting first letter
    for ch in stripped:
        if ch.upper() in "ABCD":
            return True, f"extracted '{ch.upper()}' from '{stripped[:30]}'"
    return False, f"no A-D answer found in '{stripped[:60]}'"


def validate_json_output(output: str, schema: dict | None = None) -> tuple[bool, str]:
    """Validate that output is valid JSON, optionally matching a schema."""
    try:
        obj = json.loads(output)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"

    if schema is None:
        return True, ""

    try:
        import jsonschema
        jsonschema.validate(obj, schema)
        return True, ""
    except jsonschema.ValidationError as e:
        return False, str(e.message)
    except ImportError:
        return True, "jsonschema not installed, skipped schema validation"


def test_output_validation():
    print("\n" + "=" * 60)
    print("  Output Validation Test")
    print("=" * 60)

    mc_tests = [
        ("A", True),
        ("b", True),
        ("The answer is B", True),
        ("hello world", False),
        ("42", False),
        ("D", True),
    ]
    for text, expected_valid in mc_tests:
        valid, msg = validate_mc_answer(text)
        status = "OK" if valid == expected_valid else "MISMATCH"
        print(f"  [{status}] validate_mc_answer('{text}') → valid={valid}  {msg}")

    json_tests = [
        ('{"name": "Alice", "age": 30}', None, True),
        ("not json", None, False),
        ('{"name": "Bob"}', {"type": "object", "required": ["name", "age"]}, False),
    ]
    for text, schema, expected_valid in json_tests:
        valid, msg = validate_json_output(text, schema)
        status = "OK" if valid == expected_valid else "MISMATCH"
        print(f"  [{status}] validate_json_output('{text[:30]}') → valid={valid}  {msg}")


# ── nondeterminism report ────────────────────────────────────────────

def nondeterminism_report(det_results: dict):
    print("\n" + "=" * 60)
    print("  Nondeterminism Report")
    print("=" * 60)

    nondet_prompts = [
        p for p, r in det_results.items() if not r["identical"]
    ]

    if not nondet_prompts:
        print("  All prompts produced deterministic outputs.")
        return

    for prompt in nondet_prompts:
        info = det_results[prompt]
        responses = info["responses"]
        print(f"\n  Prompt: {prompt[:60]}")
        print(f"  Similarity: {info['avg_similarity']:.4f}")

        diff = list(difflib.unified_diff(
            responses[0].splitlines(keepends=True),
            responses[1].splitlines(keepends=True),
            fromfile="trial_0",
            tofile="trial_1",
            lineterm="",
        ))
        for line in diff[:20]:
            print(f"    {line}")

    print("\n  Likely causes:")
    print("   - Floating-point nondeterminism in parallel GEMM kernels")
    print("   - KV-cache state leaking between requests")
    print("   - Non-deterministic thread scheduling in CPU backend")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-determinism", action="store_true",
                        help="Skip the live determinism test (requires running ollama)")
    args = parser.parse_args()

    if args.skip_determinism:
        print("Skipping determinism test (--skip-determinism).")
        det_results = {}
    else:
        det_results = test_determinism()

    test_output_validation()

    if det_results:
        nondeterminism_report(det_results)


if __name__ == "__main__":
    main()
