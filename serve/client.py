"""
Sample client demonstrating prompt generations against the ollama endpoint.
"""

import time
import requests

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen2.5:1.5b"

PROMPTS = [
    {
        "tag": "Factual",
        "prompt": "What is the capital of France? Answer in one sentence.",
    },
    {
        "tag": "Reasoning",
        "prompt": "If a train travels 60 km/h for 2.5 hours, how far does it go? Show your work.",
    },
    {
        "tag": "Code",
        "prompt": "Write a Python function that checks whether a string is a palindrome.",
    },
    {
        "tag": "Creative",
        "prompt": "Write a haiku about programming.",
    },
    {
        "tag": "Extraction",
        "prompt": 'Extract the name and age from this sentence: "Alice is 30 years old." Return JSON.',
    },
]


def generate(prompt: str, max_tokens: int = 256) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", ""), elapsed


def main():
    print(f"Running {len(PROMPTS)} sample generations against {MODEL}\n")
    print("=" * 70)

    for item in PROMPTS:
        tag, prompt = item["tag"], item["prompt"]
        print(f"\n[{tag}] {prompt}")
        print("-" * 70)

        try:
            response, elapsed = generate(prompt)
            print(response.strip())
            print(f"\n  Time: {elapsed:.2f}s")
        except requests.ConnectionError:
            print("  ERROR: Cannot connect to ollama. Run `python serve/serve.py` first.")
            return

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
