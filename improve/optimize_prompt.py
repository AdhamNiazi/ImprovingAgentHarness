"""
Prompt optimization for HellaSwag benchmark improvement.

Implements three optimization levels:
  Level 0 (baseline): Raw lm-eval-harness default prompt
  Level 1: Instruction-tuned template + answer normalization
  Level 2: + Semantic few-shot examples (3-shot)
  Level 3: + Self-consistency (k-sample majority vote)

Each level is independently toggleable for ablation study.
"""

import json
import pickle
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_CACHE = DATA_DIR / "fewshot_index.pkl"

CHOICE_LABELS = ["A", "B", "C", "D"]

# ── Prompt templates ─────────────────────────────────────────────────

BASELINE_TEMPLATE = """{ctx} {ending}"""

IMPROVED_INSTRUCTION = (
    "Read the context below and choose the most natural and logical continuation.\n\n"
)

IMPROVED_TEMPLATE = """{instruction}Context: {ctx}

Which ending best continues the text?
A) {ending_a}
B) {ending_b}
C) {ending_c}
D) {ending_d}

Answer:"""

FEWSHOT_EXAMPLE_TEMPLATE = """Context: {ctx}

Which ending best continues the text?
A) {ending_a}
B) {ending_b}
C) {ending_c}
D) {ending_d}

Answer: {answer}
"""


# ── Template builders ────────────────────────────────────────────────

BASELINE_GEN_TEMPLATE = """{ctx}

A) {ending_a}
B) {ending_b}
C) {ending_c}
D) {ending_d}

Answer:"""


def format_baseline(item: dict) -> tuple[str, list[str]]:
    """Default harness-style prompt (no optimization)."""
    ctx = item.get("ctx", item.get("ctx_a", ""))
    endings = item["endings"]
    return ctx, endings


def format_baseline_generative(item: dict) -> str:
    """Minimal generative prompt with no instruction or optimization."""
    ctx = item.get("ctx", item.get("ctx_a", ""))
    endings = item["endings"]
    return BASELINE_GEN_TEMPLATE.format(
        ctx=ctx.strip(),
        ending_a=endings[0],
        ending_b=endings[1],
        ending_c=endings[2],
        ending_d=endings[3],
    )


def format_improved(item: dict, fewshot_examples: list[dict] | None = None) -> str:
    """Instruction-tuned template with optional few-shot examples."""
    ctx = item.get("ctx", item.get("ctx_a", ""))
    endings = item["endings"]

    parts = []

    if fewshot_examples:
        parts.append(IMPROVED_INSTRUCTION)
        parts.append("Here are some examples:\n\n")
        for ex in fewshot_examples:
            ex_ctx = ex.get("ctx", ex.get("ctx_a", ""))
            ex_endings = ex["endings"]
            label = int(ex["label"])
            parts.append(FEWSHOT_EXAMPLE_TEMPLATE.format(
                ctx=ex_ctx.strip(),
                ending_a=ex_endings[0],
                ending_b=ex_endings[1],
                ending_c=ex_endings[2],
                ending_d=ex_endings[3],
                answer=CHOICE_LABELS[label],
            ))
        parts.append("\nNow answer this one:\n\n")

    parts.append(IMPROVED_TEMPLATE.format(
        instruction="" if fewshot_examples else IMPROVED_INSTRUCTION,
        ctx=ctx.strip(),
        ending_a=endings[0],
        ending_b=endings[1],
        ending_c=endings[2],
        ending_d=endings[3],
    ))

    return "".join(parts)


# ── Few-shot retrieval ───────────────────────────────────────────────

_index_cache = None


def load_fewshot_index() -> dict:
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    if not INDEX_CACHE.exists():
        raise FileNotFoundError(
            f"Few-shot index not found at {INDEX_CACHE}. "
            "Run prepare_data.py first."
        )
    with open(INDEX_CACHE, "rb") as f:
        _index_cache = pickle.load(f)
    return _index_cache


def get_fewshot_examples(val_idx: int, k: int = 3) -> list[dict]:
    """Retrieve k most similar training examples for a validation item."""
    index = load_fewshot_index()
    neighbour_ids = index["neighbours"].get(val_idx, [])[:k]
    return [index["train_items"][i] for i in neighbour_ids]


# ── Answer normalization ─────────────────────────────────────────────

def normalize_answer(raw: str, num_choices: int = 4) -> int | None:
    """Extract choice index from model output. Returns 0-3 or None."""
    raw = raw.strip()

    for i, label in enumerate(CHOICE_LABELS[:num_choices]):
        if raw.upper().startswith(label):
            return i

    for i, label in enumerate(CHOICE_LABELS[:num_choices]):
        if label in raw.upper():
            return i

    return None


# ── Decoding configurations ──────────────────────────────────────────

BASELINE_DECODING = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "seed": 42,
    "num_predict": 4,
}

IMPROVED_DECODING = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "seed": 42,
    "num_predict": 4,
}

SELFCONSISTENCY_DECODING = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 4,
}

# ── Config presets ───────────────────────────────────────────────────

CONFIGS = {
    "baseline": {
        "description": "Raw lm-eval-harness default (loglikelihood scoring)",
        "use_instruction": False,
        "use_fewshot": False,
        "self_consistency_k": 0,
        "decoding": BASELINE_DECODING,
    },
    "template_only": {
        "description": "Instruction-tuned template, no few-shot",
        "use_instruction": True,
        "use_fewshot": False,
        "self_consistency_k": 0,
        "decoding": IMPROVED_DECODING,
    },
    "fewshot_3": {
        "description": "Instruction template + 3-shot semantic retrieval",
        "use_instruction": True,
        "use_fewshot": True,
        "fewshot_k": 3,
        "self_consistency_k": 0,
        "decoding": IMPROVED_DECODING,
    },
    "full": {
        "description": "Template + 3-shot + self-consistency (k=5)",
        "use_instruction": True,
        "use_fewshot": True,
        "fewshot_k": 3,
        "self_consistency_k": 5,
        "decoding": SELFCONSISTENCY_DECODING,
    },
}


if __name__ == "__main__":
    print("Available optimization configs:")
    for name, cfg in CONFIGS.items():
        print(f"  {name:20s} — {cfg['description']}")

    print("\nExample improved prompt (no few-shot):")
    sample = {
        "ctx": "A woman is outside with a bucket and a dog. The dog is running around.",
        "endings": [
            "She picks up the bucket and throws it.",
            "She starts washing the dog with the bucket.",
            "The dog grabs the bucket and runs away.",
            "A cat appears from nowhere.",
        ],
        "label": 1,
    }
    print(format_improved(sample))
