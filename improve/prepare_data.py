"""
Prepare HellaSwag data for inference-time optimization.

Downloads the validation split, computes sentence embeddings for few-shot
retrieval, and builds a FAISS/numpy index for fast nearest-neighbour lookup.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
TRAIN_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl"

EMBED_CACHE = DATA_DIR / "embeddings.pkl"
INDEX_CACHE = DATA_DIR / "fewshot_index.pkl"


def download_split(url: str, name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    if path.exists():
        print(f"  {name} already cached at {path}")
    else:
        print(f"  Downloading {name}...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
        print(f"  Saved {path} ({len(r.text) // 1024} KB)")

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def make_text(item: dict) -> str:
    """Combine context + activity label into a single searchable string."""
    ctx = item.get("ctx", item.get("ctx_a", ""))
    activity = item.get("activity_label", "")
    return f"{activity}: {ctx}".strip()


def compute_embeddings(texts: list[str]) -> np.ndarray:
    """Compute lightweight TF-IDF style embeddings using word hashing.

    We avoid heavy dependencies (sentence-transformers) by using a simple
    but effective character n-gram hashing approach. This is fast and works
    well enough for nearest-neighbour retrieval on HellaSwag.
    """
    DIM = 4096
    rng = np.random.RandomState(42)

    embeddings = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        words = text.lower().split()
        for w in words:
            for n in range(2, 5):
                for j in range(len(w) - n + 1):
                    ngram = w[j:j + n]
                    h = hash(ngram) % DIM
                    embeddings[i, h] += 1.0
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] /= norm
    return embeddings


def build_fewshot_index(train_items: list[dict], val_items: list[dict]) -> dict:
    """Build an index mapping each val item to its k-nearest train neighbours."""
    print("  Computing train embeddings...")
    train_texts = [make_text(it) for it in train_items]
    train_embs = compute_embeddings(train_texts)

    print("  Computing val embeddings...")
    val_texts = [make_text(it) for it in val_items]
    val_embs = compute_embeddings(val_texts)

    print("  Finding nearest neighbours...")
    K = 5
    index = {}
    batch_size = 200
    for start in range(0, len(val_embs), batch_size):
        end = min(start + batch_size, len(val_embs))
        sims = val_embs[start:end] @ train_embs.T
        for i, row_idx in enumerate(range(start, end)):
            top_k = np.argsort(sims[i])[-K:][::-1]
            index[row_idx] = top_k.tolist()
        if end % 1000 < batch_size:
            print(f"    {end}/{len(val_embs)}")

    return {
        "train_items": train_items,
        "val_items": val_items,
        "neighbours": index,
        "k": K,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Downloading HellaSwag splits...")
    val_items = download_split(HELLASWAG_URL, "hellaswag_val")
    train_items = download_split(TRAIN_URL, "hellaswag_train")
    print(f"  Val: {len(val_items)} items, Train: {len(train_items)} items")

    print("[2/4] Building few-shot retrieval index...")
    if INDEX_CACHE.exists():
        print(f"  Index already cached at {INDEX_CACHE}")
    else:
        index = build_fewshot_index(train_items, val_items)
        with open(INDEX_CACHE, "wb") as f:
            pickle.dump(index, f)
        print(f"  Saved index to {INDEX_CACHE}")

    print("[3/4] Verifying data integrity...")
    sample = val_items[0]
    assert "ctx" in sample or "ctx_a" in sample, "Missing context field"
    assert "endings" in sample, "Missing endings field"
    assert "label" in sample, "Missing label field"
    print(f"  Sample: {make_text(sample)[:80]}...")
    print(f"  Choices: {len(sample['endings'])}")
    print(f"  Label: {sample['label']}")

    print("[4/4] Done. Data ready for optimization.")


if __name__ == "__main__":
    main()
