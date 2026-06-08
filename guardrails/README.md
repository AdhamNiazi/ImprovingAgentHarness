# Guardrails & Determinism

## What We Test

### 1. Deterministic Generation

We send five diverse prompts three times each through ollama with settings
designed to eliminate sampling randomness:

| Parameter   | Value | Rationale                                      |
|-------------|-------|-------------------------------------------------|
| temperature | 0     | Disables random sampling; uses argmax decoding   |
| top_p       | 1     | No nucleus filtering                             |
| top_k       | 1     | Only the single most-likely token is considered  |
| seed        | 42    | Fixes the RNG state in the sampler               |

For each prompt we check byte-identical output across all trials.

### 2. Output Validation

Two lightweight validators ensure model outputs conform to expected formats:

- **Multiple-choice regex** (`^[A-Da-d]$`): Checks that a model answer is a
  single letter A-D.  Falls back to extracting the first A-D character from
  longer responses.
- **JSON schema validator**: Parses the output as JSON and optionally validates
  against a `jsonschema` schema object.

### 3. Nondeterminism Detection

When any prompt produces divergent responses, we report:

- Exact-match rate across trials
- Character-level similarity (via `difflib.SequenceMatcher`)
- Unified diff of the first divergent pair

## Where Nondeterminism Persists

Even with `temperature=0, top_k=1, seed=42`, nondeterminism can still appear:

1. **Floating-point ordering** — Matrix multiplications on multi-threaded CPU
   (or GPU) can sum partial products in different orders across runs, producing
   slightly different logits that occasionally flip the argmax at low-confidence
   positions.

2. **KV-cache state** — If the ollama server reuses a KV cache from a previous
   request, the numerical state may differ depending on prior request history.
   Restarting the server between trials eliminates this.

3. **Thread scheduling** — The OS scheduler may assign different cores/threads
   on each run, changing SIMD lane assignments and reduce-order in BLAS kernels.

4. **Quantisation artefacts** — When running quantised models (e.g. Q4_K_M),
   dequantisation can be order-dependent, amplifying float rounding differences.

## How to Run

```bash
# Full test (requires ollama serving the model)
python guardrails/validate.py

# Skip the live determinism test (only runs validation logic)
python guardrails/validate.py --skip-determinism
```
