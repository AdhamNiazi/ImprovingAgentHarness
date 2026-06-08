#!/usr/bin/env bash
# ============================================================
# HellaSwag Benchmark Improvement — Full Evaluation Pipeline
#
# Runs baseline + all optimized configs, computes stats, and
# produces comparison output with 95% confidence intervals.
#
# Usage:
#   bash improve/eval.sh          # full run (100 samples)
#   bash improve/eval.sh 50       # quick test (50 samples)
# ============================================================

set -e

LIMIT=${1:-100}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../.venv/Scripts/python"

# Fall back to system python if venv doesn't exist
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
fi
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python"
fi

echo "============================================"
echo "HellaSwag Improvement Evaluation"
echo "Model: qwen2.5:1.5b"
echo "Limit: $LIMIT samples"
echo "Python: $VENV_PYTHON"
echo "============================================"
echo ""

# Step 1: Prepare data (download + build few-shot index)
echo "[Step 1/4] Preparing data..."
$VENV_PYTHON "$SCRIPT_DIR/prepare_data.py"
echo ""

# Step 2: Run baseline (loglikelihood scoring)
echo "[Step 2/4] Running baseline..."
$VENV_PYTHON "$SCRIPT_DIR/infer.py" --config baseline --limit "$LIMIT"
echo ""

# Step 3: Run optimized configs
echo "[Step 3/4] Running optimized configs..."
for config in template_only fewshot_3 full; do
    echo "--- Config: $config ---"
    $VENV_PYTHON "$SCRIPT_DIR/infer.py" --config "$config" --limit "$LIMIT"
    echo ""
done

# Step 4: Summary
echo "[Step 4/4] Final comparison..."
echo ""
$VENV_PYTHON -c "
import json
from pathlib import Path

results_dir = Path('$SCRIPT_DIR/results')
configs = ['baseline', 'template_only', 'fewshot_3', 'full']
print(f'{\"Config\":<20} {\"Accuracy\":>10} {\"95% CI\":>24} {\"Time\":>8} {\"Lift\":>8}')
print('-' * 74)

base_acc = None
for name in configs:
    path = results_dir / f'{name}.json'
    if not path.exists():
        print(f'{name:<20} (not run)')
        continue
    data = json.load(open(path))['stats']
    acc = data['accuracy']
    ci = f'[{data[\"ci_95_low\"]:.4f}, {data[\"ci_95_high\"]:.4f}]'
    t = f'{data[\"elapsed_s\"]:.0f}s'
    if base_acc is None:
        base_acc = acc
        lift = '-'
    else:
        lift = f'+{(acc - base_acc):.4f}'
    print(f'{name:<20} {acc:>10.4f} {ci:>24} {t:>8} {lift:>8}')
"

echo ""
echo "Results saved to: $SCRIPT_DIR/results/"
echo "Done."
