#!/usr/bin/env bash
# Example workflow: split -> train -> render test -> metrics.
# Usage:
#   chmod +x scripts/train_render_metrics_test.sh
#   ./scripts/train_render_metrics_test.sh [SOURCE_DIR] [MODEL_DIR]
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE="${1:-data}"
MODEL="${2:-data/output}"

echo "=== [1/4] Write sparse/0/test.txt (skip if already set) ==="
python scripts/split_colmap_train_test.py -s "$SOURCE" --strategy random --test_ratio 0.1 --seed 0

echo ""
echo "=== [2/4] Train (run manually; must use --eval --llffhold 0) ==="
echo "example:"
echo "  python train_optimized.py -s \"$SOURCE\" -m \"$MODEL\" --eval --llffhold 0 --iterations 30000"
read -r -p "Press Enter after training finishes..."

echo "=== [3/4] Render test views only ==="
python render.py -m "$MODEL" --skip_train --quiet

echo "=== [4/4] Metrics (test split) ==="
python metrics.py -m "$MODEL" --split test

echo "Done. See $MODEL/results.json"
