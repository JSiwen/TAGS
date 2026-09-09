#!/usr/bin/env bash
# A100 full-resolution transfer of the current best recipe: v15 E1b densify7k.
#
# 3080 winner (images_4 @ 800, not comparable in absolute PSNR):
#   output/v15_30k_fix_e1b_densify7k/bicycle_v15_3080_e1b_densify7k
#   ours_28500  PSNR 26.6930 / SSIM 0.828843 / LPIPS 0.149942
#
# This script keeps E1b's densify_until=7200 + late_stabilize losses,
# but switches the protocol to official full-res:
#   IMAGES=images  RESOLUTION=-1  30k  --eval --llffhold 8
#
# Usage (repo root on A100):
#   source scripts/configs/bicycle_v15_e1b_a100.env
#   bash scripts/run_bicycle_v15_e1b_a100.sh
#
# Minimal override:
#   SOURCE=/path/to/bicycle DEPTHS=/path/to/bicycle/depth_any_out \
#     bash scripts/run_bicycle_v15_e1b_a100.sh
#
# Skip training, only render/metrics:
#   RUN_TRAIN=0 RUN_RENDER_METRICS=1 bash scripts/run_bicycle_v15_e1b_a100.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/scripts/configs/bicycle_v15_e1b_a100.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/configs/bicycle_v15_e1b_a100.env"
fi

export EXPERIMENT="${EXPERIMENT:-E1b}"
export ITERATIONS="${ITERATIONS:-30000}"
export RESOLUTION="${RESOLUTION:--1}"
export IMAGES="${IMAGES:-images}"
export SOURCE="${SOURCE:-/root/highspeedstorage/TAGS-A100/data/bicycle}"
export DEPTHS="${DEPTHS:-/root/highspeedstorage/TAGS-A100/data/bicycle/depth_any_out}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/highspeedstorage/TAGS-A100/TAGSs/output/v15_e1b_a100}"
export CASE_NAME="${CASE_NAME:-bicycle_v15_a100_e1b_densify7k}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export RUN_RENDER_METRICS="${RUN_RENDER_METRICS:-1}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
export GPU_ID="${GPU_ID:-0}"
export TAGGS_CPU_KNN_INIT="${TAGGS_CPU_KNN_INIT:-1}"

echo "=== v15 E1b A100 full-res ==="
echo "EXPERIMENT:   ${EXPERIMENT}"
echo "CASE_NAME:    ${CASE_NAME}"
echo "PROTOCOL:     ${IMAGES} @ resolution=${RESOLUTION}"
echo "SOURCE:       ${SOURCE}"
echo "DEPTHS:       ${DEPTHS}"
echo "OUTPUT_ROOT:  ${OUTPUT_ROOT}"
echo "GPU_ID:       ${GPU_ID}"
echo ""
echo "Note: absolute PSNR will differ from the 3080 images_4@800 run (26.693)."
echo "      Compare against A100 full-res v4_restore / v14, not the 3080 number."
echo ""

bash "$ROOT/scripts/run_bicycle_v15_30k_densify_fix.sh"
