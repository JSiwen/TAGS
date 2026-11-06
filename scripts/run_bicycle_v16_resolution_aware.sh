#!/usr/bin/env bash
# v16 resolution-aware staged training on A100 full-res bicycle.
#
# Uses enable_resolution_aware_schedule to derive densify/SPA/late-loss iters
# from effective training resolution. Optional runtime adaptive densify extension.
#
# Usage (repo root):
#   source scripts/configs/bicycle_v16_resolution_aware.env
#   bash scripts/run_bicycle_v16_resolution_aware.sh
#
# Planner-only ablation:
#   VARIANT=planner_only bash scripts/run_bicycle_v16_resolution_aware.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/scripts/configs/bicycle_v16_resolution_aware.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/configs/bicycle_v16_resolution_aware.env"
fi

VARIANT="${VARIANT:-auto}"
SOURCE="${SOURCE:-/root/highspeedstorage/TAGS-A100/data/bicycle}"
DEPTHS="${DEPTHS:-/root/highspeedstorage/TAGS-A100/data/bicycle/depth_any_out}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/highspeedstorage/TAGS-A100/TAGSs/output/v16_resolution_aware_a100}"
CASE_NAME="${CASE_NAME:-bicycle_v16_a100_resolution_aware}"
ITERATIONS="${ITERATIONS:-30000}"
RESOLUTION="${RESOLUTION:--1}"
IMAGES="${IMAGES:-images}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER_METRICS="${RUN_RENDER_METRICS:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"

PSNR_DROP_THRESHOLD="${PSNR_DROP_THRESHOLD:-0.05}"
PERCEPTUAL_PSNR_BUDGET="${PERCEPTUAL_PSNR_BUDGET:-0.10}"
CHRONIC_TRIPLE_LIMIT="${CHRONIC_TRIPLE_LIMIT:-1}"

ENABLE_ADAPTIVE=1
if [[ "${VARIANT,,}" == "planner_only" ]]; then
  ENABLE_ADAPTIVE=0
fi

MODEL_DIR="$OUTPUT_ROOT/$CASE_NAME"
DONE_MARKER="$MODEL_DIR/.v16_resolution_aware_done"
LOG_PATH="$LOG_DIR/${CASE_NAME}.log"
STAGE_MD="$OUTPUT_ROOT/${CASE_NAME}_stage_selection_summary.md"
STAGE_JSON="$OUTPUT_ROOT/${CASE_NAME}_stage_selection_summary.json"
CHECKPOINT_MD="$OUTPUT_ROOT/${CASE_NAME}_checkpoint_summary.md"
CHECKPOINT_JSON="$OUTPUT_ROOT/${CASE_NAME}_checkpoint_summary.json"
PRIMARY_ITER="28500"
FINAL_METHOD="ours_${ITERATIONS}"
BASELINE_METHOD="ours_7000"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

echo ""
echo "=== v16 resolution-aware | VARIANT=${VARIANT} | ${CASE_NAME} ==="
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "MODEL_DIR:   ${MODEL_DIR}"
echo "PROTOCOL:    ${IMAGES} @ resolution=${RESOLUTION}"
echo "RAS:         enable_resolution_aware_schedule=1"
echo "ADAPTIVE:    enable_adaptive_stage_controller=${ENABLE_ADAPTIVE}"
echo "LOG:         ${LOG_PATH}"
echo ""

if [[ ! -f "$SOURCE/sparse/0/cameras.bin" ]]; then
  echo "SOURCE not found or missing COLMAP: $SOURCE" >&2
  exit 1
fi
if [[ ! -d "$DEPTHS" ]]; then
  echo "DEPTHS not found: $DEPTHS" >&2
  exit 1
fi

TRAIN_ARGS=(
  -s "$SOURCE"
  -m "$MODEL_DIR"
  --images "$IMAGES"
  --depths "$DEPTHS"
  --resolution "$RESOLUTION"
  --data_device cpu
  --eval
  --llffhold 8
  --iterations "$ITERATIONS"
  --save_iterations 28500 "$ITERATIONS"
  --no_save_schedule_point_clouds
  --densify_from_iter 500
  --densify_grad_threshold 0.00016
  --enable_geo_prior
  --geo_prior_depth
  --geo_prior_normal
  --depth_prior_weight_final 0.05
  --normal_prior_weight_final 0.02
  --normal_prior_tv_weight 0.003
  --geo_prior_start_iter 0
  --lpips_loss_weight_final "${LATE_STABILIZE_LPIPS_WEIGHT_FINAL:-0.006}"
  --lpips_loss_max_weight "${LATE_STABILIZE_LPIPS_MAX_WEIGHT:-0.010}"
  --lpips_loss_interval "${LATE_STABILIZE_LPIPS_INTERVAL:-10}"
  --psnr_loss_weight_final "${LATE_STABILIZE_PSNR_WEIGHT_FINAL:-0.008}"
  --psnr_loss_max_weight "${LATE_STABILIZE_PSNR_MAX_WEIGHT:-0.012}"
  --color_align_weight_final "${LATE_STABILIZE_COLOR_ALIGN_WEIGHT_FINAL:-0.0010}"
  --color_align_max_weight "${LATE_STABILIZE_COLOR_ALIGN_MAX_WEIGHT:-0.0020}"
  --color_align_std_weight 0.25
  --enable_resolution_adaptive
  --enable_resolution_aware_schedule
  --enable_group_lr_schedule
  --lr_geo_mult_stage1 1.10
  --lr_geo_mult_stage2 1.00
  --lr_geo_mult_stage3 "${LATE_STABILIZE_LR_GEO_MULT_STAGE3:-0.40}"
  --lr_color_mult_stage1 0.75
  --lr_color_mult_stage2 1.05
  --lr_color_mult_stage3 "${LATE_STABILIZE_LR_COLOR_MULT_STAGE3:-0.55}"
  --enable_geometry_controller
  --enable_live_loss_logging
  --live_loss_log_interval 1
  --enable_sparse_optimization
  --sparse_mode spa
  --spa_prune_ratio 0.05
  --prune_ratio2 0.04
  --spa_metric hybrid
  --disable_viewer
  --ras_ref_pixels "${RAS_REF_PIXELS:-0}"
  --ras_scale_min "${RAS_SCALE_MIN:-1.0}"
  --ras_scale_max "${RAS_SCALE_MAX:-2.5}"
  --ras_densify_base "${RAS_DENSIFY_BASE:-7200}"
  --ras_densify_min "${RAS_DENSIFY_MIN:-10000}"
  --ras_densify_max "${RAS_DENSIFY_MAX:-18000}"
  --ras_spa_delay "${RAS_SPA_DELAY:-2000}"
  --adaptive_densify_extend_step "${ADAPTIVE_DENSIFY_EXTEND_STEP:-2000}"
  --adaptive_densify_max_iter "${ADAPTIVE_DENSIFY_MAX_ITER:-18000}"
  --adaptive_point_budget "${ADAPTIVE_POINT_BUDGET:-8000000}"
)

if [[ "$ENABLE_ADAPTIVE" == "1" ]]; then
  TRAIN_ARGS+=(--enable_adaptive_stage_controller)
fi

case_has_metrics() {
  python - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
results_path = model_dir / "results.json"
if not results_path.is_file():
    raise SystemExit(1)
data = json.loads(results_path.read_text(encoding="utf-8"))
for method in ("ours_28500", f"ours_{30000}"):
    if method not in data:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

if [[ "$RUN_TRAIN" == "1" ]]; then
  FREE_KB="$(df -Pk "$OUTPUT_ROOT" | awk 'NR == 2 {print $4}')"
  MIN_FREE_KB="$((6 * 1024 * 1024))"
  if [[ -n "$FREE_KB" && "$FREE_KB" -lt "$MIN_FREE_KB" ]]; then
    echo "Insufficient free disk space at $OUTPUT_ROOT: need at least 6 GiB for two representative PLYs; have $((FREE_KB / 1024)) MiB." >&2
    exit 1
  fi
  if [[ "$SKIP_COMPLETED" == "1" ]] && { [[ -f "$DONE_MARKER" ]] || case_has_metrics; }; then
    echo "Skip training (already completed): $CASE_NAME"
  else
    mkdir -p "$MODEL_DIR"
    echo "Training -> $MODEL_DIR"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
      TAGGS_CPU_KNN_INIT="${TAGGS_CPU_KNN_INIT:-1}" \
      TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD="${TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD:-50000}" \
      TAGGS_DENSIFY_INTERVAL_MULT_FULLRES="${TAGGS_DENSIFY_INTERVAL_MULT_FULLRES:-2}" \
      TAGGS_DENSIFY_EMPTY_CACHE="${TAGGS_DENSIFY_EMPTY_CACHE:-1}" \
      KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}" \
      python train_optimized.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_PATH"
  fi
fi

if [[ "$RUN_RENDER_METRICS" == "1" ]]; then
  PLAN_JSON="$MODEL_DIR/schedule_plan.json"
  # Schedule checkpoints remain evaluation/logging points. Only the two
  # representative PLYs requested by this runner are rendered and retained.
  STAGE_ITERS=(28500 "$ITERATIONS")
  for iter in "${STAGE_ITERS[@]}"; do
    point_path="$MODEL_DIR/point_cloud/iteration_${iter}/point_cloud.ply"
    if [[ ! -f "$point_path" ]]; then
      echo "Skip render iteration $iter (missing $point_path)"
      continue
    fi
    echo "Render iteration $iter"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python render.py -m "$MODEL_DIR" --iteration "$iter" --skip_train --quiet
  done
  echo "Metrics -> $MODEL_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python metrics.py -m "$MODEL_DIR" --split test
fi

python scripts/summarize_v15_fast_checkpoint.py \
  --model_dir "$MODEL_DIR" \
  --primary_iter "$PRIMARY_ITER" \
  --final_iter "$ITERATIONS" \
  --out_json "$CHECKPOINT_JSON" \
  --out_md "$CHECKPOINT_MD"

python scripts/summarize_v15_stage_selection.py \
  --model_dir "$MODEL_DIR" \
  --out_md "$STAGE_MD" \
  --out_json "$STAGE_JSON" \
  --config_label "v16 resolution-aware, images=${IMAGES}, resolution=${RESOLUTION}, variant=${VARIANT}" \
  --baseline_method "$BASELINE_METHOD" \
  --final_method "$FINAL_METHOD" \
  --psnr_drop_threshold "$PSNR_DROP_THRESHOLD" \
  --perceptual_psnr_budget "$PERCEPTUAL_PSNR_BUDGET" \
  --chronic_triple_limit "$CHRONIC_TRIPLE_LIMIT"

date -Is > "$DONE_MARKER"

echo ""
echo "=== v16 complete ==="
echo "  schedule_plan: ${MODEL_DIR}/schedule_plan.json"
echo "  stage summary: ${STAGE_MD}"
echo "  log:           ${LOG_PATH}"
