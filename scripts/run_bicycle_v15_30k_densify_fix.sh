#!/usr/bin/env bash
# v15 30k densify-fix experiments — Linux / A100 / H800
#
# Best current recipe: E1b (densify_until=7200, late_stabilize)
#
# Usage (repo root):
#   bash scripts/run_bicycle_v15_30k_densify_fix.sh
#
# Run E1b explicitly:
#   EXPERIMENT=E1b bash scripts/run_bicycle_v15_30k_densify_fix.sh
#
# Force retrain:
#   EXPERIMENT=E1b SKIP_COMPLETED=0 bash scripts/run_bicycle_v15_30k_densify_fix.sh
#
# Optional env file:
#   source scripts/configs/bicycle_v15_30k_densify_fix.env
#   bash scripts/run_bicycle_v15_30k_densify_fix.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPERIMENT="${EXPERIMENT:-E1b}"
EXPERIMENT="${EXPERIMENT^^}"

SOURCE="${SOURCE:-$ROOT/data/bicycle}"
DEPTHS="${DEPTHS:-$ROOT/data/bicycle/depth_any_out}"
ITERATIONS="${ITERATIONS:-30000}"
RESOLUTION="${RESOLUTION:-800}"
IMAGES="${IMAGES:-images_4}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER_METRICS="${RUN_RENDER_METRICS:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/output/v15_logs_linux}"

PSNR_DROP_THRESHOLD="${PSNR_DROP_THRESHOLD:-0.05}"
PERCEPTUAL_PSNR_BUDGET="${PERCEPTUAL_PSNR_BUDGET:-0.10}"
CHRONIC_TRIPLE_LIMIT="${CHRONIC_TRIPLE_LIMIT:-1}"

DEFAULT_STAGE_SPEC="7000,12000,14000,15000,16000,17000,18000,20000,24000,27000,28500,30000"
E1B_STAGE_SPEC="7000,7200,8000,10000,12000,14000,15000,16000,17000,18000,20000,24000,27000,28500,30000"

# late_stabilize defaults (same as Windows E1b)
LATE_STABILIZE_LR_GEO_MULT_STAGE3="${LATE_STABILIZE_LR_GEO_MULT_STAGE3:-0.40}"
LATE_STABILIZE_LR_COLOR_MULT_STAGE3="${LATE_STABILIZE_LR_COLOR_MULT_STAGE3:-0.55}"
LATE_STABILIZE_LPIPS_START_FULL="${LATE_STABILIZE_LPIPS_START_FULL:-24000}"
LATE_STABILIZE_LPIPS_WEIGHT_FINAL="${LATE_STABILIZE_LPIPS_WEIGHT_FINAL:-0.006}"
LATE_STABILIZE_LPIPS_MAX_WEIGHT="${LATE_STABILIZE_LPIPS_MAX_WEIGHT:-0.010}"
LATE_STABILIZE_LPIPS_INTERVAL="${LATE_STABILIZE_LPIPS_INTERVAL:-10}"
LATE_STABILIZE_COLOR_ALIGN_START_FULL="${LATE_STABILIZE_COLOR_ALIGN_START_FULL:-25000}"
LATE_STABILIZE_COLOR_ALIGN_WEIGHT_FINAL="${LATE_STABILIZE_COLOR_ALIGN_WEIGHT_FINAL:-0.0010}"
LATE_STABILIZE_COLOR_ALIGN_MAX_WEIGHT="${LATE_STABILIZE_COLOR_ALIGN_MAX_WEIGHT:-0.0020}"
LATE_STABILIZE_PSNR_START_FULL="${LATE_STABILIZE_PSNR_START_FULL:-22000}"
LATE_STABILIZE_PSNR_WEIGHT_FINAL="${LATE_STABILIZE_PSNR_WEIGHT_FINAL:-0.008}"
LATE_STABILIZE_PSNR_MAX_WEIGHT="${LATE_STABILIZE_PSNR_MAX_WEIGHT:-0.012}"

load_profile() {
  local id="$1"
  case "$id" in
    E0)
      PROFILE_ID="E0"
      PROFILE_LABEL="diagnostic baseline (densify until 18k)"
      OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/output/v15_30k_fix_e0_diag}"
      CASE_NAME="${CASE_NAME:-bicycle_v15_3080_e0_diag_baseline}"
      DENSIFY_UNTIL_FULL="18000"
      SPA_PRUNE_RATIO="0.05"
      PRUNE_RATIO2="0.04"
      TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD="${TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD:-50000}"
      TAGGS_DENSIFY_INTERVAL_MULT_FULLRES="${TAGGS_DENSIFY_INTERVAL_MULT_FULLRES:-2}"
      STAGE_SPEC="${RAW_STAGE_ITERATIONS:-$DEFAULT_STAGE_SPEC}"
      LPIPS_START_FULL="$LATE_STABILIZE_LPIPS_START_FULL"
      LPIPS_WEIGHT_FINAL="$LATE_STABILIZE_LPIPS_WEIGHT_FINAL"
      ;;
    E1)
      PROFILE_ID="E1"
      PROFILE_LABEL="early stop densify at 12k"
      OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/output/v15_30k_fix_e1_densify12k}"
      CASE_NAME="${CASE_NAME:-bicycle_v15_3080_e1_densify12k}"
      DENSIFY_UNTIL_FULL="12000"
      SPA_PRUNE_RATIO="0.05"
      PRUNE_RATIO2="0.04"
      TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD="${TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD:-50000}"
      TAGGS_DENSIFY_INTERVAL_MULT_FULLRES="${TAGGS_DENSIFY_INTERVAL_MULT_FULLRES:-2}"
      STAGE_SPEC="${RAW_STAGE_ITERATIONS:-$DEFAULT_STAGE_SPEC}"
      LPIPS_START_FULL="$LATE_STABILIZE_LPIPS_START_FULL"
      LPIPS_WEIGHT_FINAL="$LATE_STABILIZE_LPIPS_WEIGHT_FINAL"
      ;;
    E1B)
      PROFILE_ID="E1b"
      PROFILE_LABEL="align densify until 7.2k (12k promotion ratio 18000*12/30)"
      OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/output/v15_30k_fix_e1b_densify7k}"
      CASE_NAME="${CASE_NAME:-bicycle_v15_3080_e1b_densify7k}"
      DENSIFY_UNTIL_FULL="7200"
      SPA_PRUNE_RATIO="0.05"
      PRUNE_RATIO2="0.04"
      TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD="${TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD:-50000}"
      TAGGS_DENSIFY_INTERVAL_MULT_FULLRES="${TAGGS_DENSIFY_INTERVAL_MULT_FULLRES:-2}"
      STAGE_SPEC="${RAW_STAGE_ITERATIONS:-$E1B_STAGE_SPEC}"
      LPIPS_START_FULL="$LATE_STABILIZE_LPIPS_START_FULL"
      LPIPS_WEIGHT_FINAL="$LATE_STABILIZE_LPIPS_WEIGHT_FINAL"
      ;;
    E2)
      PROFILE_ID="E2"
      PROFILE_LABEL="E1 + tighter point-growth gate"
      OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/output/v15_30k_fix_e2_pointgate}"
      CASE_NAME="${CASE_NAME:-bicycle_v15_3080_e2_pointgate}"
      DENSIFY_UNTIL_FULL="12000"
      SPA_PRUNE_RATIO="0.05"
      PRUNE_RATIO2="0.04"
      TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD="${TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD:-800000}"
      TAGGS_DENSIFY_INTERVAL_MULT_FULLRES="${TAGGS_DENSIFY_INTERVAL_MULT_FULLRES:-3}"
      STAGE_SPEC="${RAW_STAGE_ITERATIONS:-$DEFAULT_STAGE_SPEC}"
      LPIPS_START_FULL="$LATE_STABILIZE_LPIPS_START_FULL"
      LPIPS_WEIGHT_FINAL="$LATE_STABILIZE_LPIPS_WEIGHT_FINAL"
      ;;
    E3)
      PROFILE_ID="E3"
      PROFILE_LABEL="E1 + stronger SPA prune"
      OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/output/v15_30k_fix_e3_strongprune}"
      CASE_NAME="${CASE_NAME:-bicycle_v15_3080_e3_strongprune}"
      DENSIFY_UNTIL_FULL="12000"
      SPA_PRUNE_RATIO="0.08"
      PRUNE_RATIO2="0.06"
      TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD="${TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD:-50000}"
      TAGGS_DENSIFY_INTERVAL_MULT_FULLRES="${TAGGS_DENSIFY_INTERVAL_MULT_FULLRES:-2}"
      STAGE_SPEC="${RAW_STAGE_ITERATIONS:-$DEFAULT_STAGE_SPEC}"
      LPIPS_START_FULL="$LATE_STABILIZE_LPIPS_START_FULL"
      LPIPS_WEIGHT_FINAL="$LATE_STABILIZE_LPIPS_WEIGHT_FINAL"
      ;;
    E4)
      PROFILE_ID="E4"
      PROFILE_LABEL="E1 + lighter late LPIPS"
      OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/output/v15_30k_fix_e4_lightlpips}"
      CASE_NAME="${CASE_NAME:-bicycle_v15_3080_e4_lightlpips}"
      DENSIFY_UNTIL_FULL="12000"
      SPA_PRUNE_RATIO="0.05"
      PRUNE_RATIO2="0.04"
      TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD="${TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD:-50000}"
      TAGGS_DENSIFY_INTERVAL_MULT_FULLRES="${TAGGS_DENSIFY_INTERVAL_MULT_FULLRES:-2}"
      STAGE_SPEC="${RAW_STAGE_ITERATIONS:-$DEFAULT_STAGE_SPEC}"
      LPIPS_START_FULL="26000"
      LPIPS_WEIGHT_FINAL="0.004"
      ;;
    *)
      echo "Unknown EXPERIMENT='$EXPERIMENT'. Use E0|E1|E1b|E2|E3|E4" >&2
      exit 1
      ;;
  esac
}

parse_stage_iterations() {
  IFS=',' read -r -a STAGE_ITERATIONS <<< "$STAGE_SPEC"
  STAGE_METHODS=()
  for iter in "${STAGE_ITERATIONS[@]}"; do
    STAGE_METHODS+=("ours_${iter}")
  done
  STAGE_TRACE="$(IFS=,; echo "${STAGE_ITERATIONS[*]}")"
}

case_has_metrics() {
  python - "$MODEL_DIR" "${STAGE_METHODS[@]}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
methods = sys.argv[2:]
results_path = model_dir / "results.json"
if not results_path.is_file():
    raise SystemExit(1)
data = json.loads(results_path.read_text(encoding="utf-8"))
for method in methods:
    if method not in data:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

run_experiment() {
  load_profile "$EXPERIMENT"
  parse_stage_iterations

  MODEL_DIR="$OUTPUT_ROOT/$CASE_NAME"
  CONFIG_LABEL="exp=${PROFILE_ID}, ${PROFILE_LABEL}, images=${IMAGES}, resolution=${RESOLUTION}, iterations=${ITERATIONS}, densify_until_full=${DENSIFY_UNTIL_FULL}"
  DONE_MARKER="$MODEL_DIR/.v15_densify_fix_done"
  LOG_PATH="$LOG_DIR/${CASE_NAME}.log"
  STAGE_MD="$OUTPUT_ROOT/${CASE_NAME}_stage_selection_summary.md"
  STAGE_JSON="$OUTPUT_ROOT/${CASE_NAME}_stage_selection_summary.json"
  CHECKPOINT_MD="$OUTPUT_ROOT/${CASE_NAME}_checkpoint_summary.md"
  CHECKPOINT_JSON="$OUTPUT_ROOT/${CASE_NAME}_checkpoint_summary.json"
  BASELINE_METHOD="ours_${STAGE_ITERATIONS[0]}"
  FINAL_METHOD="ours_${ITERATIONS}"
  PRIMARY_ITER="28500"

  mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$MODEL_DIR"

  echo ""
  echo "=== v15 30k densify-fix (Linux) | ${PROFILE_ID} | ${CASE_NAME} ==="
  echo "LABEL:       ${PROFILE_LABEL}"
  echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
  echo "MODEL_DIR:   ${MODEL_DIR}"
  echo "CONFIG:      ${CONFIG_LABEL}"
  echo "STAGES:      ${STAGE_TRACE}"
  echo "DENSIFY:     500 -> ${DENSIFY_UNTIL_FULL}"
  echo "SPA:         prune=${SPA_PRUNE_RATIO}/${PRUNE_RATIO2}"
  echo "LPIPS:       start=${LPIPS_START_FULL}, weight=${LPIPS_WEIGHT_FINAL}"
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
    --densify_from_iter 500
    --densify_until_iter "$DENSIFY_UNTIL_FULL"
    --densify_grad_threshold 0.00016
    --enable_geo_prior
    --geo_prior_depth
    --geo_prior_normal
    --depth_prior_weight_final 0.05
    --normal_prior_weight_final 0.02
    --normal_prior_tv_weight 0.003
    --geo_prior_start_iter 0
    --geo_prior_warmup_until 5000
    --geo_prior_decay_start 12000
    --geo_prior_decay_until 25000
    --lpips_loss_start_iter "$LPIPS_START_FULL"
    --lpips_loss_weight_final "$LPIPS_WEIGHT_FINAL"
    --lpips_loss_max_weight "$LATE_STABILIZE_LPIPS_MAX_WEIGHT"
    --lpips_loss_interval "$LATE_STABILIZE_LPIPS_INTERVAL"
    --psnr_loss_start_iter "$LATE_STABILIZE_PSNR_START_FULL"
    --psnr_loss_weight_final "$LATE_STABILIZE_PSNR_WEIGHT_FINAL"
    --psnr_loss_max_weight "$LATE_STABILIZE_PSNR_MAX_WEIGHT"
    --color_align_start_iter "$LATE_STABILIZE_COLOR_ALIGN_START_FULL"
    --color_align_weight_final "$LATE_STABILIZE_COLOR_ALIGN_WEIGHT_FINAL"
    --color_align_max_weight "$LATE_STABILIZE_COLOR_ALIGN_MAX_WEIGHT"
    --color_align_std_weight 0.25
    --enable_resolution_adaptive
    --resolution_schedule "1200:4,4000:2,${ITERATIONS}:1"
    --enable_group_lr_schedule
    --lr_stage1_end 6000
    --lr_stage2_end 18000
    --lr_geo_mult_stage1 1.10
    --lr_geo_mult_stage2 1.00
    --lr_geo_mult_stage3 "$LATE_STABILIZE_LR_GEO_MULT_STAGE3"
    --lr_color_mult_stage1 0.75
    --lr_color_mult_stage2 1.05
    --lr_color_mult_stage3 "$LATE_STABILIZE_LR_COLOR_MULT_STAGE3"
    --enable_geometry_controller
    --enable_live_loss_logging
    --live_loss_log_interval 1
    --live_stage_iterations "$STAGE_TRACE"
    --enable_sparse_optimization
    --sparse_mode spa
    --spa_start_iter 13000
    --spa_end_iter 25000
    --spa_prune_ratio "$SPA_PRUNE_RATIO"
    --prune_ratio2 "$PRUNE_RATIO2"
    --spa_metric hybrid
    --test_iterations "${STAGE_ITERATIONS[@]}"
    --save_iterations "${STAGE_ITERATIONS[@]}"
    --disable_viewer
  )

  if [[ "$RUN_TRAIN" == "1" ]]; then
    if [[ "$SKIP_COMPLETED" == "1" ]] && { [[ -f "$DONE_MARKER" ]] || case_has_metrics; }; then
      echo "Skip training (already completed): $CASE_NAME"
    else
      echo "Training -> $MODEL_DIR"
      CUDA_VISIBLE_DEVICES="$GPU_ID" \
        TAGGS_CPU_KNN_INIT="${TAGGS_CPU_KNN_INIT:-1}" \
        TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD="$TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD" \
        TAGGS_DENSIFY_INTERVAL_MULT_FULLRES="$TAGGS_DENSIFY_INTERVAL_MULT_FULLRES" \
        TAGGS_DENSIFY_EMPTY_CACHE="${TAGGS_DENSIFY_EMPTY_CACHE:-1}" \
        KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}" \
        python train_optimized.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_PATH"
    fi
  fi

  if [[ "$RUN_RENDER_METRICS" == "1" ]]; then
    for iter in "${STAGE_ITERATIONS[@]}"; do
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
    --config_label "$CONFIG_LABEL" \
    --baseline_method "$BASELINE_METHOD" \
    --final_method "$FINAL_METHOD" \
    --psnr_drop_threshold "$PSNR_DROP_THRESHOLD" \
    --perceptual_psnr_budget "$PERCEPTUAL_PSNR_BUDGET" \
    --chronic_triple_limit "$CHRONIC_TRIPLE_LIMIT"

  date -Is > "$DONE_MARKER"

  echo ""
  echo "=== Gate hints ==="
  echo "  Quality-primary target: ours_${PRIMARY_ITER}"
  echo "  Perceptual target:      ours_${ITERATIONS}"
  echo "  Stage summary:          ${STAGE_MD}"
  echo "  Log:                    ${LOG_PATH}"
}

echo "=== bicycle v15 30k densify-fix (Linux) ==="
echo "EXPERIMENT:   ${EXPERIMENT}"
echo "PROTOCOL:     ${IMAGES} @ resolution=${RESOLUTION}"
echo "ITERATIONS:   ${ITERATIONS}"
echo "SOURCE:       ${SOURCE}"
echo "DEPTHS:       ${DEPTHS}"
echo "GPU_ID:       ${GPU_ID}"
echo ""

run_experiment

echo ""
echo "Done."
