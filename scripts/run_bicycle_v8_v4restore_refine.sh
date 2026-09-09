#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE="${SOURCE:-/root/highspeedstorage/TAGS-A100/data/bicycle}"
DEPTHS="${DEPTHS:-/root/highspeedstorage/TAGS-A100/data/bicycle/depth_any_out}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/highspeedstorage/TAGS-A100/TAGSs/output}"
ITERATIONS="${ITERATIONS:-30000}"
METHOD="${METHOD:-ours_${ITERATIONS}}"
BASELINE_CASE="${BASELINE_CASE:-bicycle_integrated_v4_restore}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER_METRICS="${RUN_RENDER_METRICS:-1}"
SKIP_COMPLETED_CASES="${SKIP_COMPLETED_CASES:-1}"

base_args=(
  -s "$SOURCE"
  --images images
  --depths "$DEPTHS"
  --resolution -1
  --data_device cpu
  --eval
  --llffhold 8
  --iterations "$ITERATIONS"
  --position_lr_init 0.00016
  --position_lr_final 0.0000016
  --position_lr_delay_mult 0.01
  --position_lr_max_steps "$ITERATIONS"
  --feature_lr 0.0025
  --opacity_lr 0.025
  --scaling_lr 0.005
  --rotation_lr 0.001
  --exposure_lr_init 0.01
  --exposure_lr_final 0.001
  --exposure_lr_delay_steps 0
  --exposure_lr_delay_mult 0.0
  --percent_dense 0.01
  --lambda_dssim 0.2
  --densification_interval 100
  --opacity_reset_interval 3000
  --densify_from_iter 500
  --densify_until_iter 18000
  --densify_grad_threshold 0.00016
  --depth_l1_weight_init 1.0
  --depth_l1_weight_final 0.01
  --enable_geo_prior
  --geo_prior_depth
  --geo_prior_normal
  --depth_prior_loss_type edge_aware_log_l1
  --normal_prior_loss_type cosine
  --depth_prior_pearson_weight 0.0
  --depth_prior_local_pearson_weight 0.05
  --normal_prior_tv_weight 0.003
  --depth_prior_edge_quantile 0.92
  --depth_prior_rgb_edge_quantile 0.95
  --depth_prior_var_quantile 0.92
  --depth_prior_min_mask_fraction 0.2
  --normal_tv_depth_edge_quantile 0.7
  --normal_tv_rgb_edge_quantile 0.7
  --depth_prior_weight_init 0.0
  --depth_prior_weight_final 0.05
  --depth_prior_weight_min 0.01
  --normal_prior_weight_init 0.0
  --normal_prior_weight_final 0.02
  --normal_prior_weight_min 0.0
  --geo_prior_low_texture_relax_strength 0.0
  --geo_prior_low_texture_relax_min 0.90
  --geo_prior_warmup_until 5000
  --geo_prior_decay_start 12000
  --geo_prior_decay_until 24000
  --detail_loss_weight_init 0.0
  --detail_loss_weight_final 0.0
  --detail_loss_start_iter 4000
  --detail_loss_edge_quantile 0.75
  --detail_loss_max_weight 0.05
  --edge_guided_densify_factor 0.90
  --lpips_loss_weight_init 0.0
  --lpips_loss_weight_final 0.010
  --lpips_loss_start_iter 16000
  --lpips_loss_interval 4
  --lpips_loss_max_weight 0.02
  --enable_resolution_adaptive
  --resolution_schedule "1200:4,4000:2,30000:1"
  --enable_group_lr_schedule
  --lr_stage1_end 6000
  --lr_stage2_end 18000
  --lr_geo_mult_stage1 1.10
  --lr_geo_mult_stage2 1.00
  --lr_geo_mult_stage3 0.75
  --lr_color_mult_stage1 0.75
  --lr_color_mult_stage2 1.05
  --lr_color_mult_stage3 0.80
  --enable_geometry_controller
  --geometry_start_iter 9000
  --geometry_interval 250
  --geometry_min_opacity 0.008
  --geometry_scale_cap_quantile 0.98
  --geometry_scale_cap_max 0.12
  --geometry_prune_scale_factor 0.85
  --geometry_protect_grad_quantile 0.95
  --enable_sparse_optimization
  --sparse_mode spa
  --spa_interval 1000
  --spa_start_iter 13000
  --spa_end_iter 25000
  --spa_prune_ratio 0.05
  --prune_ratio2 0.04
  --rho_lr 0.0005
  --spa_metric hybrid
  --spa_require_full_resolution
  --spa_after_fullres_delay 3000
  --spa_max_prune_points 30000
  --spa_protect_top_fraction 0.05
  --test_iterations 7000 "$ITERATIONS"
  --save_iterations 7000 "$ITERATIONS"
  --disable_viewer
)

run_case() {
  local name="$1"
  shift
  local model_dir="$OUTPUT_ROOT/$name"
  local done_marker="$model_dir/.v8_v4restore_refine_case_done"

  echo "=== Case: $name ==="
  if [[ "$SKIP_COMPLETED_CASES" == "1" ]] && case_is_complete "$name"; then
    echo "Skip completed case: $name"
    return
  fi

  if [[ "$RUN_TRAIN" == "1" ]]; then
    if [[ "$SKIP_COMPLETED_CASES" == "1" ]] && case_has_final_model "$model_dir"; then
      echo "Final model exists for $name; skip training and continue with render/metrics."
    else
      TAGGS_CPU_KNN_INIT="${TAGGS_CPU_KNN_INIT:-1}" python train_optimized.py \
        -m "$model_dir" \
        "${base_args[@]}" \
        "$@"
    fi
  fi

  if [[ "$RUN_RENDER_METRICS" == "1" ]]; then
    if [[ "$SKIP_COMPLETED_CASES" == "1" ]] && case_has_metrics "$model_dir"; then
      echo "Metrics exist for $name; skip render/metrics."
    else
      python render.py -m "$model_dir" --skip_train --quiet
      python metrics.py -m "$model_dir" --split test
    fi
  fi

  mkdir -p "$model_dir"
  {
    echo "case=$name"
    echo "method=$METHOD"
    echo "iterations=$ITERATIONS"
    date -Is
  } > "$done_marker"
}

case_has_metrics() {
  local model_dir="$1"
  python - "$model_dir" "$METHOD" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
method = sys.argv[2]

def has_method(filename):
    path = model_dir / filename
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return method in data

raise SystemExit(0 if has_method("results.json") and has_method("per_view.json") else 1)
PY
}

case_has_final_model() {
  local model_dir="$1"
  [[ -f "$model_dir/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]]
}

case_is_complete() {
  local name="$1"
  local model_dir="$OUTPUT_ROOT/$name"
  local done_marker="$model_dir/.v8_v4restore_refine_case_done"

  if [[ "$RUN_RENDER_METRICS" == "1" ]]; then
    [[ -f "$done_marker" ]] || case_has_metrics "$model_dir"
  elif [[ "$RUN_TRAIN" == "1" ]]; then
    [[ -f "$done_marker" ]] || case_has_final_model "$model_dir"
  else
    [[ -f "$done_marker" ]]
  fi
}

summary_cases=(
  bicycle_v7_color_ultralate
  bicycle_v7_geo_decay_long
  bicycle_v8_color_geo_ultralate
  bicycle_v8_color_geo_lean
  bicycle_v8_color_geo_soft
  bicycle_v8_color_lean
)

# v7 showed color_ultralate and geo_decay_long as the only aggregate-positive
# moves. Start by testing the direct composition.
run_case bicycle_v8_color_geo_ultralate \
  --geo_prior_decay_until 26000 \
  --color_align_weight_init 0.0 \
  --color_align_weight_final 0.0015 \
  --color_align_start_iter 26000 \
  --color_align_max_weight 0.004 \
  --color_align_std_weight 0.25 \
  --color_align_hard_view_boost 1.0 \
  --color_align_low_grad_quantile 0.50 \
  --color_align_min_mask_fraction 0.20 \
  --color_align_require_fullres

# Same composition, but reduce the color-statistics term to protect views where
# color_ultralate paid a PSNR/LPIPS cost, especially 00007/00008/00021.
run_case bicycle_v8_color_geo_lean \
  --geo_prior_decay_until 26000 \
  --color_align_weight_init 0.0 \
  --color_align_weight_final 0.0012 \
  --color_align_start_iter 27000 \
  --color_align_max_weight 0.003 \
  --color_align_std_weight 0.20 \
  --color_align_hard_view_boost 1.0 \
  --color_align_low_grad_quantile 0.45 \
  --color_align_min_mask_fraction 0.20 \
  --color_align_require_fullres

# Keep the color_ultralate settings and only partially extend geometry decay;
# this checks whether the geo gain saturates before 26000.
run_case bicycle_v8_color_geo_soft \
  --geo_prior_decay_until 25000 \
  --color_align_weight_init 0.0 \
  --color_align_weight_final 0.0015 \
  --color_align_start_iter 26000 \
  --color_align_max_weight 0.004 \
  --color_align_std_weight 0.25 \
  --color_align_hard_view_boost 1.0 \
  --color_align_low_grad_quantile 0.50 \
  --color_align_min_mask_fraction 0.20 \
  --color_align_require_fullres

# Isolate the conservative color schedule so the next summary can separate the
# benefit of weaker color alignment from the benefit of longer geo decay.
run_case bicycle_v8_color_lean \
  --color_align_weight_init 0.0 \
  --color_align_weight_final 0.0012 \
  --color_align_start_iter 27000 \
  --color_align_max_weight 0.003 \
  --color_align_std_weight 0.20 \
  --color_align_hard_view_boost 1.0 \
  --color_align_low_grad_quantile 0.45 \
  --color_align_min_mask_fraction 0.20 \
  --color_align_require_fullres

python scripts/summarize_v4restore_ablation.py \
  --output_root "$OUTPUT_ROOT" \
  --baseline_case "$BASELINE_CASE" \
  --cases "${summary_cases[@]}" \
  --method "$METHOD" \
  --out_json "$OUTPUT_ROOT/bicycle_v8_v4restore_refine_summary.json" \
  --out_md "$OUTPUT_ROOT/bicycle_v8_v4restore_refine_summary.md"

echo "v4_restore refinement finished."
