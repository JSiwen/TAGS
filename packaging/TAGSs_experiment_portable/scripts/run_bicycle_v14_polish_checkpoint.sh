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
GATE_REFERENCE_CASE="${GATE_REFERENCE_CASE:-bicycle_v8_color_geo_soft}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER_METRICS="${RUN_RENDER_METRICS:-1}"
SKIP_COMPLETED_CASES="${SKIP_COMPLETED_CASES:-1}"
ONLY_CASE="${ONLY_CASE:-}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/v14_polish_checkpoint_logs}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
EXPORT_STAGE_TB="${EXPORT_STAGE_TB:-1}"

GATE_PSNR_DROP="${GATE_PSNR_DROP:-0.01}"
GATE_SSIM_DROP="${GATE_SSIM_DROP:-0.0002}"
GATE_LPIPS_RISE="${GATE_LPIPS_RISE:-0.0005}"
GATE_PSNR_REGRESSED_VIEWS="${GATE_PSNR_REGRESSED_VIEWS:-10}"
GATE_LPIPS_REGRESSED_VIEWS="${GATE_LPIPS_REGRESSED_VIEWS:-12}"
GATE_CHRONIC_PSNR_DROP="${GATE_CHRONIC_PSNR_DROP:-0.12}"
GATE_CHRONIC_AVG_LPIPS_DELTA="${GATE_CHRONIC_AVG_LPIPS_DELTA:-0.0015}"

STAGE_ITERATIONS=(7000 18000 24000 25500 28500 "$ITERATIONS")
STAGE_METHODS=()
for iter in "${STAGE_ITERATIONS[@]}"; do
  STAGE_METHODS+=("ours_${iter}")
done

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "No GPU ids provided in GPU_IDS_CSV."
  exit 1
fi
mkdir -p "$LOG_DIR"

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
  --geo_prior_start_iter 0
  --geo_prior_warmup_until 5000
  --geo_prior_decay_start 12000
  --geo_prior_decay_until 25000
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
  --color_align_weight_init 0.0
  --color_align_weight_final 0.0015
  --color_align_start_iter 26000
  --color_align_max_weight 0.004
  --color_align_std_weight 0.25
  --color_align_hard_view_boost 1.0
  --color_align_low_grad_quantile 0.50
  --color_align_min_mask_fraction 0.20
  --color_align_require_fullres
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
  --test_iterations "${STAGE_ITERATIONS[@]}"
  --save_iterations "${STAGE_ITERATIONS[@]}"
  --disable_viewer
)

case_has_final_model() {
  local model_dir="$1"
  [[ -f "$model_dir/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]]
}

case_has_metrics() {
  local model_dir="$1"
  python - "$model_dir" "$METHOD" "${STAGE_METHODS[@]}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
final_method = sys.argv[2]
stage_methods = sys.argv[3:]

def has_method(filename, method):
    path = model_dir / filename
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return method in data

if not (has_method("results.json", final_method) and has_method("per_view.json", final_method)):
    raise SystemExit(1)

for method in stage_methods:
    method_dir = model_dir / "test" / method
    if not ((method_dir / "renders").is_dir() and (method_dir / "gt").is_dir()):
        raise SystemExit(1)
    if not (has_method("results.json", method) and has_method("per_view.json", method)):
        raise SystemExit(1)

raise SystemExit(0)
PY
}

case_has_stage_tensorboard() {
  local model_dir="$1"
  [[ -f "$model_dir/.v14_stage_tensorboard_done" ]]
}

export_stage_tensorboard() {
  local model_dir="$1"
  if [[ "$EXPORT_STAGE_TB" != "1" ]]; then
    return
  fi
  if [[ "$SKIP_COMPLETED_CASES" == "1" ]] && case_has_stage_tensorboard "$model_dir"; then
    return
  fi
  python scripts/export_stage_metrics_to_tensorboard.py \
    --case_dir "$model_dir" \
    --gate_reference_dir "$OUTPUT_ROOT/$GATE_REFERENCE_CASE" \
    --final_method "$METHOD" \
    --stage_methods "${STAGE_METHODS[@]}" \
    --gate_psnr_drop "$GATE_PSNR_DROP" \
    --gate_ssim_drop "$GATE_SSIM_DROP" \
    --gate_lpips_rise "$GATE_LPIPS_RISE" \
    --gate_psnr_regressed_views "$GATE_PSNR_REGRESSED_VIEWS" \
    --gate_lpips_regressed_views "$GATE_LPIPS_REGRESSED_VIEWS" \
    --gate_chronic_psnr_drop "$GATE_CHRONIC_PSNR_DROP" \
    --gate_chronic_avg_lpips_delta "$GATE_CHRONIC_AVG_LPIPS_DELTA"
  date -Is > "$model_dir/.v14_stage_tensorboard_done"
}

case_is_complete() {
  local name="$1"
  local model_dir="$OUTPUT_ROOT/$name"
  local done_marker="$model_dir/.v14_polish_checkpoint_case_done"
  if [[ "$RUN_RENDER_METRICS" == "1" ]]; then
    [[ -f "$done_marker" ]] && case_has_metrics "$model_dir"
  elif [[ "$RUN_TRAIN" == "1" ]]; then
    [[ -f "$done_marker" ]] && case_has_final_model "$model_dir"
  else
    [[ -f "$done_marker" ]]
  fi
}

render_stage_iterations() {
  local gpu_id="$1"
  local model_dir="$2"
  for iter in "${STAGE_ITERATIONS[@]}"; do
    local point_path="$model_dir/point_cloud/iteration_${iter}/point_cloud.ply"
    if [[ ! -f "$point_path" ]]; then
      echo "Skip render iteration $iter for $model_dir: missing $point_path"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu_id" python render.py -m "$model_dir" --iteration "$iter" --skip_train --quiet
  done
}

run_case() {
  local gpu_id="$1"
  local name="$2"
  shift 2
  local model_dir="$OUTPUT_ROOT/$name"
  local done_marker="$model_dir/.v14_polish_checkpoint_case_done"

  if [[ -n "$ONLY_CASE" ]] && [[ "$name" != "$ONLY_CASE" ]]; then
    return
  fi

  echo "=== Case: $name on GPU $gpu_id ==="
  if [[ "$SKIP_COMPLETED_CASES" == "1" ]] && case_is_complete "$name"; then
    echo "Skip completed case: $name"
    export_stage_tensorboard "$model_dir"
    return
  fi

  if [[ "$RUN_TRAIN" == "1" ]]; then
    if [[ "$SKIP_COMPLETED_CASES" == "1" ]] && case_has_final_model "$model_dir"; then
      echo "Final model exists for $name; skip training and continue with render/metrics."
    else
      CUDA_VISIBLE_DEVICES="$gpu_id" TAGGS_CPU_KNN_INIT="${TAGGS_CPU_KNN_INIT:-1}" python train_optimized.py \
        -m "$model_dir" \
        "${base_args[@]}" \
        "$@"
    fi
  fi

  if [[ "$RUN_RENDER_METRICS" == "1" ]]; then
    if [[ "$SKIP_COMPLETED_CASES" == "1" ]] && case_has_metrics "$model_dir"; then
      echo "Metrics exist for $name; skip render/metrics."
    else
      render_stage_iterations "$gpu_id" "$model_dir"
      CUDA_VISIBLE_DEVICES="$gpu_id" python metrics.py -m "$model_dir" --split test
    fi
    export_stage_tensorboard "$model_dir"
  fi

  mkdir -p "$model_dir"
  {
    echo "case=$name"
    echo "method=$METHOD"
    echo "iterations=$ITERATIONS"
    echo "stage_iterations=${STAGE_ITERATIONS[*]}"
    echo "gate_reference_case=$GATE_REFERENCE_CASE"
    echo "export_stage_tb=$EXPORT_STAGE_TB"
    date -Is
  } > "$done_marker"
}

cases=(
  bicycle_v14_v8_guard_recheck
  bicycle_v14_v8_guard_polish_soft075
  bicycle_v14_v8_guard_polish_soft050
  bicycle_v14_v8_guard_refine_end30000
  bicycle_v14_v8_monitor_28500_select
)

pids=()
pid_names=()
pid_logs=()
launched=0

launch_case() {
  local index="$1"
  local name="$2"
  shift 2
  local gpu_id="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
  local log_path="$LOG_DIR/${name}.log"
  (
    run_case "$gpu_id" "$name" "$@"
  ) >"$log_path" 2>&1 &
  pids+=("$!")
  pid_names+=("$name")
  pid_logs+=("$log_path")
}

if [[ -z "$ONLY_CASE" || "$ONLY_CASE" == "bicycle_v14_v8_guard_recheck" ]]; then
  launch_case "$launched" bicycle_v14_v8_guard_recheck \
    --enable_balanced_late_controller \
    --late_controller_structure_end_iter 25000 \
    --late_controller_handoff_end_iter 26000 \
    --late_controller_refine_end_iter 28500 \
    --late_controller_lpips_guard_start_iter 16000 \
    --late_controller_lpips_guard_max_boost 1.10 \
    --late_controller_lpips_guard_view_boost 1.05 \
    --late_controller_lpips_guard_chronic_boost 1.08 \
    --late_controller_lpips_guard_margin 0.08 \
    --late_controller_chronic_views "00008.png,00011.png,00016.png,00018.png,00021.png" \
    --late_controller_color_stage_boost 1.00 \
    --late_controller_color_polish_scale 1.00
  launched=$((launched + 1))
fi

if [[ -z "$ONLY_CASE" || "$ONLY_CASE" == "bicycle_v14_v8_guard_polish_soft075" ]]; then
  launch_case "$launched" bicycle_v14_v8_guard_polish_soft075 \
    --enable_balanced_late_controller \
    --late_controller_structure_end_iter 25000 \
    --late_controller_handoff_end_iter 26000 \
    --late_controller_refine_end_iter 28500 \
    --late_controller_lpips_guard_start_iter 16000 \
    --late_controller_lpips_guard_max_boost 1.10 \
    --late_controller_lpips_guard_view_boost 1.05 \
    --late_controller_lpips_guard_chronic_boost 1.08 \
    --late_controller_lpips_guard_margin 0.08 \
    --late_controller_chronic_views "00008.png,00011.png,00016.png,00018.png,00021.png" \
    --late_controller_color_stage_boost 1.00 \
    --late_controller_color_polish_scale 0.75
  launched=$((launched + 1))
fi

if [[ -z "$ONLY_CASE" || "$ONLY_CASE" == "bicycle_v14_v8_guard_polish_soft050" ]]; then
  launch_case "$launched" bicycle_v14_v8_guard_polish_soft050 \
    --enable_balanced_late_controller \
    --late_controller_structure_end_iter 25000 \
    --late_controller_handoff_end_iter 26000 \
    --late_controller_refine_end_iter 28500 \
    --late_controller_lpips_guard_start_iter 16000 \
    --late_controller_lpips_guard_max_boost 1.10 \
    --late_controller_lpips_guard_view_boost 1.05 \
    --late_controller_lpips_guard_chronic_boost 1.08 \
    --late_controller_lpips_guard_margin 0.08 \
    --late_controller_chronic_views "00008.png,00011.png,00016.png,00018.png,00021.png" \
    --late_controller_color_stage_boost 1.00 \
    --late_controller_color_polish_scale 0.50
  launched=$((launched + 1))
fi

if [[ -z "$ONLY_CASE" || "$ONLY_CASE" == "bicycle_v14_v8_guard_refine_end30000" ]]; then
  launch_case "$launched" bicycle_v14_v8_guard_refine_end30000 \
    --enable_balanced_late_controller \
    --late_controller_structure_end_iter 25000 \
    --late_controller_handoff_end_iter 26000 \
    --late_controller_refine_end_iter 30000 \
    --late_controller_lpips_guard_start_iter 16000 \
    --late_controller_lpips_guard_max_boost 1.10 \
    --late_controller_lpips_guard_view_boost 1.05 \
    --late_controller_lpips_guard_chronic_boost 1.08 \
    --late_controller_lpips_guard_margin 0.08 \
    --late_controller_chronic_views "00008.png,00011.png,00016.png,00018.png,00021.png" \
    --late_controller_color_stage_boost 1.00 \
    --late_controller_color_polish_scale 1.00
  launched=$((launched + 1))
fi

if [[ -z "$ONLY_CASE" || "$ONLY_CASE" == "bicycle_v14_v8_monitor_28500_select" ]]; then
  launch_case "$launched" bicycle_v14_v8_monitor_28500_select
fi

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  name="${pid_names[$idx]}"
  log_path="${pid_logs[$idx]}"
  if wait "$pid"; then
    echo "Case completed: $name (log: $log_path)"
  else
    echo "Case failed: $name (log: $log_path)"
    status=1
  fi
done

cases_for_summary=("${cases[@]}")
if [[ -n "$ONLY_CASE" ]]; then
  cases_for_summary=("$ONLY_CASE")
fi

python scripts/summarize_v4restore_ablation.py \
  --output_root "$OUTPUT_ROOT" \
  --baseline_case "$BASELINE_CASE" \
  --gate_reference_case "$GATE_REFERENCE_CASE" \
  --cases "${cases_for_summary[@]}" \
  --method "$METHOD" \
  --stage_methods "${STAGE_METHODS[@]}" \
  --gate_psnr_drop "$GATE_PSNR_DROP" \
  --gate_ssim_drop "$GATE_SSIM_DROP" \
  --gate_lpips_rise "$GATE_LPIPS_RISE" \
  --gate_psnr_regressed_views "$GATE_PSNR_REGRESSED_VIEWS" \
  --gate_lpips_regressed_views "$GATE_LPIPS_REGRESSED_VIEWS" \
  --gate_chronic_psnr_drop "$GATE_CHRONIC_PSNR_DROP" \
  --gate_chronic_avg_lpips_delta "$GATE_CHRONIC_AVG_LPIPS_DELTA" \
  --out_json "$OUTPUT_ROOT/bicycle_v14_polish_checkpoint_summary.json" \
  --out_md "$OUTPUT_ROOT/bicycle_v14_polish_checkpoint_summary.md"

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "v14 polish/checkpoint experiment finished."
