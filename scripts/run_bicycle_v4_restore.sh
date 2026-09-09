#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE="${SOURCE:-/root/highspeedstorage/data/bicycle}"
DEPTHS="${DEPTHS:-/root/highspeedstorage/data/bicycle/depth_any_out}"
MODEL="${MODEL:-/root/highspeedstorage/TAGSs/output/bicycle_integrated_v4_restore}"
ITERATIONS="${ITERATIONS:-30000}"

TAGGS_CPU_KNN_INIT="${TAGGS_CPU_KNN_INIT:-1}" python train_optimized.py \
  -s "$SOURCE" \
  -m "$MODEL" \
  --images images \
  --depths "$DEPTHS" \
  --resolution -1 \
  --data_device cpu \
  --eval \
  --llffhold 8 \
  --iterations "$ITERATIONS" \
  --position_lr_init 0.00016 \
  --position_lr_final 0.0000016 \
  --position_lr_delay_mult 0.01 \
  --position_lr_max_steps "$ITERATIONS" \
  --feature_lr 0.0025 \
  --opacity_lr 0.025 \
  --scaling_lr 0.005 \
  --rotation_lr 0.001 \
  --exposure_lr_init 0.01 \
  --exposure_lr_final 0.001 \
  --exposure_lr_delay_steps 0 \
  --exposure_lr_delay_mult 0.0 \
  --percent_dense 0.01 \
  --lambda_dssim 0.2 \
  --densification_interval 100 \
  --opacity_reset_interval 3000 \
  --densify_from_iter 500 \
  --densify_until_iter 18000 \
  --densify_grad_threshold 0.00016 \
  --depth_l1_weight_init 1.0 \
  --depth_l1_weight_final 0.01 \
  --enable_geo_prior \
  --geo_prior_depth \
  --geo_prior_normal \
  --depth_prior_loss_type edge_aware_log_l1 \
  --normal_prior_loss_type cosine \
  --depth_prior_pearson_weight 0.0 \
  --depth_prior_local_pearson_weight 0.05 \
  --normal_prior_tv_weight 0.0015 \
  --depth_prior_edge_quantile 0.92 \
  --depth_prior_rgb_edge_quantile 0.95 \
  --depth_prior_var_quantile 0.92 \
  --depth_prior_min_mask_fraction 0.2 \
  --normal_tv_depth_edge_quantile 0.7 \
  --normal_tv_rgb_edge_quantile 0.7 \
  --depth_prior_weight_init 0.0 \
  --depth_prior_weight_final 0.04 \
  --depth_prior_weight_min 0.01 \
  --normal_prior_weight_init 0.0 \
  --normal_prior_weight_final 0.012 \
  --normal_prior_weight_min 0.0 \
  --geo_prior_low_texture_relax_strength 0.0 \
  --geo_prior_low_texture_relax_min 0.90 \
  --geo_prior_warmup_until 4000 \
  --geo_prior_decay_start 10000 \
  --geo_prior_decay_until 20000 \
  --detail_loss_weight_init 0.0 \
  --detail_loss_weight_final 0.03 \
  --detail_loss_start_iter 4000 \
  --detail_loss_edge_quantile 0.75 \
  --detail_loss_max_weight 0.05 \
  --enable_hard_view_loss_reweight \
  --hard_view_loss_reweight_start_iter 18000 \
  --hard_view_loss_reweight_max 1.12 \
  --color_align_weight_init 0.0 \
  --color_align_weight_final 0.002 \
  --color_align_start_iter 24000 \
  --color_align_max_weight 0.006 \
  --color_align_std_weight 0.5 \
  --color_align_hard_view_boost 1.0 \
  --color_align_low_grad_quantile 0.60 \
  --color_align_min_mask_fraction 0.20 \
  --color_align_require_fullres \
  --enable_resolution_adaptive \
  --resolution_schedule "1200:4,4000:2,30000:1" \
  --enable_sparse_optimization \
  --sparse_mode spa \
  --spa_interval 1000 \
  --spa_start_iter 13000 \
  --spa_end_iter 25000 \
  --spa_prune_ratio 0.05 \
  --prune_ratio2 0.04 \
  --rho_lr 0.0005 \
  --spa_metric hybrid \
  --spa_require_full_resolution \
  --spa_after_fullres_delay 3000 \
  --spa_max_prune_points 30000 \
  --spa_protect_top_fraction 0.05 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000 \
  --disable_viewer

python render.py -m "$MODEL" --skip_train --quiet
python metrics.py -m "$MODEL" --split test
