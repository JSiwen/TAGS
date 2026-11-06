#!/usr/bin/env python3
"""v3 Group C: 分辨率敏感性实验 (1/16 vs 1/4 vs 1/2 物理像素, 完整 TAGS)

v3 计划的第三组实验: 完整 TAGSs 在 3 种物理像素分辨率下分别训练与评估.
观察分辨率变化对:
  - 重建质量 (PSNR/SSIM/LPIPS)
  - 资源消耗 (训练时间, 峰值显存)
  - 稀疏化效果 (最终高斯数, 模型大小)

实验组 (3 种物理像素, 完整 TAGSs 配置, 统一 --images images_2):
  C.1 Res_1_16 : --resolution 8 (主阶段 1/16 物理像素, 309×206 for bicycle)
  C.2 Res_1_4  : --resolution 2 (主阶段 1/4 物理像素,  1236×822 for bicycle) — 与 A/B 一致
  C.3 Res_1_2  : --resolution 1 (主阶段 1/2 物理像素,  2473×1643 for bicycle)

场景: 4 场景 (bicycle, garden, bonsai, room) × 3 seed (0,1,2) = 36 runs
(全分辨率实验 seed 减半, 平衡工作量)

**2026-10-17 关键修正** (详见 docs/SECTION_4_6_FULL_COMPARISON.md §1.0):
  旧版 v3 用 --images images_2/4/ 切换源图目录 + --resolution -1 (默认),
  实际上所有 --images 配置都被 args.resolution=-1 的 "global_down=1600px" 上限
  拉到 1600px, 导致 C.1/C.2/C.3 实际差异 < 100 像素, 失去分辨率敏感性意义.
  改用 --resolution 显式控制 (8/2/1) 真正区分三档物理像素, 配合
  --images 统一为 images_2 (避免 baseline dataset 准备不一致).

输出:
  output/exp4v3_C_{scene}/C.X_*/summary.json
  output/exp4_stats/exp4v3_C_resolution_sensitivity.csv
  docs/exp4v3_C_resolution_sensitivity.md

**算力预算 (2026-10-16 统一)**: 所有 3 个分辨率组 (C.1/C.2/C.3) 共享 `--max_gaussians 2,500,000`,
与 Group A/B 完全一致, 详见 `docs/SECTION_4_6_FULL_COMPARISON.md` §1.1.1.
C.3 1/2 物理若显存吃紧会自动 raise, 届时调低 budget (见 EXP4_RUN_BOOK §11).

用法:
  python scripts/run_exp4v3_C_resolution_sensitivity.py
  python scripts/run_exp4v3_C_resolution_sensitivity.py --scenes bicycle --groups C.2 --seeds 0
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp4_runner import run_full, load_summary  # type: ignore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# §4.2/§4.3/§4.4 选定的最终参数 (Full TAGS)
PRUNE_RATIO2 = 0.04
LPIPS_INIT = 0.004
LPIPS_FINAL = 0.001

# 几何窗口 (T3: 后期精修)
GEO_START = 26000
GEO_WARMUP = 30000
GEO_DECAY_START = 26000
GEO_DECAY_UNTIL = 30000
DEPTH_FINAL = 0.05
NORMAL_FINAL = 0.02

SCENES_V3 = ["bicycle", "garden", "bonsai", "room"]


def _common_lr_args(iterations: int = 30000) -> List[str]:
    return [
        "--data_device", "cpu",
        "--eval",
        "--llffhold", "8",
        "--lambda_dssim", "0.2",
        "--densification_interval", "100",
        "--opacity_reset_interval", "3000",
        "--densify_from_iter", "500",
        "--densify_until_iter", "18000",
        "--densify_until_fraction", "0.50",
        "--densify_grad_threshold", "0.00016",
        # 统一 --images images_2; --resolution 由 GROUPS 配置决定 (1/2/8)
        # 详见 docs/SECTION_4_6_FULL_COMPARISON.md §1.0 (2026-10-17 修订)
        "--images", "images_2",
        "--max_gaussians", "2500000",
        "--prune_opacity_threshold", "0.02",
        "--prune_min_visibility", "0",
        "--prune_max_screen_size", "-1",
        "--prune_grad_threshold", "0.0",
        "--prune_grad_min_denom", "1.0",
        "--prune_only_from_fraction", "0.50",
        "--prune_until_fraction", "1.0",
        "--position_lr_init", "0.00016",
        "--position_lr_final", "0.0000016",
        "--position_lr_delay_mult", "0.01",
        "--position_lr_max_steps", str(iterations),
        "--feature_lr", "0.0025",
        "--opacity_lr", "0.025",
        "--scaling_lr", "0.005",
        "--rotation_lr", "0.001",
        "--exposure_lr_init", "0.0",
        "--exposure_lr_final", "0.0",
        "--percent_dense", "0.01",
        "--enable_adaptive_stage_controller", "False",
    ]


def full_tags_args(iterations: int = 30000) -> List[str]:
    """完整 TAGS 配置 (Full TAGS): Geo + SPA + LPIPS staged + 多分辨率."""
    args = _common_lr_args(iterations)
    # 几何先验
    args += [
        "--enable_geo_prior",
        "--geo_prior_depth",
        "--geo_prior_normal",
        "--depth_prior_loss_type", "edge_aware_log_l1",
        "--normal_prior_loss_type", "cosine",
        "--depth_prior_pearson_weight", "0.0",
        "--depth_prior_local_pearson_weight", "0.05",
        "--normal_prior_tv_weight", "0.003",
        "--depth_prior_edge_quantile", "0.92",
        "--depth_prior_rgb_edge_quantile", "0.95",
        "--depth_prior_var_quantile", "0.92",
        "--depth_prior_min_mask_fraction", "0.20",
        "--normal_tv_depth_edge_quantile", "0.70",
        "--normal_tv_rgb_edge_quantile", "0.70",
        "--geo_prior_start_iter", str(GEO_START),
        "--geo_prior_warmup_until", str(GEO_WARMUP),
        "--geo_prior_decay_start", str(GEO_DECAY_START),
        "--geo_prior_decay_until", str(GEO_DECAY_UNTIL),
        "--depth_prior_weight_init", "0.04",
        "--depth_prior_weight_final", str(DEPTH_FINAL),
        "--depth_prior_weight_min", "0.01",
        "--normal_prior_weight_init", "0.02",
        "--normal_prior_weight_final", str(NORMAL_FINAL),
        "--normal_prior_weight_min", "0.0",
        "--geo_prior_fullres_relax_max", "0.35",
    ]
    # GaussianSpa
    args += [
        "--enable_sparse_optimization",
        "--sparse_mode", "spa",
        "--prune_ratio2", str(PRUNE_RATIO2),
        "--spa_start_iter", "13000",
        "--spa_end_iter", "25000",
        "--spa_interval", "1000",
        "--rho_lr", "0.0005",
        "--spa_metric", "hybrid",
        "--spa_require_full_resolution", "True",
        "--spa_after_fullres_delay", "3000",
        "--spa_max_prune_points", "30000",
        "--spa_protect_top_fraction", "0.05",
    ]
    # LPIPS staged
    args += [
        "--enable_multiscale_lpips",
        "--lpips_loss_weight_init", str(LPIPS_INIT),
        "--lpips_loss_weight_final", str(LPIPS_FINAL),
        "--lpips_loss_start_iter", "0",
        "--lpips_loss_schedule", "staged",
        "--lpips_loss_staged_iters", "0:0.004,8000:0.001,13000:0.001,25000:0.0",
    ]
    # 多分辨率调度
    args += [
        "--enable_resolution_aware_schedule",
        "--resolution_schedule", f"1200:4,4000:2,{iterations}:1",
    ]
    return args


# 3 个分辨率组 (全部为 Full TAGS, 通过 --resolution 区分物理像素)
# 2026-10-17 修订: 统一 --images images_2, --resolution 8/2/1 控制物理像素
GROUPS: Dict[str, Dict] = {
    "C.1": {
        "name": "Res_1_16",
        "resolution": 8,  # 主阶段 1/16 物理像素, bicycle = 2473/8 = 309 px
        "description": "1/16 物理像素 (309×206 for bicycle) + 完整 TAGS",
    },
    "C.2": {
        "name": "Res_1_4",
        "resolution": 2,  # 主阶段 1/4 物理像素, bicycle = 2473/2 = 1236 px (与 A/B 一致)
        "description": "1/4 物理像素 (1236×822 for bicycle) + 完整 TAGS",
    },
    "C.3": {
        "name": "Res_1_2",
        "resolution": 1,  # 主阶段 1/2 物理像素, bicycle = 2473/1 = 2473 px
        "description": "1/2 物理像素 (2473×1643 for bicycle) + 完整 TAGS",
    },
}


def main():
    parser = argparse.ArgumentParser(description="v3 Group C: 分辨率敏感性实验")
    parser.add_argument("--scenes", default="bicycle,garden,bonsai,room")
    parser.add_argument("--groups", default="C.1,C.2,C.3", help="3 分辨率组")
    parser.add_argument("--seeds", default="0,1,2", help="3 seed (全分辨率实验 seed 减半)")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group: {g}, available: {list(GROUPS)}")
    for s in scenes:
        if s not in SCENES_V3:
            raise SystemExit(f"v3 Group C 4 场景: {SCENES_V3} (got {s})")

    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(scenes) * len(groups) * len(seeds)
    done = 0
    t0 = time.time()
    for scene in scenes:
        for group in groups:
            cfg = GROUPS[group]
            resolution = cfg["resolution"]
            extra_args = full_tags_args(args.iterations)
            group_id = f"{group}_{cfg['name']}"
            for seed in seeds:
                summary_path = output_root / f"exp4v3_C_{scene}" / group_id / "summary.json"
                # 2026-10-17 修订: skip 判定要双重验证 (summary 有 PSNR + final PLY 真的在)
                if args.skip_existing and summary_path.is_file():
                    s = load_summary(summary_path)
                    seed_info = s.get("seeds", {}).get(str(seed), {})
                    final_ply = (output_root / f"exp4v3_C_{scene}" / group_id /
                                 "train" / f"seed{seed}" / "point_cloud" /
                                 f"iteration_{args.iterations}" / "point_cloud.ply")
                    if (seed_info.get("psnr") is not None
                            and final_ply.is_file()):
                        print(f"[skip] {scene} {group_id} seed={seed} (final ply 在)")
                        done += 1
                        continue
                    elif seed_info.get("psnr") is not None and not final_ply.is_file():
                        print(f"[stale] {scene} {group_id} seed={seed}: summary 有 PSNR 但 final ply 不在, 重跑")
                done += 1
                print(f"\n[Group C {done}/{total}] {scene} {group_id} (resolution={resolution}) seed={seed}")
                if args.dry_run:
                    print(f"  DRY-RUN resolution={resolution}")
                    continue
                info = run_full(
                    scene=scene,
                    group=group_id,
                    seed=seed,
                    extra_args=extra_args,
                    output_root=output_root / f"exp4v3_C_{scene}",
                    iterations=args.iterations,
                    python=args.python,
                    images_dir="images_2",  # 2026-10-17 统一: 全部用 images_2
                    resolution=resolution,
                )
                print(f"  PSNR={info.get('psnr')} N={info.get('num_points')}")
    elapsed = time.time() - t0
    print(f"\n[v3 Group C] {done} runs done in {elapsed/60:.1f} min")

    if args.summarize:
        summarize_group_c(output_root)


def summarize_group_c(output_root: Path) -> None:
    """汇总 Group C 的分辨率敏感性结果."""
    stats_root = output_root / "exp4_stats"
    stats_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in SCENES_V3:
        for group_id, cfg in GROUPS.items():
            full_id = f"{group_id}_{cfg['name']}"
            summary_path = output_root / f"exp4v3_C_{scene}" / full_id / "summary.json"
            if not summary_path.is_file():
                continue
            summary = load_summary(summary_path)
            seeds_data = summary.get("seeds", {})
            for seed_str, info in seeds_data.items():
                rows.append({
                    "scene": scene,
                    "group": full_id,
                    "resolution": cfg["name"],
                    "resolution_arg": cfg["resolution"],
                    "seed": int(seed_str),
                    "psnr": info.get("psnr"),
                    "ssim": info.get("ssim"),
                    "lpips": info.get("lpips"),
                    "num_points": info.get("num_points"),
                })

    if not rows:
        print("[v3 Group C] No summary data found.")
        return

    # 写汇总
    csv_path = stats_root / "exp4v3_C_resolution_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        f.write("scene,resolution,resolution_arg,n_seeds,psnr_mean,psnr_std,ssim_mean,ssim_std,lpips_mean,lpips_std,num_points_mean\n")
        for scene in SCENES_V3:
            for group_id, cfg in GROUPS.items():
                scene_rows = [r for r in rows if r["scene"] == scene and r["resolution"] == cfg["name"]]
                if not scene_rows:
                    continue
                psnrs = [r["psnr"] for r in scene_rows if r["psnr"] is not None]
                ssims = [r["ssim"] for r in scene_rows if r["ssim"] is not None]
                lpipss = [r["lpips"] for r in scene_rows if r["lpips"] is not None]
                pts = [r["num_points"] for r in scene_rows if r["num_points"] is not None]
                f.write(",".join([
                    scene,
                    cfg["name"],
                    str(cfg["resolution"]),
                    str(len(scene_rows)),
                    f"{statistics.mean(psnrs):.3f}" if psnrs else "n/a",
                    f"{statistics.stdev(psnrs):.3f}" if len(psnrs) > 1 else "0.000",
                    f"{statistics.mean(ssims):.4f}" if ssims else "n/a",
                    f"{statistics.stdev(ssims):.4f}" if len(ssims) > 1 else "0.0000",
                    f"{statistics.mean(lpipss):.4f}" if lpipss else "n/a",
                    f"{statistics.stdev(lpipss):.4f}" if len(lpipss) > 1 else "0.0000",
                    f"{statistics.mean(pts):.0f}" if pts else "n/a",
                ]) + "\n")
    print(f"[v3 Group C] Wrote {csv_path} ({len(rows)} raw rows)")


if __name__ == "__main__":
    main()
