#!/usr/bin/env python3
"""v3 标准分辨率验证 (1/2 物理像素 × 完整 TAGSs × 3 seed = 9 runs)

v3 计划补充实验: 在主实验 (1/4 物理像素, --resolution 2) 完成后, 挑选 2-3 个
具有代表性的 Mip-NeRF 360 场景, 用完整 TAGSs 在 1/2 物理像素下训练与评估,
验证 1/4 物理像素得出的结论是否能扩展到 1/2 物理像素.

代表性场景选取:
  - bicycle (户外复杂, 论文主表 1)
  - bonsai  (室内复杂, 论文主表 2)
  - garden  (户外中等, 论文主表 3)

完整 TAGSs 配置: 几何先验 + GaussianSpa + LPIPS staged + 多分辨率

**2026-10-17 关键修正** (详见 docs/SECTION_4_6_FULL_COMPARISON.md §1.0):
  旧版 std 验证用 --images images + --resolution -1 (默认) → 实际 1600px.
  现改为 --images images_2 + --resolution 1 (= 2473px = 1/2 物理像素),
  与 C.3 模式完全对齐, 跨验证可读.

输出:
  output/exp4v3_std_{scene}/Std_FullTAGS/summary.json
  output/exp4_stats/exp4v3_std_resolution_validation.csv
  docs/exp4v3_std_resolution_validation.md

**算力预算 (2026-10-16 统一)**: 与 Group A/B/C 共享 `--max_gaussians 2,500,000`,
详见 `docs/SECTION_4_6_FULL_COMPARISON.md` §1.1.1. 若 1/2 物理像素 OOM, 见
EXP4_RUN_BOOK §11 处理 (降 budget 到 1.5M 或仅 1 seed).

用法:
  python scripts/run_exp4v3_std_resolution_validation.py
  python scripts/run_exp4v3_std_resolution_validation.py --scenes bicycle --seeds 0
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

# §4.2/§4.3/§4.4 选定的最终参数
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

# v3 选定的代表性场景 (bicycle + 1 室内 + 1 户外)
DEFAULT_SCENES = ["bicycle", "bonsai", "garden"]


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
        # 2026-10-17 修订: --images images_2 + --resolution 1 (= 1/2 物理像素)
        # 与 C.3 模式对齐, 跨验证可读
        "--images", "images_2",
        "--resolution", "1",
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
    """完整 TAGS 配置 (Full TAGS) 用于标准分辨率验证."""
    args = _common_lr_args(iterations)
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
    args += [
        "--enable_multiscale_lpips",
        "--lpips_loss_weight_init", str(LPIPS_INIT),
        "--lpips_loss_weight_final", str(LPIPS_FINAL),
        "--lpips_loss_start_iter", "0",
        "--lpips_loss_schedule", "staged",
        "--lpips_loss_staged_iters", "0:0.004,8000:0.001,13000:0.001,25000:0.0",
    ]
    args += [
        "--enable_resolution_aware_schedule",
        "--resolution_schedule", f"1200:4,4000:2,{iterations}:1",
    ]
    return args


def main():
    parser = argparse.ArgumentParser(description="v3 标准分辨率验证 (2-3 场景 × 完整 TAGS × 3 seed)")
    parser.add_argument("--scenes", default="bicycle,bonsai,garden",
                        help="2-3 场景: bicycle,bonsai,garden (推荐)")
    parser.add_argument("--seeds", default="0,1,2", help="3 seed")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--images-dir", default="images_2",
                        help="2026-10-17 统一为 images_2 (1/2 物理源图). "
                             "实际像素由 --resolution 控制 (默认 1 = 1/2 物理像素).")
    parser.add_argument("--resolution", type=int, default=1,
                        help="v3 std 验证主阶段训练像素 = orig_w / resolution. "
                             "默认 1 (配合 images_2 给出 1/2 物理像素, = C.3 模式).")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for s in scenes:
        if s not in ("bicycle", "garden", "bonsai", "room"):
            raise SystemExit(f"v3 标准分辨率仅支持 mip360 场景: bicycle/garden/bonsai/room (got {s})")

    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(scenes) * len(seeds)
    done = 0
    t0 = time.time()
    for scene in scenes:
        for seed in seeds:
            group_id = f"Std_FullTAGS"
            summary_path = output_root / f"exp4v3_std_{scene}" / group_id / "summary.json"
            # 2026-10-17 修订: skip 判定要双重验证 (summary 有 PSNR + final PLY 真的在)
            if args.skip_existing and summary_path.is_file():
                s = load_summary(summary_path)
                seed_info = s.get("seeds", {}).get(str(seed), {})
                final_ply = (output_root / f"exp4v3_std_{scene}" / group_id /
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
            print(f"\n[v3 std {done}/{total}] {scene} {group_id} "
                  f"(images={args.images_dir}, resolution={args.resolution}) seed={seed}")
            if args.dry_run:
                print(f"  DRY-RUN images_dir={args.images_dir}, resolution={args.resolution}")
                continue
            info = run_full(
                scene=scene,
                group=group_id,
                seed=seed,
                extra_args=full_tags_args(args.iterations),
                output_root=output_root / f"exp4v3_std_{scene}",
                iterations=args.iterations,
                python=args.python,
                images_dir=args.images_dir,
                resolution=args.resolution,
            )
            print(f"  PSNR={info.get('psnr')} N={info.get('num_points')}")
    elapsed = time.time() - t0
    print(f"\n[v3 std] {done} runs done in {elapsed/60:.1f} min")

    if args.summarize:
        summarize_std(output_root)


def summarize_std(output_root: Path) -> None:
    """汇总标准分辨率验证结果."""
    stats_root = output_root / "exp4_stats"
    stats_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in DEFAULT_SCENES:
        summary_path = output_root / f"exp4v3_std_{scene}" / "Std_FullTAGS" / "summary.json"
        if not summary_path.is_file():
            continue
        summary = load_summary(summary_path)
        seeds_data = summary.get("seeds", {})
        for seed_str, info in seeds_data.items():
            rows.append({
                "scene": scene,
                "seed": int(seed_str),
                "resolution": "std",
                "images_dir": "images",
                "psnr": info.get("psnr"),
                "ssim": info.get("ssim"),
                "lpips": info.get("lpips"),
                "num_points": info.get("num_points"),
            })

    if not rows:
        print("[v3 std] No summary data found.")
        return

    # 写汇总
    csv_path = stats_root / "exp4v3_std_resolution_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        f.write("scene,resolution,images_dir,n_seeds,psnr_mean,psnr_std,ssim_mean,ssim_std,lpips_mean,lpips_std,num_points_mean\n")
        for scene in DEFAULT_SCENES:
            scene_rows = [r for r in rows if r["scene"] == scene]
            if not scene_rows:
                continue
            psnrs = [r["psnr"] for r in scene_rows if r["psnr"] is not None]
            ssims = [r["ssim"] for r in scene_rows if r["ssim"] is not None]
            lpipss = [r["lpips"] for r in scene_rows if r["lpips"] is not None]
            pts = [r["num_points"] for r in scene_rows if r["num_points"] is not None]
            f.write(",".join([
                scene,
                "std",
                "images",
                str(len(scene_rows)),
                f"{statistics.mean(psnrs):.3f}" if psnrs else "n/a",
                f"{statistics.stdev(psnrs):.3f}" if len(psnrs) > 1 else "0.000",
                f"{statistics.mean(ssims):.4f}" if ssims else "n/a",
                f"{statistics.stdev(ssims):.4f}" if len(ssims) > 1 else "0.0000",
                f"{statistics.mean(lpipss):.4f}" if lpipss else "n/a",
                f"{statistics.stdev(lpipss):.4f}" if len(lpipss) > 1 else "0.0000",
                f"{statistics.mean(pts):.0f}" if pts else "n/a",
            ]) + "\n")
    print(f"[v3 std] Wrote {csv_path} ({len(rows)} raw rows)")


if __name__ == "__main__":
    main()
