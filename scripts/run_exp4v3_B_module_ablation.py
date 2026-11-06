#!/usr/bin/env python3
"""v3 Group B: 模块消融实验 (逐项启用模块)

v3 计划的第二组实验: 在 1/4 物理分辨率下, 逐项启用 TAGS 的核心模块, 验证每个模块的独立贡献.
"除被消融的模块外, 其他设置保持一致, 才能判断每个模块的独立贡献."

实验组 (逐项启用, 与 §4.5 多模块消融的区别是更紧凑, 只关心 3 个核心模块):
  B.0 Raw3DGS          : 3DGS baseline
  B.1 + Geo only       : 3DGS + 几何先验
  B.2 + Geo + SPA      : 3DGS + 几何先验 + GaussianSpa
  B.3 + Geo + Staged   : 3DGS + 几何先验 + LPIPS staged
  B.4 Full TAGS        : 3DGS + 全部 3 个模块 (Geo + SPA + Staged)

这相当于把 §4.5 的 2^3 析因设计简化到 5 组线性"逐项启用"消融.
对比 Group A 的 4 方法, Group B 更细致地分解"3 个核心模块的独立贡献".

场景: 4 场景 (bicycle, garden, bonsai, room) × 3 seed = 60 runs (v3 用户决策 3 seed)
分辨率 (2026-10-17 修订, 详见 docs/SECTION_4_6_FULL_COMPARISON.md §1.0):
  与 Group A 完全一致: --images images_2 + --resolution 2 (1/4 物理像素)
  + resolution_schedule "1200:4,4000:2,30000:1" (warmup 1/16→1/8→1/4)

输出:
  output/exp4v3_B_{scene}/B.X_*/summary.json
  output/exp4_stats/exp4v3_B_module_ablation.csv
  docs/exp4v3_B_module_ablation.md

**算力预算 (2026-10-16 统一)**: 所有 5 个消融组 (B.0–B.4) 共享 `--max_gaussians 2,500,000`,
与 Group A 完全一致, 详见 `docs/SECTION_4_6_FULL_COMPARISON.md` §1.1.1.

用法:
  # 跑全部 60 runs
  python scripts/run_exp4v3_B_module_ablation.py

  # Smoke
  python scripts/run_exp4v3_B_module_ablation.py --scenes bicycle --groups B0 --seeds 0
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
        # 分辨率: --images images_2 (1/2 物理源图) + --resolution 2 (= 1/4 物理像素)
        # 见 docs/SECTION_4_6_FULL_COMPARISON.md §1.0
        "--images", "images_2",
        "--resolution", "2",
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


def _geo_args() -> List[str]:
    return [
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


def _spa_args() -> List[str]:
    return [
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


def _lpips_staged_args() -> List[str]:
    """分阶段优化策略 (LPIPS staged, §4.3 主张)."""
    return [
        "--enable_multiscale_lpips",
        "--lpips_loss_weight_init", str(LPIPS_INIT),
        "--lpips_loss_weight_final", str(LPIPS_FINAL),
        "--lpips_loss_start_iter", "0",
        "--lpips_loss_schedule", "staged",
        "--lpips_loss_staged_iters", "0:0.004,8000:0.001,13000:0.001,25000:0.0",
    ]


# 5 个组 (逐项启用, B.4 = Full TAGS)
GROUPS: Dict[str, Dict] = {
    "B0": {
        "name": "Raw3DGS",
        "description": "3DGS baseline (无任何 TAGS 模块)",
        "modules": {"geo": False, "spa": False, "staged": False},
    },
    "B1": {
        "name": "Geo_only",
        "description": "+ 几何先验 (depth + normal, 后期介入)",
        "modules": {"geo": True, "spa": False, "staged": False},
    },
    "B2": {
        "name": "Geo_SPA",
        "description": "+ 几何先验 + GaussianSpa 稀疏化",
        "modules": {"geo": True, "spa": True, "staged": False},
    },
    "B3": {
        "name": "Geo_Staged",
        "description": "+ 几何先验 + LPIPS staged 分阶段",
        "modules": {"geo": True, "spa": False, "staged": True},
    },
    "B4": {
        "name": "FullTAGS",
        "description": "Full TAGS (Geo + SPA + Staged 全开)",
        "modules": {"geo": True, "spa": True, "staged": True},
    },
}


def build_args(modules: Dict[str, bool], iterations: int = 30000) -> List[str]:
    """根据 modules 字典构造 train args."""
    args = list(_common_lr_args(iterations))

    # 几何先验
    if modules.get("geo"):
        args += _geo_args()
    else:
        args += ["--enable_geo_prior", "False"]

    # GaussianSpa
    if modules.get("spa"):
        args += _spa_args()
    else:
        args += ["--enable_sparse_optimization", "False", "--sparse_mode", "none", "--prune_ratio2", "0.0"]

    # LPIPS staged
    if modules.get("staged"):
        args += _lpips_staged_args()
    else:
        args += ["--lpips_loss_weight_init", "0.0", "--lpips_loss_weight_final", "0.0"]

    return args


def main():
    parser = argparse.ArgumentParser(description="v3 Group B: 模块消融实验 (1/2 分辨率)")
    parser.add_argument("--scenes", default="bicycle,garden,bonsai,room")
    parser.add_argument("--groups", default="B0,B1,B2,B3,B4", help="5 组逐项启用")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--images-dir", default="images_2",
                        help="v3: 源图目录 (1/2 物理, e.g. images_2). "
                             "实际训练像素由 --resolution 控制.")
    parser.add_argument("--resolution", type=int, default=2,
                        help="v3: 主阶段训练像素 = orig_w / resolution. "
                             "B 主实验默认 2 (配合 images_2 给出 1/4 物理像素).")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true",
                        help="跑完后写 module_ablation.csv")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    groups = [g.strip().upper() for g in args.groups.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group: {g}, available: {list(GROUPS)}")
    for s in scenes:
        if s not in SCENES_V3:
            raise SystemExit(f"v3 Group B 4 场景: {SCENES_V3} (got {s})")

    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(scenes) * len(groups) * len(seeds)
    done = 0
    t0 = time.time()
    for scene in scenes:
        for group in groups:
            cfg = GROUPS[group]
            extra_args = build_args(cfg["modules"], args.iterations)
            group_id = f"B.{group}_{cfg['name']}"
            for seed in seeds:
                summary_path = output_root / f"exp4v3_B_{scene}" / group_id / "summary.json"
                # 2026-10-17 修订: skip 判定要双重验证 (summary 有 PSNR + final PLY 真的在)
                if args.skip_existing and summary_path.is_file():
                    s = load_summary(summary_path)
                    seed_info = s.get("seeds", {}).get(str(seed), {})
                    final_ply = (output_root / f"exp4v3_B_{scene}" / group_id /
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
                print(f"\n[Group B {done}/{total}] {scene} {group_id} seed={seed}")
                if args.dry_run:
                    print(f"  DRY-RUN modules={cfg['modules']}")
                    continue
                info = run_full(
                    scene=scene,
                    group=group_id,
                    seed=seed,
                    extra_args=extra_args,
                    output_root=output_root / f"exp4v3_B_{scene}",
                    iterations=args.iterations,
                    python=args.python,
                    images_dir=args.images_dir,
                    resolution=args.resolution,
                )
                print(f"  PSNR={info.get('psnr')} N={info.get('num_points')}")
    elapsed = time.time() - t0
    print(f"\n[v3 Group B] {done} runs done in {elapsed/60:.1f} min")

    if args.summarize:
        summarize_group_b(output_root)


def summarize_group_b(output_root: Path) -> None:
    """汇总 Group B 的结果, 写 module_ablation.csv."""
    stats_root = output_root / "exp4_stats"
    stats_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in SCENES_V3:
        for group_id, cfg in GROUPS.items():
            full_id = f"B.{group_id}_{cfg['name']}"
            summary_path = output_root / f"exp4v3_B_{scene}" / full_id / "summary.json"
            if not summary_path.is_file():
                continue
            summary = load_summary(summary_path)
            seeds_data = summary.get("seeds", {})
            for seed_str, info in seeds_data.items():
                rows.append({
                    "scene": scene,
                    "group": full_id,
                    "ablation": cfg["name"],
                    "geo": cfg["modules"]["geo"],
                    "spa": cfg["modules"]["spa"],
                    "staged": cfg["modules"]["staged"],
                    "seed": int(seed_str),
                    "psnr": info.get("psnr"),
                    "ssim": info.get("ssim"),
                    "lpips": info.get("lpips"),
                    "num_points": info.get("num_points"),
                })

    if not rows:
        print("[v3 Group B] No summary data found.")
        return

    # 写汇总 (跨场景主表 + per-scene 详细)
    csv_path = stats_root / "exp4v3_B_module_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        # 写表头
        f.write("ablation,scene,geo,spa,staged,n_seeds,psnr_mean,psnr_std,ssim_mean,ssim_std,lpips_mean,lpips_std,num_points_mean\n")
        for ablation in sorted(set(r["ablation"] for r in rows), key=lambda x: list(GROUPS.keys()).index(next(k for k, v in GROUPS.items() if v["name"] == x))):
            for scene in SCENES_V3:
                scene_rows = [r for r in rows if r["ablation"] == ablation and r["scene"] == scene]
                if not scene_rows:
                    continue
                psnrs = [r["psnr"] for r in scene_rows if r["psnr"] is not None]
                ssims = [r["ssim"] for r in scene_rows if r["ssim"] is not None]
                lpipss = [r["lpips"] for r in scene_rows if r["lpips"] is not None]
                pts = [r["num_points"] for r in scene_rows if r["num_points"] is not None]
                first = scene_rows[0]
                f.write(",".join([
                    ablation,
                    scene,
                    str(first["geo"]),
                    str(first["spa"]),
                    str(first["staged"]),
                    str(len(scene_rows)),
                    f"{statistics.mean(psnrs):.3f}" if psnrs else "n/a",
                    f"{statistics.stdev(psnrs):.3f}" if len(psnrs) > 1 else "0.000",
                    f"{statistics.mean(ssims):.4f}" if ssims else "n/a",
                    f"{statistics.stdev(ssims):.4f}" if len(ssims) > 1 else "0.0000",
                    f"{statistics.mean(lpipss):.4f}" if lpipss else "n/a",
                    f"{statistics.stdev(lpipss):.4f}" if len(lpipss) > 1 else "0.0000",
                    f"{statistics.mean(pts):.0f}" if pts else "n/a",
                ]) + "\n")
    print(f"[v3 Group B] Wrote {csv_path} ({len(rows)} raw rows)")


if __name__ == "__main__":
    main()
