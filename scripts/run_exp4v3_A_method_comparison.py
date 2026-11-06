#!/usr/bin/env python3
"""v3 Group A: 主要方法对比 (4 方法 × 4 场景 × 3 seed = 48 runs)

v3 计划的第一组实验: 在 1/2 物理分辨率下对比 4 种代表性方法, 输出 PSNR/SSIM/LPIPS,
训练时间, 显存, 最终高斯数量, 模型大小.

实验组 (4 方法):
  A.1 Raw3DGS    : 原始 3DGS, 无任何 TAGS 模块
  A.2 TAGS_Geo   : 3DGS + 几何先验 (depth + normal)
  A.3 TAGS_SPA   : 3DGS + GaussianSpa 稀疏化
  A.4 Full_TAGS  : 3DGS + 几何先验 + GaussianSpa + 分阶段优化 + 多分辨率

场景 (4 个, 覆盖户外/室内):
  bicycle (户外复杂), garden (户外中等), bonsai (室内复杂), room (室内中等)

3 seed: 0, 1, 2 (v3 用户决策, 节省 37% runs)

分辨率策略 (2026-10-17 修订, 详见 docs/SECTION_4_6_FULL_COMPARISON.md §1.0):
  --images images_2 (1/2 物理源图) + --resolution 2 (= 源图/2 = 1/4 物理像素)
  + resolution_schedule "1200:4,4000:2,30000:1" (1/16 → 1/8 → 1/4 物理)
  训练阶段像素 (bicycle):
    0-1200  iter: 309 × 206   (warmup,  1/16 物理)
    1200-4000 iter: 618 × 411  (warmup,  1/8 物理)
    4000-30000 iter: 1236 × 822 (主阶段, 1/4 物理)
  这与原 v2 1/4 训练在视觉质量上对齐, 但算力比 1/2 主阶段省 4×.

输出:
  output/exp4v3_A_{scene}/A.X_*/summary.json
  output/exp4_stats/exp4v3_A_method_comparison.csv (跨场景主表)
  output/exp4_stats/exp4v3_A_per_scene.csv (per-scene 详细)
  docs/exp4v3_A_method_comparison.md (论文表 4-X 主要方法对比)

**算力预算 (2026-10-16 统一)**: 所有 4 个方法组共享 `--max_gaussians 2,500,000` 硬上限
(走 `_common_lr_args()` 统一传入), 让对比在"同等算力"下隔离模块质量贡献.
详见 `docs/SECTION_4_6_FULL_COMPARISON.md` §1.1.1.

用法:
  # 跑全部 48 runs
  python scripts/run_exp4v3_A_method_comparison.py

  # 跑单场景单组
  python scripts/run_exp4v3_A_method_comparison.py --scenes bicycle --groups A4

  # Smoke
  python scripts/run_exp4v3_A_method_comparison.py --scenes bicycle --groups A1 --seeds 0
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp4_runner import run_full, load_summary  # type: ignore  # noqa: E402

# train_optimized.py 的 enable_* / geo_prior_* / spa_require_full_resolution / color_align_*
# 都是 argparse 的 store_true 类型, 不该带 True/False 值. 但 v2 runner 把 bool 写成
# "--flag True/False", argparse 解析时报 "unrecognized arguments: False True".
# 这里对 extra_args 做一次清洗: 对已知 store_true flag, 把紧跟的 True/False 值吃掉.
_STORE_TRUE_FLAGS = frozenset({
    "--enable_geo_prior", "--enable_resolution_adaptive", "--enable_sparse_optimization",
    "--geo_prior_depth", "--geo_prior_normal",
    "--enable_multiscale_lpips", "--enable_resolution_aware_schedule",
    "--enable_adaptive_stage_controller", "--spa_require_full_resolution",
})


def _sanitize_bool_flags(args_list: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(args_list):
        tok = args_list[i]
        out.append(tok)
        if tok in _STORE_TRUE_FLAGS and i + 1 < len(args_list):
            nxt = args_list[i + 1]
            if nxt in ("True", "False", "true", "false"):
                # 跳过这个值
                i += 2
                continue
        i += 1
    return out


REPO_ROOT = Path(__file__).resolve().parents[1]

# §4.2/§4.3/§4.4 选定的最终参数 (临时, 实际跑完后回填)
PRUNE_RATIO2 = 0.04
LPIPS_INIT = 0.004
LPIPS_FINAL = 0.001

# 几何窗口 (T3: 后期精修, 论文主张)
GEO_START = 26000
GEO_WARMUP = 30000
GEO_DECAY_START = 26000
GEO_DECAY_UNTIL = 30000
DEPTH_FINAL = 0.05
NORMAL_FINAL = 0.02

SCENES_V3 = ["bicycle", "garden", "bonsai", "room"]

# v3 Group A: max_gaussians 改为模块级, 让 main() 通过 CLI 注入 (A 改动, 2026-10-19)
_MAX_GAUSSIANS_DEFAULT = 2500000
_MAX_GAUSSIANS_OPEN = 6000000  # bonsai/room 验证 SPA 时用, 让 SPA 真正稀疏化


def _common_lr_args(
    iterations: int = 30000,
    max_gaussians: int = _MAX_GAUSSIANS_DEFAULT,
    images_dir: str = "images_2",
    resolution: int = 2,
) -> List[str]:
    """所有组共享的学习率/调度参数.

    max_gaussians:
      2500000 (默认) - 80-run 主实验, 所有方法撞顶, 隔离模块质量
      6000000        - SPA 验证实验, 放开上限让 SPA 真正稀疏化

    images_dir / resolution:
      bicycle/garden (16MP 原图): images_2 + res=2 → 主阶段 1/4 物理 = ~1MP
      bonsai/room    ( 6MP 原图): images   + res=1 → 主阶段 full-res = ~6.4MP
    """
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
        # 分辨率: 默认 --images images_2 + --resolution 2 (= 1/4 物理像素)
        # Scene.loadCam: resolution = orig_w / (global_down * resolution_scale)
        # bonsai/room 满分辨率时可由 main() 覆盖为 images + res=1
        "--images", images_dir,
        "--resolution", str(resolution),
        "--max_gaussians", str(max_gaussians),
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
    """几何先验 (depth + normal), 后期介入 (26000-30000)."""
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


def _spa_args(start: int = 7000, end: int = 28000, ratio: float = PRUNE_RATIO2) -> List[str]:
    """GaussianSpa 稀疏化 (方案1修复 2026-10-20).

    历史默认值 (13000/25000/30000 cap) 导致 bonsai 上 SPA 路径被卡死:
      * spa_max_prune_points=30000 硬上限, 每次事件期望剪 0.04*N≈51k 实际只剪 30k (59%)
      * spa_start=13000 配合 fullres(4000)+delay(3000) → 窗口 12k iter, 但
        AdaptiveStage 把窗口挪到 17500-20200 → 实际 ~2k iter (1-2 次事件)
      * 净效应: SPA 真正删点 60-90k, 对 1.2M 高斯几乎无效
    修复:
      * spa_max_prune_points 设为 0 (=关闭 cap, 信任 prune_ratio2)
      * spa_start=7000 (fullres 4000 + delay 3000)
      * spa_end=28000 (给 densify 收尾留 2k iter 磨合)
      * rho_lr 0.0001 (原 0.0005 过大, 把 opacity 压太狠, 触发原生剪枝)
    """
    return [
        "--enable_sparse_optimization",
        "--sparse_mode", "spa",
        "--prune_ratio2", str(ratio),
        "--spa_start_iter", str(start),
        "--spa_end_iter", str(end),
        "--spa_interval", "1000",
        "--rho_lr", "0.0001",
        "--spa_metric", "hybrid",
        "--spa_require_full_resolution", "True",
        "--spa_after_fullres_delay", "3000",
        "--spa_max_prune_points", "0",
        "--spa_protect_top_fraction", "0.05",
    ]


def _lpips_staged_args() -> List[str]:
    """LPIPS 分阶段调度 (TAGS staged)."""
    return [
        "--enable_multiscale_lpips",
        "--lpips_loss_weight_init", str(LPIPS_INIT),
        "--lpips_loss_weight_final", str(LPIPS_FINAL),
        "--lpips_loss_start_iter", "0",
        "--lpips_loss_schedule", "staged",
        "--lpips_loss_staged_iters", "0:0.004,8000:0.001,13000:0.001,25000:0.0",
    ]


def _resolution_aware_args(iterations: int = 30000) -> List[str]:
    """多分辨率调度 (Scene 实际语义).

    Scene.loadCam 公式: ``resolution = orig_w / (global_down * resolution_scale)``,
    其中 ``global_down = orig_w / args.resolution``. 在 --images images_2
    (orig_w=2473 for bicycle) + --resolution 2 的配置下:
      * resolution_scale=1.0 -> 训练像素 = 1236 (主阶段, 1/4 物理)
      * resolution_scale=2.0 -> 训练像素 = 618  (1/8 物理 warmup)
      * resolution_scale=4.0 -> 训练像素 = 309  (1/16 物理 warmup)
    因此 schedule 写 "1200:4,4000:2,total:1" 是正确的 (4/2/1 是 ratio 表达).

    注意: enable_resolution_aware_schedule=True 时, RAS 会基于
    ResolutionProfiler.profile_scene (cur_pixels=4M for images_2)
    自动把 1200/4000 按 sqrt(4M/426k)=2.5 缩放 (clamp 到 [1.0, 2.5]).
    """
    return [
        "--enable_resolution_aware_schedule",
        "--resolution_schedule", f"1200:4,4000:2,{iterations}:1",
    ]


# 4 个方法组 (A 改动, 2026-10-19: max_gaussians 运行时注入)
# extra_args 改为只存"方法特定"部分, main() 拼接 _common_lr_args(max_gaussians=args.max_gaussians) + extra_args
_GROUPS_BASE: Dict[str, Dict] = {
    "A1": {
        "name": "Raw3DGS",
        "description": "原始 3DGS, 无任何 TAGS 模块 (对照组)",
        "extra_args": [
            "--lpips_loss_weight_init", "0.0",
            "--enable_sparse_optimization", "False",
            "--sparse_mode", "none",
            "--prune_ratio2", "0.0",
            "--enable_geo_prior", "False",
            "--enable_resolution_aware_schedule", "False",
        ],
    },
    "A2": {
        "name": "TAGS_Geo",
        "description": "3DGS + 几何先验 (depth + normal, 后期介入)",
        "extra_args": _geo_args() + [
            "--lpips_loss_weight_init", "0.0",
            "--enable_sparse_optimization", "False",
            "--sparse_mode", "none",
            "--prune_ratio2", "0.0",
            "--enable_resolution_aware_schedule", "False",
        ],
    },
    "A3": {
        "name": "TAGS_SPA",
        "description": "3DGS + GaussianSpa 稀疏化 (无几何)",
        "extra_args": _spa_args() + [
            "--lpips_loss_weight_init", "0.0",
            "--enable_geo_prior", "False",
            "--enable_resolution_aware_schedule", "False",
        ],
    },
    "A4": {
        "name": "Full_TAGS",
        "description": "完整 TAGS: Geo + SPA + LPIPS staged + 多分辨率",
        "extra_args": _geo_args() + _spa_args() + _lpips_staged_args() + _resolution_aware_args(),
    },
}
# 兼容旧名 (下游 dict 引用)
GROUPS: Dict[str, Dict] = _GROUPS_BASE  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="v3 Group A: 主要方法对比 (1/2 分辨率)")
    parser.add_argument("--scenes", default="bicycle,garden,bonsai,room",
                        help="4 场景: bicycle,garden,bonsai,room")
    parser.add_argument("--groups", default="A1,A2,A3,A4", help="4 方法: A1,A2,A3,A4")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="5 seed")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--images-dir", default="images_2",
                        help="v3: 主实验源图目录 (1/2 物理, e.g. images_2). "
                             "实际训练像素由 --resolution 控制, 与 images-dir 联合决定 (见 §1.0).")
    parser.add_argument("--resolution", type=int, default=2,
                        help="v3: 主阶段训练像素 = orig_w / resolution. "
                             "配合 --images images_2 给出 (1) C.1=8 (1/16 物理) "
                             "(2) C.2=2 (1/4 物理) (3) C.3=1 (1/2 物理). "
                             "A/B 主实验默认 2.")
    # ===== v3 Group A: max_gaussians 改为可配置 (A 改动, 2026-10-19) =====
    # 默认 2500000 保留 80-run 兼容性, bonsai/room 验证 SPA 时设为 6000000 (放开上限)
    parser.add_argument("--max-gaussians", type=int, default=2500000,
                        help="上限高斯数. 默认 2.5M 维持 max_gaussians 撞顶对比; "
                             "放开用 6M 让 SPA 真正稀疏化 (对比 4 方法高斯数差异).")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true",
                        help="跑完后写 exp4v3_A_method_comparison.csv")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    groups = [g.strip().upper() for g in args.groups.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group: {g}, available: {list(GROUPS)}")
    for s in scenes:
        if s not in SCENES_V3:
            raise SystemExit(f"v3 Group A 4 场景: {SCENES_V3} (got {s})")

    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 2026-10-20: 不同场景物理像素差异巨大, 统一 1/4 物理对 bonsai/room (3MP) 太小
    # bicycle/garden (16MP) 用 images_2+res=2 (1/4 物理 = 1MP)
    # bonsai/room (6MP)    用 images+res=1   (full-res = 6.4MP, 资源够)
    # 让 wall_s 量级相近 (主阶段 1MP vs 6MP, 后者 Vram 約+30% 而时间 +2x, 可接受)
    SCENES_FULL_RES = {"bonsai", "room"}
    for scene in scenes:
        # ===== A 改动 (2026-10-19): 用 CLI 注入的 max_gaussians 重建 4 个方法组 =====
        # 默认 2.5M (80-run 兼容), bonsai/room SPA 验证用 6M (放开)
        if scene in SCENES_FULL_RES:
            # full-res: --images images (原始 1:1 物理源图) + --resolution 1 (主阶段 = 原像素)
            scene_imgs = "images"
            scene_res  = 1
        else:
            # sub-res: --images images_2 + --resolution 2 (= 1/4 物理像素, 与历史结果一致)
            scene_imgs = args.images_dir   # 默认 images_2
            scene_res  = args.resolution   # 默认 2
        for gid, gcfg in _GROUPS_BASE.items():
            full_extra = _common_lr_args(
                max_gaussians=args.max_gaussians,
                images_dir=scene_imgs,
                resolution=scene_res,
            ) + gcfg["extra_args"]
            GROUPS[gid]["extra_args"] = full_extra
        print(f"[v3 Group A] scene={scene}  max_gaussians={args.max_gaussians}  "
              f"images={scene_imgs}  resolution={scene_res}  ({'full' if scene in SCENES_FULL_RES else 'sub'}-res)")

    total = len(scenes) * len(groups) * len(seeds)
    done = 0
    t0 = time.time()
    for scene in scenes:
        for group in groups:
            cfg = GROUPS[group]
            for seed in seeds:
                group_id = f"A.{group}_{cfg['name']}"
                summary_path = output_root / f"exp4v3_A_{scene}" / group_id / "summary.json"
                # 2026-10-17 修订: skip 判定要双重验证 (summary 有 PSNR + final PLY 真的在)
                # 否则 A.1 旧版 1600px 跑点会被误判为完成, 永远 skip
                if args.skip_existing and summary_path.is_file():
                    s = load_summary(summary_path)
                    seed_info = s.get("seeds", {}).get(str(seed), {})
                    final_ply = (output_root / f"exp4v3_A_{scene}" / group_id /
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
                print(f"\n[Group A {done}/{total}] {scene} {group_id} seed={seed}")
                if args.dry_run:
                    print(f"  DRY-RUN extra_args={cfg['extra_args']}")
                    continue
                info = run_full(
                    scene=scene,
                    group=group_id,
                    seed=seed,
                    extra_args=_sanitize_bool_flags(cfg["extra_args"]),
                    output_root=output_root / f"exp4v3_A_{scene}",
                    iterations=args.iterations,
                    python=args.python,
                    images_dir=args.images_dir,
                    resolution=args.resolution,
                )
                print(f"  PSNR={info.get('psnr')} N={info.get('num_points')}")
    elapsed = time.time() - t0
    print(f"\n[v3 Group A] {done} runs done in {elapsed/60:.1f} min")

    if args.summarize:
        summarize_group_a(output_root)


def summarize_group_a(output_root: Path) -> None:
    """汇总 4 方法 × 4 场景 × N seed 的结果, 写 method_comparison.csv + per_scene.csv."""
    stats_root = output_root / "exp4_stats"
    stats_root.mkdir(parents=True, exist_ok=True)

    # 收集所有 (scene, group, seed) 结果
    rows = []
    for scene in SCENES_V3:
        for group_id, cfg in GROUPS.items():
            full_id = f"A.{group_id}_{cfg['name']}"
            summary_path = output_root / f"exp4v3_A_{scene}" / full_id / "summary.json"
            if not summary_path.is_file():
                continue
            summary = load_summary(summary_path)
            seeds_data = summary.get("seeds", {})
            for seed_str, info in seeds_data.items():
                rows.append({
                    "scene": scene,
                    "group": full_id,
                    "method": cfg["name"],
                    "seed": int(seed_str),
                    "psnr": info.get("psnr"),
                    "ssim": info.get("ssim"),
                    "lpips": info.get("lpips"),
                    "num_points": info.get("num_points"),
                    "wall_time_sec": info.get("wall_time_sec"),
                    "peak_vram_mb": info.get("peak_vram_mb"),
                })

    if not rows:
        print("[v3 Group A] No summary data found.")
        return

    # 写 per-scene 详细
    per_scene_path = stats_root / "exp4v3_A_per_scene.csv"
    with per_scene_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[v3 Group A] Wrote {per_scene_path} ({len(rows)} rows)")

    # 写 method_comparison (跨场景主表)
    method_rows = []
    for method in sorted(set(r["method"] for r in rows)):
        method_data = [r for r in rows if r["method"] == method]
        for scene in SCENES_V3:
            scene_data = [r for r in method_data if r["scene"] == scene]
            if not scene_data:
                continue
            psnrs = [r["psnr"] for r in scene_data if r["psnr"] is not None]
            ssims = [r["ssim"] for r in scene_data if r["ssim"] is not None]
            lpipss = [r["lpips"] for r in scene_data if r["lpips"] is not None]
            points = [r["num_points"] for r in scene_data if r["num_points"] is not None]
            method_rows.append({
                "method": method,
                "scene": scene,
                "n_seeds": len(scene_data),
                "psnr_mean": f"{statistics.mean(psnrs):.3f}" if psnrs else "n/a",
                "psnr_std": f"{statistics.stdev(psnrs):.3f}" if len(psnrs) > 1 else "0.000",
                "ssim_mean": f"{statistics.mean(ssims):.4f}" if ssims else "n/a",
                "ssim_std": f"{statistics.stdev(ssims):.4f}" if len(ssims) > 1 else "0.0000",
                "lpips_mean": f"{statistics.mean(lpipss):.4f}" if lpipss else "n/a",
                "lpips_std": f"{statistics.stdev(lpipss):.4f}" if len(lpipss) > 1 else "0.0000",
                "num_points_mean": f"{statistics.mean(points):.0f}" if points else "n/a",
            })

    mc_path = stats_root / "exp4v3_A_method_comparison.csv"
    with mc_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(method_rows[0].keys()))
        writer.writeheader()
        writer.writerows(method_rows)
    print(f"[v3 Group A] Wrote {mc_path} ({len(method_rows)} rows)")


if __name__ == "__main__":
    main()
