#!/usr/bin/env python3
"""v2 §4.2 高斯稀疏化强度选择 (5 组 × 3 场景 × 5 seed = 75 runs)

对应论文 §4.2.3 实验分组 A/B/C/D/E.
3 场景: bicycle (主) + garden (户外副本) + bonsai (室内副本)
5 seed: 0, 1, 2, 3, 4

实验组 (含 prune_ratio2 调整):
  4.2.A Baseline  : SPA off (sparse_mode=none), prune_ratio2=0
  4.2.B Weak      : sparse_mode=spa, prune_ratio2=0.02
  4.2.C Reference : sparse_mode=spa, prune_ratio2=0.04
  4.2.D MedStrong : sparse_mode=spa, prune_ratio2=0.06
  4.2.E Strong    : sparse_mode=spa, prune_ratio2=0.08

固定条件:
  - LPIPS / PSNR / color align 全部置 0
  - geo_prior 关闭
  - 分辨率调度关闭
  - 30000 iter, --llffhold 8, --images images_4

输出:
  output/exp4_{scene}/4_2_* 目录
  output/exp4_{scene}/4_2_*/summary.json
  output/exp4_stats/exp4_2_*.csv 汇总
  output/exp4_stats/exp4_2_paired_t.csv 配对 t 检验
  docs/exp4_2_sparsity_ratio.md

用法:
  # 跑全部 75 runs
  python scripts/run_exp4_2_sweep.py

  # 跑单场景单组
  python scripts/run_exp4_2_sweep.py --scenes bicycle --groups C

  # 单个 (scene, group, seed)
  python scripts/run_exp4_2_sweep.py --scenes bicycle --groups C --seeds 0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp4_runner import run_full, load_summary  # type: ignore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

GROUPS = {
    "A": {
        "name": "Baseline",
        "description": "SPA off, prune_ratio2=0",
        "extra_args": [
            "--sparse_mode", "none",
            "--prune_ratio2", "0.0",
            "--spa_prune_ratio", "0.0",
            # 全关其他损失
            "--lpips_loss_weight_init", "0.0",
            "--lpips_loss_weight_final", "0.0",
            "--psnr_loss_weight_init", "0.0",
            "--psnr_loss_weight_final", "0.0",
            "--color_align_weight_init", "0.0",
            "--color_align_weight_final", "0.0",
            "--detail_loss_weight_init", "0.0",
            "--detail_loss_weight_final", "0.0",
        ],
    },
    "B": {
        "name": "Weak",
        "description": "sparse_mode=spa, prune_ratio2=0.02",
        "extra_args": [
            "--sparse_mode", "spa",
            "--prune_ratio2", "0.02",
            "--spa_start_iter", "13000",
            "--spa_end_iter", "25000",
            "--lpips_loss_weight_init", "0.0",
            "--lpips_loss_weight_final", "0.0",
            "--psnr_loss_weight_init", "0.0",
            "--psnr_loss_weight_final", "0.0",
            "--color_align_weight_init", "0.0",
            "--color_align_weight_final", "0.0",
            "--detail_loss_weight_init", "0.0",
            "--detail_loss_weight_final", "0.0",
        ],
    },
    "C": {
        "name": "Reference",
        "description": "sparse_mode=spa, prune_ratio2=0.04",
        "extra_args": [
            "--sparse_mode", "spa",
            "--prune_ratio2", "0.04",
            "--spa_start_iter", "13000",
            "--spa_end_iter", "25000",
            "--lpips_loss_weight_init", "0.0",
            "--lpips_loss_weight_final", "0.0",
            "--psnr_loss_weight_init", "0.0",
            "--psnr_loss_weight_final", "0.0",
            "--color_align_weight_init", "0.0",
            "--color_align_weight_final", "0.0",
            "--detail_loss_weight_init", "0.0",
            "--detail_loss_weight_final", "0.0",
        ],
    },
    "D": {
        "name": "MediumStrong",
        "description": "sparse_mode=spa, prune_ratio2=0.06",
        "extra_args": [
            "--sparse_mode", "spa",
            "--prune_ratio2", "0.06",
            "--spa_start_iter", "13000",
            "--spa_end_iter", "25000",
            "--lpips_loss_weight_init", "0.0",
            "--lpips_loss_weight_final", "0.0",
            "--psnr_loss_weight_init", "0.0",
            "--psnr_loss_weight_final", "0.0",
            "--color_align_weight_init", "0.0",
            "--color_align_weight_final", "0.0",
            "--detail_loss_weight_init", "0.0",
            "--detail_loss_weight_final", "0.0",
        ],
    },
    "E": {
        "name": "Strong",
        "description": "sparse_mode=spa, prune_ratio2=0.08",
        "extra_args": [
            "--sparse_mode", "spa",
            "--prune_ratio2", "0.08",
            "--spa_start_iter", "13000",
            "--spa_end_iter", "25000",
            "--lpips_loss_weight_init", "0.0",
            "--lpips_loss_weight_final", "0.0",
            "--psnr_loss_weight_init", "0.0",
            "--psnr_loss_weight_final", "0.0",
            "--color_align_weight_init", "0.0",
            "--color_align_weight_final", "0.0",
            "--detail_loss_weight_init", "0.0",
            "--detail_loss_weight_final", "0.0",
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(description="v2 §4.2 高斯稀疏化强度选择")
    parser.add_argument("--scenes", default="bicycle,garden,bonsai", help="3 场景: bicycle,garden,bonsai")
    parser.add_argument("--groups", default="A,B,C,D,E", help="5 组: A,B,C,D,E")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="5 seed")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--images-dir", default="images_2",
                        help="v3: 主实验用 images_2 (1/2 分辨率), 备选 images_4 / images")
    parser.add_argument("--skip-existing", action="store_true", help="summary.json 已存在且 psnr 非空则跳过")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    groups = [g.strip().upper() for g in args.groups.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for s in scenes:
        if s not in ("bicycle", "garden", "bonsai"):
            raise SystemExit(f"§4.2 只支持 3 场景: bicycle,garden,bonsai (got {s})")
    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group: {g}, available: {list(GROUPS)}")

    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(scenes) * len(groups) * len(seeds)
    done = 0
    t0 = time.time()
    for scene in scenes:
        for group in groups:
            cfg = GROUPS[group]
            for seed in seeds:
                group_id = f"4_2_{group}_{cfg['name']}"
                summary_path = output_root / scene / group_id / "summary.json"
                if args.skip_existing and summary_path.is_file():
                    s = load_summary(summary_path)
                    seed_info = s.get("seeds", {}).get(str(seed), {})
                    if seed_info.get("psnr") is not None:
                        print(f"[skip] {scene} {group_id} seed={seed} (existing psnr={seed_info['psnr']})")
                        done += 1
                        continue
                done += 1
                print(f"\n[{done}/{total}] {scene} {group_id} seed={seed}")
                if args.dry_run:
                    print(f"  DRY-RUN extra_args={cfg['extra_args']}")
                    continue
                info = run_full(
                    scene=scene,
                    group=group_id,
                    seed=seed,
                    extra_args=cfg["extra_args"],
                    output_root=output_root,
                    iterations=args.iterations,
                    python=args.python,
                    images_dir=args.images_dir,
                )
                print(f"  PSNR={info.get('psnr')} N={info.get('num_points')}")
    elapsed = time.time() - t0
    print(f"\n[§4.2] {done} runs done in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
