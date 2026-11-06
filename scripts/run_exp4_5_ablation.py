#!/usr/bin/env python3
"""v2 §4.5 多模块协同消融 (2^k 析因 + negative control)

7 组 × 4 场景 × 5 seed = 140 runs

实验组:
  4.5.A Baseline (3DGS)   : 所有 TAGS 模块关闭
  4.5.B w/o Perceptual    : LPIPS off, SPA on, Geo on, Resolution on
  4.5.C w/o SPA           : LPIPS on, SPA off, Geo on, Resolution on
  4.5.D w/o Geo           : LPIPS on, SPA on, Geo off, Resolution on
  4.5.E w/o Resolution    : LPIPS on, SPA on, Geo on, Resolution off
  4.5.F Full TAGS         : 全部开
  4.5.G Negative Control  : SPA 时段故意设到 25000-30000 (几何窗口之后), 验证消融方向正确

固定:
  - §4.2/§4.3/§4.4 选定的最终参数 (PRUNE_RATIO2, LPIPS 调度, GEO 窗口与权重)
  - 30000 iter, 5 seed, 4 场景 (bicycle, garden, bonsai, room)

输出:
  output/exp4_{scene}/4_5_*/summary.json
  output/exp4_{scene}/4_5_*/test/results.json (由 metrics.py)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp4_runner import run_full, load_summary  # type: ignore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# §4.2/§4.3/§4.4 选定的最终参数 (临时, 实际选定后回填)
PRUNE_RATIO2 = 0.04
LPIPS_INIT = 0.004
LPIPS_FINAL = 0.001
# 几何窗口 (T3)
GEO_START = 26000
GEO_WARMUP = 27000
GEO_DECAY_START = 29000
GEO_DECAY_UNTIL = 30000
DEPTH_FINAL = 0.05
NORMAL_FINAL = 0.02

SCENES_V2 = ["bicycle", "garden", "bonsai", "room"]


def _baseline_no_geo():
    return [
        "--lpips_loss_weight_init", "0.0",
        "--lpips_loss_weight_final", "0.0",
        "--psnr_loss_weight_init", "0.0",
        "--psnr_loss_weight_final", "0.0",
        "--color_align_weight_init", "0.0",
        "--color_align_weight_final", "0.0",
        "--detail_loss_weight_init", "0.0",
        "--detail_loss_weight_final", "0.0",
        "--sparse_mode", "none",
        "--prune_ratio2", "0.0",
        "--spa_prune_ratio", "0.0",
    ]


def _geo_args(start=GEO_START, warmup=GEO_WARMUP, decay_start=GEO_DECAY_START, decay_until=GEO_DECAY_UNTIL,
              depth_final=DEPTH_FINAL, normal_final=NORMAL_FINAL):
    return [
        "--enable_geo_prior",
        "--geo_prior_depth",
        "--geo_prior_normal",
        "--geo_prior_start_iter", str(start),
        "--geo_prior_warmup_until", str(warmup),
        "--geo_prior_decay_start", str(decay_start),
        "--geo_prior_decay_until", str(decay_until),
        "--depth_prior_weight_init", "0.0",
        "--depth_prior_weight_final", str(depth_final),
        "--normal_prior_weight_init", "0.0",
        "--normal_prior_weight_final", str(normal_final),
    ]


def _spa_args(start=13000, end=25000, ratio=PRUNE_RATIO2):
    return [
        "--sparse_mode", "spa",
        "--prune_ratio2", str(ratio),
        "--spa_start_iter", str(start),
        "--spa_end_iter", str(end),
    ]


def _lpips_args(init=LPIPS_INIT, final=LPIPS_FINAL, schedule="step"):
    return [
        "--lpips_loss_weight_init", str(init),
        "--lpips_loss_weight_final", str(final),
        "--lpips_loss_schedule", schedule,
    ]


def _resolution_args():
    return [
        "--enable_resolution_adaptive",
        "--enable_resolution_aware_schedule",
        # 默认 ras_densify_base=7200, spa_delay=2000 等
    ]


GROUPS = {
    "A": {
        "name": "Baseline_3DGS",
        "description": "3DGS baseline (所有 TAGS 模块关闭)",
        "extra_args": _baseline_no_geo(),
    },
    "B": {
        "name": "wo_Perceptual",
        "description": "LPIPS off, 其他全开",
        "extra_args": _geo_args() + _spa_args() + _resolution_args() + [
            "--lpips_loss_weight_init", "0.0",
            "--lpips_loss_weight_final", "0.0",
        ],
    },
    "C": {
        "name": "wo_SPA",
        "description": "SPA off, 其他全开",
        "extra_args": _geo_args() + _resolution_args() + _lpips_args() + [
            "--sparse_mode", "none",
            "--prune_ratio2", "0.0",
            "--spa_prune_ratio", "0.0",
        ],
    },
    "D": {
        "name": "wo_Geo",
        "description": "Geo off, 其他全开",
        "extra_args": _spa_args() + _resolution_args() + _lpips_args(),
    },
    "E": {
        "name": "wo_Resolution",
        "description": "Resolution off, 其他全开",
        "extra_args": _geo_args() + _spa_args() + _lpips_args(),
    },
    "F": {
        "name": "Full_TAGS",
        "description": "Full TAGS (所有模块全开)",
        "extra_args": _geo_args() + _spa_args() + _resolution_args() + _lpips_args(),
    },
    # Negative Control: SPA 时段故意设到 25000-30000 (几何窗口之后, 应介于 3DGS 和 Full TAGS 之间)
    # 若此组表现优于 4.5.C (w/o SPA), 说明 v2 设计方向错
    "G": {
        "name": "NegativeControl_LateSPA",
        "description": "SPA 时段 25000-30000 (几何窗口之后, 反向对照)",
        "extra_args": _geo_args() + _resolution_args() + _lpips_args() + _spa_args(start=25000, end=30000),
    },
}


def main():
    parser = argparse.ArgumentParser(description="v2 §4.5 多模块协同消融 (含 negative control)")
    parser.add_argument("--scenes", default="bicycle,garden,bonsai,room")
    parser.add_argument("--groups", default="A,B,C,D,E,F,G")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--images-dir", default="images_2",
                        help="v3: 主实验用 images_2 (1/2 分辨率)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    groups = [g.strip().upper() for g in args.groups.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group: {g}, available: {list(GROUPS)}")
    for s in scenes:
        if s not in SCENES_V2:
            raise SystemExit(f"§4.5 4 场景: {SCENES_V2} (got {s})")

    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(scenes) * len(groups) * len(seeds)
    done = 0
    t0 = time.time()
    for scene in scenes:
        for group in groups:
            cfg = GROUPS[group]
            for seed in seeds:
                group_id = f"4_5_{group}_{cfg['name']}"
                summary_path = output_root / scene / group_id / "summary.json"
                if args.skip_existing and summary_path.is_file():
                    s = load_summary(summary_path)
                    seed_info = s.get("seeds", {}).get(str(seed), {})
                    if seed_info.get("psnr") is not None:
                        print(f"[skip] {scene} {group_id} seed={seed}")
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
    print(f"\n[§4.5] {done} runs done in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
