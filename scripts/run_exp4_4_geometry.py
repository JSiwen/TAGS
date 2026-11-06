#!/usr/bin/env python3
"""v2 §4.4 几何先验介入时机与约束强度 (配对设计)

配对设计 (v2 关键改进):
  T0 = S0 (No-Geo), 4.4 共享基线, 不重复跑
  T3 = S2 (Geo-Late + Base 1.0), 4.4 共享基线, 不重复跑
  总独立组数: 6 组 (T0, T1, T2, T3, S1, S3) × 4 场景 × 5 seed = 120 runs

实验组 (4.4.2 介入时机):
  4.4.T0 No-Geo  : 全程关闭 (基线, 与 S0 共享)
  4.4.T1 Early   : geo_prior_warmup_until=5000, decay_until=5000 (早期介入)
  4.4.T2 Mid     : geo_prior_start_iter=13000, warmup_until=15000, decay_until=17000 (与 SPA 重叠)
  4.4.T3 Late    : geo_prior_start_iter=26000, warmup_until=27000, decay_until=30000 (后期精修, 论文主张)

实验组 (4.4.3 约束强度, 介入窗口 = T3):
  4.4.S0 = T0     : No-Geo (共享)
  4.4.S1 Weak     : depth_final=0.025, normal_final=0.01 (Base 的 0.5x)
  4.4.S2 = T3     : Base (depth_final=0.05, normal_final=0.02) (共享)
  4.4.S3 Strong   : depth_final=0.075, normal_final=0.03 (Base 的 1.5x)

注意:
  - 4.4 用的是 "配对设计", 所有 6 个独立组都用同一组 seed 跑, 可直接做 paired t-test
  - 配对 t 检验在 summarize_exp4_5.py 中实现

固定条件:
  - 复用 §4.3 选定的 LPIPS 调度 + §4.2 选定的 prune_ratio2
  - 30000 iter, 各组几何窗口长度 4000 iter (T1/T2/T3) 或 0 (T0)
  - 可靠性掩码覆盖率 ≥ 20% (默认)

输出:
  output/exp4_{scene}/4_4_*/summary.json
  output/exp4_{scene}/4_4_*/geo_eval/ 每视角几何一致性
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp4_runner import run_full, load_summary  # type: ignore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# §4.2/§4.3 选定 (临时, 实际跑完 §4.2 §4.3 后回填)
PRUNE_RATIO2 = 0.04
LPIPS_INIT = 0.004
LPIPS_FINAL = 0.001


def _common_args():
    """所有 4.4 组共享的基线参数."""
    return [
        # SPA (§4.2 选定)
        "--sparse_mode", "spa",
        "--prune_ratio2", str(PRUNE_RATIO2),
        "--spa_start_iter", "13000",
        "--spa_end_iter", "25000",
        # LPIPS (§4.3 选定)
        "--lpips_loss_weight_init", str(LPIPS_INIT),
        "--lpips_loss_weight_final", str(LPIPS_FINAL),
        "--lpips_loss_schedule", "step",
        # PSNR / color align 关闭
        "--psnr_loss_weight_init", "0.0",
        "--psnr_loss_weight_final", "0.0",
        "--color_align_weight_init", "0.0",
        "--color_align_weight_final", "0.0",
        "--detail_loss_weight_init", "0.0",
        "--detail_loss_weight_final", "0.0",
        # 通用
        "--depth_prior_min_mask_fraction", "0.20",
    ]


# 6 个独立组
GROUPS = {
    # 时机 (T 系列)
    "T0": {
        "name": "NoGeo",
        "description": "全程关闭 geo_prior, 与 S0 共享",
        "extra_args": _common_args() + [
            # geo_prior 默认关闭 (--enable_geo_prior 不传)
        ],
    },
    "T1": {
        "name": "GeoEarly",
        "description": "早期介入: warmup_until=5000, decay_until=5000 (窗口 1000-5000)",
        "extra_args": _common_args() + [
            "--enable_geo_prior",
            "--geo_prior_depth",
            "--geo_prior_normal",
            "--geo_prior_start_iter", "1000",
            "--geo_prior_warmup_until", "3000",
            "--geo_prior_decay_start", "4500",
            "--geo_prior_decay_until", "5000",
        ],
    },
    "T2": {
        "name": "GeoMid",
        "description": "中期介入 (与 SPA 重叠): 13000-17000",
        "extra_args": _common_args() + [
            "--enable_geo_prior",
            "--geo_prior_depth",
            "--geo_prior_normal",
            "--geo_prior_start_iter", "13000",
            "--geo_prior_warmup_until", "15000",
            "--geo_prior_decay_start", "16000",
            "--geo_prior_decay_until", "17000",
        ],
    },
    "T3": {
        "name": "GeoLate",
        "description": "后期精修 (论文主张): 26000-30000, Base 强度",
        "extra_args": _common_args() + [
            "--enable_geo_prior",
            "--geo_prior_depth",
            "--geo_prior_normal",
            "--geo_prior_start_iter", "26000",
            "--geo_prior_warmup_until", "27000",
            "--geo_prior_decay_start", "29000",
            "--geo_prior_decay_until", "30000",
            "--depth_prior_weight_init", "0.0",
            "--depth_prior_weight_final", "0.05",
            "--normal_prior_weight_init", "0.0",
            "--normal_prior_weight_final", "0.02",
        ],
    },
    # 强度 (S 系列, 介入窗口 = T3)
    "S1": {
        "name": "GeoWeak",
        "description": "后期介入 + 弱强度 (Base 的 0.5x)",
        "extra_args": _common_args() + [
            "--enable_geo_prior",
            "--geo_prior_depth",
            "--geo_prior_normal",
            "--geo_prior_start_iter", "26000",
            "--geo_prior_warmup_until", "27000",
            "--geo_prior_decay_start", "29000",
            "--geo_prior_decay_until", "30000",
            "--depth_prior_weight_init", "0.0",
            "--depth_prior_weight_final", "0.025",  # 0.5x Base
            "--normal_prior_weight_init", "0.0",
            "--normal_prior_weight_final", "0.01",
        ],
    },
    "S3": {
        "name": "GeoStrong",
        "description": "后期介入 + 强强度 (Base 的 1.5x)",
        "extra_args": _common_args() + [
            "--enable_geo_prior",
            "--geo_prior_depth",
            "--geo_prior_normal",
            "--geo_prior_start_iter", "26000",
            "--geo_prior_warmup_until", "27000",
            "--geo_prior_decay_start", "29000",
            "--geo_prior_decay_until", "30000",
            "--depth_prior_weight_init", "0.0",
            "--depth_prior_weight_final", "0.075",  # 1.5x Base
            "--normal_prior_weight_init", "0.0",
            "--normal_prior_weight_final", "0.03",
        ],
    },
}

# 4 场景 (v2 全部)
SCENES_V2 = ["bicycle", "garden", "bonsai", "room"]


def main():
    parser = argparse.ArgumentParser(description="v2 §4.4 几何先验时机与强度 (配对设计)")
    parser.add_argument("--scenes", default="bicycle,garden,bonsai,room")
    parser.add_argument("--groups", default="T0,T1,T2,T3,S1,S3", help="6 独立组 (T3=S2, T0=S0 共享)")
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
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group: {g}, available: {list(GROUPS)}")
    for s in scenes:
        if s not in SCENES_V2:
            raise SystemExit(f"§4.4 4 场景: {SCENES_V2} (got {s})")

    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(scenes) * len(groups) * len(seeds)
    done = 0
    t0 = time.time()
    for scene in scenes:
        for group in groups:
            cfg = GROUPS[group]
            for seed in seeds:
                group_id = f"4_4_{group}_{cfg['name']}"
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
    print(f"\n[§4.4] {done} runs done in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
