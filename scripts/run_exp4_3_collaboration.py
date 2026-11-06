#!/usr/bin/env python3
"""v2 §4.3 感知损失 × 稀疏化协同 (5 组 × 3 场景 × 5 seed = 75 runs)

对应论文 §4.3.3 实验分组 A/B/C/D/E.
3 场景: bicycle, garden, bonsai
5 seed: 0..4

实验组 (LPIPS × SPA 协同):
  4.3.A Control       : LPIPS off, SPA off (4.2.A 复用)
  4.3.B Perceptual    : LPIPS on (init=final=0.004), SPA off
  4.3.C Sparse        : LPIPS off, SPA on (4.2.C 复用)
  4.3.D Joint-Fixed   : LPIPS on (init=final=0.004), SPA on
  4.3.E TAGS-Staged   : LPIPS on + staged schedule, SPA on
    4.3.E LPIPS 调度:
      8000 前 0.004 (与论文 §4.3.2 "持续感知约束" 对齐)
      8000-13000 衰减到 0.001
      13000-25000 低权重 0.001 (与 SPA 重叠)
      25000-30000 0 (放弃 LPIPS, 留给颜色/几何)

固定条件:
  - prune_ratio2 = 0.04 (§4.2 选定值, 临时)
  - geo_prior / 分辨率 / 颜色对齐 全部关闭
  - 30000 iter

注意:
  - 4.3.E 的 staged 调度: 当前 loss_builder.py 仅支持 "linear" 和 "step".
    需要在 loss_builder.py 添加 "staged" 模式:
    cfg.lpips_loss_schedule = "staged"
    + cfg.lpips_loss_staged_iters = "8000:0.004,13000:0.001,25000:0.0"
    本脚本先假设 staged 已实现, 跑起来后如果报错, 需要先扩展 loss_builder.py
  - 配对 t 检验在 summarize_exp4_5.py 中实现 (跨场景统一汇总)

输出:
  output/exp4_{scene}/4_3_*/summary.json
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp4_runner import run_full, load_summary  # type: ignore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# §4.2 选定的 prune_ratio2 (默认 0.04, 实际选定后改这里)
PRUNE_RATIO2_DEFAULT = 0.04

GROUPS = {
    "A": {
        "name": "Control",
        "description": "LPIPS off, SPA off",
        "extra_args": [
            "--sparse_mode", "none",
            "--prune_ratio2", "0.0",
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
        "name": "Perceptual",
        "description": "LPIPS on (init=final=0.004), SPA off",
        "extra_args": [
            "--sparse_mode", "none",
            "--prune_ratio2", "0.0",
            "--lpips_loss_weight_init", "0.004",
            "--lpips_loss_weight_final", "0.004",
            "--lpips_loss_schedule", "step",  # 直接跳到 final
            "--psnr_loss_weight_init", "0.0",
            "--psnr_loss_weight_final", "0.0",
            "--color_align_weight_init", "0.0",
            "--color_align_weight_final", "0.0",
            "--detail_loss_weight_init", "0.0",
            "--detail_loss_weight_final", "0.0",
        ],
    },
    "C": {
        "name": "Sparse",
        "description": "LPIPS off, SPA on (prune_ratio2=Reference)",
        "extra_args": [
            "--sparse_mode", "spa",
            "--prune_ratio2", str(PRUNE_RATIO2_DEFAULT),
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
        "name": "JointFixed",
        "description": "LPIPS on (init=final=0.004), SPA on (联合优化, 固定 LPIPS 权重)",
        "extra_args": [
            "--sparse_mode", "spa",
            "--prune_ratio2", str(PRUNE_RATIO2_DEFAULT),
            "--spa_start_iter", "13000",
            "--spa_end_iter", "25000",
            "--lpips_loss_weight_init", "0.004",
            "--lpips_loss_weight_final", "0.004",
            "--lpips_loss_schedule", "step",
            "--psnr_loss_weight_init", "0.0",
            "--psnr_loss_weight_final", "0.0",
            "--color_align_weight_init", "0.0",
            "--color_align_weight_final", "0.0",
            "--detail_loss_weight_init", "0.0",
            "--detail_loss_weight_final", "0.0",
        ],
    },
    "E": {
        "name": "TagsStaged",
        "description": "LPIPS on (staged schedule), SPA on (TAGS 分阶段调度)",
        "extra_args": [
            "--sparse_mode", "spa",
            "--prune_ratio2", str(PRUNE_RATIO2_DEFAULT),
            "--spa_start_iter", "13000",
            "--spa_end_iter", "25000",
            # 论文 §4.3.2 staged 调度: 8000 前 0.004, 衰减 0.001, 25000 后 0
            "--lpips_loss_weight_init", "0.0",  # staged 模式下 init/final 不直接用
            "--lpips_loss_weight_final", "0.0",
            "--lpips_loss_schedule", "staged",
            "--lpips_loss_staged_iters", "0:0.004,8000:0.001,13000:0.001,25000:0.0",
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
    parser = argparse.ArgumentParser(description="v2 §4.3 感知损失 × 稀疏化协同")
    parser.add_argument("--scenes", default="bicycle,garden,bonsai")
    parser.add_argument("--groups", default="A,B,C,D,E")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--prune-ratio2", type=float, default=PRUNE_RATIO2_DEFAULT,
                        help="从 §4.2 选定的 prune_ratio2 值")
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
        if s not in ("bicycle", "garden", "bonsai"):
            raise SystemExit(f"§4.3 只支持 3 场景: bicycle,garden,bonsai (got {s})")

    # 全局替换 prune_ratio2 (从 §4.2 选定)
    for g_id, cfg in GROUPS.items():
        cfg["extra_args"] = [
            str(args.prune_ratio2) if x == str(PRUNE_RATIO2_DEFAULT) else x
            for x in cfg["extra_args"]
        ]

    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(scenes) * len(groups) * len(seeds)
    done = 0
    t0 = time.time()
    for scene in scenes:
        for group in groups:
            cfg = GROUPS[group]
            for seed in seeds:
                group_id = f"4_3_{group}_{cfg['name']}"
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
    print(f"\n[§4.3] {done} runs done in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
