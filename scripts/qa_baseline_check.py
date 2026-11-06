#!/usr/bin/env python3
"""v3 计划 Step B QA gate: no-pretrain 3DGS baseline 验证 PSNR 阈值

在投入扫参前, 先用 no-pretrain 的 3DGS baseline 在 4 场景上各跑 5 seed,
验证 PSNR 是否达到阈值, 防止预处理失败拖累后续实验.

v3 改动: 默认 1/2 分辨率 (--images images_2), 阈值按分辨率调整.

- 4 场景: bicycle (主) / garden (户外) / bonsai (室内) / room (室内)
- 5 seed: 0, 1, 2, 3, 4
- 训练 30000 步
- 分辨率: 1/2 (主实验) / 1/4 (备选) / 标准 (验证)
- 输出 baseline_thresholds.json + qa_baseline_report.md

阈值参考 (基于 Inria 3DGS baseline, 论文 §5.1.1):
  - bicycle: ≥ 24.0 dB
  - garden: ≥ 25.0 dB (论文 mip360_outdoor 均值 25.6 dB)
  - bonsai: ≥ 27.0 dB (论文 mip360_indoor 均值 28.6 dB)
  - room:   ≥ 28.0 dB

v3 阈值: 1/2 分辨率 PSNR 略低于标准分辨率, 阈值可按 0.95x 放宽

用法:
  # 跑全部 4 场景 5 seed @ 1/2 分辨率
  python scripts/qa_baseline_check.py --output output/exp4_stats

  # 跑单场景 1 seed @ 1/2
  python scripts/qa_baseline_check.py --scenes bicycle --seeds 0

  # 标准分辨率验证
  python scripts/qa_baseline_check.py --scenes bicycle --seeds 0 --images-dir images
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# 论文 §5.1.1 Inria 3DGS baseline 阈值 (标准分辨率)
V2_THRESHOLDS_STD = {
    "bicycle": 24.0,
    "garden": 25.0,
    "bonsai": 27.0,
    "room": 28.0,
}

# v3 1/2 分辨率阈值 (PSNR 通常低 0.3-0.5 dB, 阈值按 0.97x 放宽)
V3_THRESHOLDS_HALF = {
    "bicycle": 23.4,  # 24.0 * 0.97
    "garden": 24.4,
    "bonsai": 26.3,
    "room": 27.2,
}

# v3 1/4 分辨率阈值 (PSNR 通常低 1.0-1.5 dB, 阈值按 0.93x 放宽)
V3_THRESHOLDS_QUARTER = {
    "bicycle": 22.5,
    "garden": 23.5,
    "bonsai": 25.3,
    "room": 26.2,
}


def get_thresholds_for_resolution(images_dir: str) -> Dict[str, float]:
    """根据 images_dir 选择对应阈值字典."""
    if images_dir == "images":
        return V2_THRESHOLDS_STD
    elif images_dir == "images_2":
        return V3_THRESHOLDS_HALF
    elif images_dir in ("images_4", "images_8"):
        return V3_THRESHOLDS_QUARTER
    return V2_THRESHOLDS_STD  # fallback


def find_metrics_json(model_dir: Path) -> Optional[Path]:
    """从 model_dir 下找 results.json (由 metrics.py 写出)."""
    cand = model_dir / "results.json"
    if cand.is_file():
        return cand
    return None


def parse_psnr(results_json: Path) -> Optional[float]:
    if not results_json.is_file():
        return None
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    # 格式: {model_dir: {method: {PSNR, SSIM, LPIPS}}}
    for method_dict in payload.values():
        for method_name, metrics in method_dict.items():
            if method_name.lower() in ("ours", "ours_test", "default"):
                return float(metrics.get("PSNR", 0.0))
        # fallback: 取第一个 method
        for metrics in method_dict.values():
            if "PSNR" in metrics:
                return float(metrics["PSNR"])
    return None


def run_one(
    scene: str,
    seed: int,
    output_root: Path,
    iterations: int = 30000,
    hold: int = 8,
    python: str = "python",
    images_dir: str = "images_2",  # v3: 默认 1/2 分辨率
) -> Dict[str, Optional[float]]:
    """对单个 (scene, seed) 跑 baseline 训练 + 渲染 + 评分, 返回 {train_exit, render_exit, psnr, threshold, passed}."""
    group_id = f"qa_baseline"
    train_dir = output_root / scene / group_id / "train" / f"seed{seed}"
    render_dir = output_root / scene / group_id / "renders" / f"eval_seed{seed}"
    metrics_dir = output_root / scene / group_id / "metrics" / f"eval_seed{seed}"

    train_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[QA] === {scene} seed={seed} ===")
    print(f"  train -> {train_dir}")
    print(f"  render -> {render_dir}")

    # 1) train: 3DGS baseline, 关掉所有 TAGS 模块
    # 注: TAGS 用的 action="store_true" 类开关, 默认 False (即关闭). 3DGS baseline 就是不传任何 enable_* 开关.
    train_cmd = [
        python, "train_optimized.py",
        "-s", f"data/{scene}",
        "-m", str(train_dir),
        "--eval",
        "--llffhold", str(hold),
        "--images", images_dir,
        "--iterations", str(iterations),
        "--seed", str(seed),
        # 关掉所有损失 (LPIPS / PSNR / color align 全部置 0)
        "--lpips_loss_weight_init", "0.0",
        "--lpips_loss_weight_final", "0.0",
        "--psnr_loss_weight_init", "0.0",
        "--psnr_loss_weight_final", "0.0",
        "--color_align_weight_init", "0.0",
        "--color_align_weight_final", "0.0",
        "--detail_loss_weight_init", "0.0",
        "--detail_loss_weight_final", "0.0",
        # 关掉 SPA, 标 prune_ratio2=0 防止 budget prune
        "--sparse_mode", "none",
        "--prune_ratio2", "0.0",
        "--spa_prune_ratio", "0.0",
        # 关闭所有高级控制器 (默认就是 False, 这里显式避免上游默认值变化)
        # --enable_geo_prior (默认 False)
        # --enable_resolution_adaptive (默认 False)
        # --enable_sparse_optimization (默认 False)
        # --enable_hard_view_mining (默认 False)
        # --enable_geometry_controller (默认 False)
        # --enable_balanced_late_controller (默认 False)
    ]
    print(f"  $ {' '.join(train_cmd)}")
    train_proc = subprocess.run(train_cmd, cwd=str(REPO_ROOT), check=False)
    if train_proc.returncode != 0:
        return {"scene": scene, "seed": seed, "train_exit": train_proc.returncode, "psnr": None}

    # 2) render
    render_cmd = [
        python, "render.py",
        "-s", f"data/{scene}",
        "-m", str(train_dir),
        "--iteration", str(iterations),
        "--skip_train",
    ]
    print(f"  $ {' '.join(render_cmd)}")
    render_proc = subprocess.run(render_cmd, cwd=str(REPO_ROOT), check=False)
    if render_proc.returncode != 0:
        return {"scene": scene, "seed": seed, "train_exit": 0, "render_exit": render_proc.returncode, "psnr": None}

    # 3) metrics
    metrics_cmd = [
        python, "metrics.py",
        "-m", str(train_dir),
    ]
    print(f"  $ {' '.join(metrics_cmd)}")
    metrics_proc = subprocess.run(metrics_cmd, cwd=str(REPO_ROOT), check=False)

    # 4) parse
    results = find_metrics_json(train_dir)
    psnr = parse_psnr(results) if results else None
    threshold_dict = get_thresholds_for_resolution(images_dir)
    threshold = threshold_dict.get(scene, 24.0)
    passed = psnr is not None and psnr >= threshold
    return {
        "scene": scene,
        "seed": seed,
        "train_exit": 0,
        "render_exit": 0,
        "metrics_exit": metrics_proc.returncode,
        "psnr": psnr,
        "threshold": threshold,
        "passed": passed,
        "images_dir": images_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="v3 QA gate: 跑 no-pretrain 3DGS baseline 验证 PSNR 阈值 (1/2 分辨率)")
    parser.add_argument("--scenes", default="bicycle,garden,bonsai,room")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--output", default="output/exp4_stats")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--images-dir", default="images_2",
                        help="v3: 主实验 images_2 (1/2 分辨率), 备选 images (标准) / images_4 (1/4)")
    parser.add_argument("--skip-existing", action="store_true", help="已通过的场景/seed 跳过")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    output_root = REPO_ROOT / args.output
    output_root.mkdir(parents=True, exist_ok=True)

    # 加载已有结果 (增量)
    results_csv = output_root / "qa_baseline_results.csv"
    existing = []
    existing_keys = set()
    if args.skip_existing and results_csv.is_file():
        with results_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("passed") in ("True", "true", "1"):
                    existing.append(row)
                    existing_keys.add((row["scene"], int(row["seed"])))

    fieldnames = ["scene", "seed", "train_exit", "render_exit", "metrics_exit", "psnr", "threshold", "passed", "images_dir"]
    new_results = []
    all_passed = True
    for scene in scenes:
        for seed in seeds:
            if (scene, seed) in existing_keys:
                print(f"[QA] skip {scene} seed={seed} (existing pass)")
                continue
            res = run_one(scene, seed, output_root, args.iterations, hold=8,
                          python=args.python, images_dir=args.images_dir)
            new_results.append(res)
            print(f"  PSNR={res.get('psnr')}, threshold={res.get('threshold')}, passed={res.get('passed')}")
            if not res.get("passed"):
                all_passed = False

    # 合并写 csv
    with results_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not results_csv.exists() or results_csv.stat().st_size == 0:
            writer.writeheader()
        for row in new_results:
            writer.writerow({k: row.get(k) for k in fieldnames})

    # 写阈值 json (按分辨率)
    thresholds_path = output_root / "baseline_thresholds.json"
    thresholds_data = {
        "images_dir": args.images_dir,
        "thresholds": get_thresholds_for_resolution(args.images_dir),
    }
    thresholds_path.write_text(json.dumps(thresholds_data, indent=2), encoding="utf-8")

    # 写汇总报告
    report = {
        "images_dir": args.images_dir,
        "scenes_tested": scenes,
        "seeds_tested": seeds,
        "thresholds": get_thresholds_for_resolution(args.images_dir),
        "results": new_results + existing,
        "all_passed": all_passed,
    }
    report_path = output_root / "qa_baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 写 markdown
    md_lines = [
        f"# QA Baseline Check Report (images_dir={args.images_dir})",
        "",
        f"阈值: {get_thresholds_for_resolution(args.images_dir)}",
        "",
        "| scene | seed | PSNR | threshold | passed | images_dir |",
        "|---|---|---|---|---|---|",
    ]
    for row in new_results + existing:
        md_lines.append(
            f"| {row.get('scene')} | {row.get('seed')} | {row.get('psnr')} | {row.get('threshold')} | {row.get('passed')} | {row.get('images_dir', args.images_dir)} |"
        )
    md_lines.append("")
    md_lines.append(f"**All passed: {all_passed}**")
    (output_root / "qa_baseline_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n[QA] All passed: {all_passed}")
    print(f"  Report: {report_path}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
