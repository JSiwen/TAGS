#!/usr/bin/env python3
"""v2 计划 Step C QA gate: baseline+geo 验证 mono_depth 标定可靠性

在 Step B (no-pretrain baseline) 通过后, 跑 baseline+geo 验证 mono_depth 标定是否真的有效:
- 对比 "baseline+geo" vs "baseline" (Step B 已有) 的 PSNR
- 差异 ≥ 0.1 dB 视为标定有效
- 不达标说明 mono_depth 标定失败, 回到 Step A 重做

4 场景 × 3 seed (无需 5 seed, 速度优先)
30000 iter, baseline+geo = enable_geo_prior + geo_prior_depth + geo_prior_normal + 默认权重

用法:
  python scripts/qa_preprocess_reliability.py --scenes bicycle,garden,bonsai,room --seeds 0,1,2
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# 至少 0.1 dB 提升视为 mono_depth 标定有效
MIN_DELTA_PSNR = 0.10


def find_psnr_in_results(model_dir: Path) -> Optional[float]:
    cand = model_dir / "results.json"
    if not cand.is_file():
        return None
    payload = json.loads(cand.read_text(encoding="utf-8"))
    for method_dict in payload.values():
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
) -> Dict[str, Optional[float]]:
    """跑 baseline+geo (打开 geo_prior, 用 mono_depth, 但不开 SPA/LPIPS)"""
    group_id = "qa_reliability"
    train_dir = output_root / scene / group_id / "train" / f"seed{seed}"
    train_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[QA-C] === {scene} seed={seed} ===")
    print(f"  train -> {train_dir}")

    # 打开 geo_prior (depth + normal), 关掉其他模块
    train_cmd = [
        python, "train_optimized.py",
        "-s", f"data/{scene}",
        "-m", str(train_dir),
        "--eval",
        "--llffhold", str(hold),
        "--images", "images_4",
        "--iterations", str(iterations),
        "--seed", str(seed),
        # 开 geo_prior
        "--enable_geo_prior",
        "--geo_prior_depth",
        "--geo_prior_normal",
        # 默认权重 (depth_final=0.05, normal_final=0.02, peak=0.0, init=0.0; 实际由 warmup/decay 调度)
        # 关掉其他损失
        "--lpips_loss_weight_init", "0.0",
        "--lpips_loss_weight_final", "0.0",
        "--psnr_loss_weight_init", "0.0",
        "--psnr_loss_weight_final", "0.0",
        "--color_align_weight_init", "0.0",
        "--color_align_weight_final", "0.0",
        "--detail_loss_weight_init", "0.0",
        "--detail_loss_weight_final", "0.0",
        # 关掉 SPA
        "--sparse_mode", "none",
        "--prune_ratio2", "0.0",
        "--spa_prune_ratio", "0.0",
    ]
    print(f"  $ {' '.join(train_cmd)}")
    train_proc = subprocess.run(train_cmd, cwd=str(REPO_ROOT), check=False)
    if train_proc.returncode != 0:
        return {"scene": scene, "seed": seed, "train_exit": train_proc.returncode}

    # render
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
        return {"scene": scene, "seed": seed, "train_exit": 0, "render_exit": render_proc.returncode}

    # metrics
    metrics_cmd = [python, "metrics.py", "-m", str(train_dir)]
    print(f"  $ {' '.join(metrics_cmd)}")
    subprocess.run(metrics_cmd, cwd=str(REPO_ROOT), check=False)

    psnr_geo = find_psnr_in_results(train_dir)
    return {
        "scene": scene,
        "seed": seed,
        "train_exit": 0,
        "render_exit": 0,
        "psnr_geo": psnr_geo,
    }


def load_baseline_psnrs(baseline_csv: Path) -> Dict[str, float]:
    """读 qa_baseline_results.csv, 汇总为 {scene: mean_psnr}."""
    if not baseline_csv.is_file():
        return {}
    out: Dict[str, List[float]] = {}
    with baseline_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("passed") not in ("True", "true", "1"):
                continue
            try:
                psnr = float(row["psnr"])
            except (KeyError, ValueError):
                continue
            scene = row.get("scene")
            if not scene:
                continue
            out.setdefault(scene, []).append(psnr)
    return {k: (sum(v) / len(v)) for k, v in out.items() if v}


def main():
    parser = argparse.ArgumentParser(description="v2 QA gate Step C: 跑 baseline+geo 验证 mono_depth 标定可靠性")
    parser.add_argument("--scenes", default="bicycle,garden,bonsai,room")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--output", default="output/exp4_stats")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--baseline-csv", default="output/exp4_stats/qa_baseline_results.csv")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    output_root = REPO_ROOT / args.output
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_csv = REPO_ROOT / args.baseline_csv
    baseline_psnrs = load_baseline_psnrs(baseline_csv)
    if not baseline_psnrs:
        print(f"WARNING: no baseline_psnrs loaded from {baseline_csv}, 跳过对比")
    print(f"baseline_psnrs: {baseline_psnrs}")

    fieldnames = ["scene", "seed", "train_exit", "render_exit", "psnr_geo", "psnr_baseline", "delta", "passed"]
    new_results = []
    all_passed = True
    for scene in scenes:
        for seed in seeds:
            res = run_one(scene, seed, output_root, args.iterations, hold=8, python=args.python)
            base = baseline_psnrs.get(scene)
            psnr_geo = res.get("psnr_geo")
            delta = (psnr_geo - base) if (psnr_geo is not None and base is not None) else None
            passed = delta is not None and delta >= MIN_DELTA_PSNR
            res["psnr_baseline"] = base
            res["delta"] = delta
            res["passed"] = passed
            new_results.append(res)
            print(f"  psnr_geo={psnr_geo} baseline={base} delta={delta} passed={passed}")
            if not passed:
                all_passed = False

    # 写 csv
    results_csv = output_root / "qa_reliability_results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in new_results:
            writer.writerow({k: row.get(k) for k in fieldnames})

    # 写 json
    report = {
        "scenes_tested": scenes,
        "seeds_tested": seeds,
        "min_delta_psnr": MIN_DELTA_PSNR,
        "baseline_psnrs": baseline_psnrs,
        "results": new_results,
        "all_passed": all_passed,
    }
    (output_root / "qa_reliability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 写 markdown
    md_lines = [
        "# QA Preprocess Reliability Report",
        "",
        f"min_delta_psnr 阈值: {MIN_DELTA_PSNR} dB (baseline+geo 应至少比 baseline 高这么多)",
        "",
        "| scene | seed | PSNR baseline | PSNR baseline+geo | delta | passed |",
        "|---|---|---|---|---|---|",
    ]
    for row in new_results:
        md_lines.append(
            f"| {row['scene']} | {row['seed']} | {row.get('psnr_baseline')} | "
            f"{row.get('psnr_geo')} | {row.get('delta')} | {row.get('passed')} |"
        )
    md_lines.append("")
    md_lines.append(f"**All passed: {all_passed}**")
    (output_root / "qa_reliability_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n[QA-C] All passed: {all_passed}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
