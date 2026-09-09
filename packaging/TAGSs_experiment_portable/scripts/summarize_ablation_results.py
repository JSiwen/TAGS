#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


TARGETS = {
    "SSIM": 0.776,
    "PSNR": 25.50,
    "LPIPS": 0.190,
}


def read_main_metrics(model_dir: Path, method: str):
    results_path = model_dir / "results.json"
    if not results_path.is_file():
        return None
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if method not in data:
        return None
    return data[method]


def read_region_metrics(model_dir: Path):
    p = model_dir / "region_metrics_test.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    item = raw.get(str(model_dir), {})
    return item


def score_case(metrics):
    if not metrics:
        return -1e9
    score = 0.0
    score += 2.0 * (float(metrics.get("SSIM", 0.0)) - TARGETS["SSIM"])
    score += 0.2 * (float(metrics.get("PSNR", 0.0)) - TARGETS["PSNR"])
    score += 2.0 * (TARGETS["LPIPS"] - float(metrics.get("LPIPS", 1.0)))
    return score


def main():
    parser = argparse.ArgumentParser(description="Summarize 3DGS bicycle ablation results.")
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--method", default="ours_30000")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    args = parser.parse_args()

    summary = {"targets": TARGETS, "cases": {}}
    for case in args.cases:
        model_dir = args.output_root / case
        metrics = read_main_metrics(model_dir, args.method)
        region = read_region_metrics(model_dir)
        summary["cases"][case] = {
            "metrics": metrics or {},
            "region_metrics": region,
            "score": score_case(metrics),
        }

    ranked = sorted(summary["cases"].items(), key=lambda kv: kv[1]["score"], reverse=True)
    summary["ranked"] = [name for name, _ in ranked]
    summary["best_case"] = ranked[0][0] if ranked else None

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# Bicycle v5 Ablation Summary")
        lines.append("")
        lines.append(f"- Targets: {TARGETS}")
        lines.append(f"- Best case: {summary['best_case']}")
        lines.append("")
        lines.append("## Ranking")
        for idx, (name, info) in enumerate(ranked, start=1):
            lines.append(f"{idx}. {name} | score={info['score']:.6f} | metrics={info['metrics']}")
        lines.append("")
        lines.append("## Selection Rule")
        lines.append("- Prefer LPIPS lower while PSNR/SSIM do not regress against baseline.")
        lines.append("- If LPIPS improves but PSNR/SSIM drops significantly, treat as perceptual overfitting.")
        args.out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"best_case": summary["best_case"], "ranked": summary["ranked"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
