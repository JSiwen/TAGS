#!/usr/bin/env python3
"""Summarize scaled v15 fast-test checkpoint proxy (28500 vs 30000)."""

import argparse
import json
from pathlib import Path


def read_method(results_path: Path, method: str):
    if not results_path.is_file():
        return None
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if method in data:
        return data[method]
    return None


def fmt_delta(before, after, key, higher_better):
    if before is None or after is None:
        return None
    b = float(before.get(key, 0.0))
    a = float(after.get(key, 0.0))
    delta = a - b
    if key == "LPIPS":
        improved = delta < 0
    else:
        improved = delta > 0 if higher_better else delta < 0
    return {"before": b, "after": a, "delta": delta, "improved": improved}


def main():
    parser = argparse.ArgumentParser(
        description="Compare scaled 28500-proxy vs final checkpoint in v15 fast test."
    )
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--primary_iter", type=int, required=True, help="Scaled 28500 proxy")
    parser.add_argument("--final_iter", type=int, required=True, help="Scaled 30000 proxy")
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    args = parser.parse_args()

    results_path = args.model_dir / "results.json"
    primary_method = f"ours_{args.primary_iter}"
    final_method = f"ours_{args.final_iter}"

    primary = read_method(results_path, primary_method)
    final = read_method(results_path, final_method)

    summary = {
        "model_dir": str(args.model_dir),
        "primary_method": primary_method,
        "final_method": final_method,
        "primary_metrics": primary,
        "final_metrics": final,
        "deltas": {},
        "recommendation": None,
    }

    if primary and final:
        for key, higher_better in (("PSNR", True), ("SSIM", True), ("LPIPS", False)):
            summary["deltas"][key] = fmt_delta(primary, final, key, higher_better)

        psnr_drop = summary["deltas"]["PSNR"]["delta"]
        ssim_drop = summary["deltas"]["SSIM"]["delta"]
        lpips_gain = -summary["deltas"]["LPIPS"]["delta"]

        if psnr_drop < 0 and ssim_drop < 0 and lpips_gain > 0:
            rec = (
                "末段出现 PSNR/SSIM 回落且 LPIPS 改善，与 v14/v15 观察一致；"
                "Candidate B（选 28500 等价点）值得在正式 30k 上验证。"
            )
        elif psnr_drop >= 0 and ssim_drop >= 0:
            rec = (
                "末段未观察到典型回落，可能迭代过少或缩放阶段不足；"
                "可尝试 FAST_TIER=8k 或正式 30k。"
            )
        else:
            rec = (
                "末段趋势混合，建议延长迭代（8k）或检查训练日志后再决定。"
            )
        summary["recommendation"] = rec

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.out_md:
        lines = [
            "# v15 Fast Checkpoint Summary",
            "",
            f"- Model: `{args.model_dir}`",
            f"- Primary (28500 proxy): `{primary_method}`",
            f"- Final (30000 proxy): `{final_method}`",
            "",
        ]
        if primary and final:
            lines.append("## Metrics")
            lines.append("")
            lines.append("| Checkpoint | PSNR | SSIM | LPIPS |")
            lines.append("| --- | --- | --- | --- |")
            lines.append(
                f"| {primary_method} | {primary.get('PSNR', 'n/a')} | "
                f"{primary.get('SSIM', 'n/a')} | {primary.get('LPIPS', 'n/a')} |"
            )
            lines.append(
                f"| {final_method} | {final.get('PSNR', 'n/a')} | "
                f"{final.get('SSIM', 'n/a')} | {final.get('LPIPS', 'n/a')} |"
            )
            lines.append("")
            lines.append("## Delta (final - primary)")
            lines.append("")
            for key in ("PSNR", "SSIM", "LPIPS"):
                d = summary["deltas"].get(key)
                if d:
                    lines.append(f"- {key}: {d['delta']:+.6f}")
            lines.append("")
            lines.append("## Recommendation")
            lines.append("")
            lines.append(summary["recommendation"] or "n/a")
        else:
            lines.append("Missing metrics in `results.json`. Run render + metrics first.")
        lines.append("")
        lines.append("## Notes")
        lines.append("- 缩放迭代下的绝对值不能与 A100 30k gate 对比。")
        lines.append("- 重点看 28500-proxy 与 30000-proxy 的相对趋势。")
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(lines), encoding="utf-8")

    if not primary or not final:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
