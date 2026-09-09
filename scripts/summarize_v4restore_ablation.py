#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


METRICS = ("SSIM", "PSNR", "LPIPS")
DEFAULT_CHRONIC_VIEWS = ("00008.png", "00011.png", "00016.png", "00018.png", "00021.png")


def read_json(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(value, digits=6):
    return f"{float(value):.{digits}f}"


def fmt_signed(value, digits=6):
    return f"{float(value):+.{digits}f}"


def metric_label(metrics):
    if not metrics:
        return "missing"
    return ", ".join(f"{metric}={fmt_float(metrics[metric])}" for metric in METRICS)


def metric_delta_label(deltas):
    if not deltas:
        return "missing"
    return ", ".join(f"{metric}={fmt_signed(deltas[metric])}" for metric in METRICS)


def read_metrics(model_dir, method):
    data = read_json(model_dir / "results.json")
    if not data or method not in data:
        return None
    return data[method]


def read_per_view(model_dir, method):
    data = read_json(model_dir / "per_view.json")
    if not data or method not in data:
        return None
    return data[method]


def read_done_marker(model_dir):
    marker_names = (
        ".v13_v8_chronic_spa_refine_case_done",
        ".v12_balanced_late_controller_case_done",
        ".v11_balanced_guarded_case_done",
        ".v10_balanced_fast_case_done",
        ".v9_v4restore_refine_case_done",
        ".v8_v4restore_refine_case_done",
        ".v4restore_ablation_case_done",
    )
    for marker_name in marker_names:
        marker = model_dir / marker_name
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip()
    return None


def find_final_iterations(model_dir):
    point_cloud_dir = model_dir / "point_cloud"
    if not point_cloud_dir.is_dir():
        return []

    iterations = []
    for path in point_cloud_dir.glob("iteration_*/point_cloud.ply"):
        try:
            iterations.append(int(path.parent.name.replace("iteration_", "")))
        except ValueError:
            continue
    return sorted(iterations)


def count_rendered_images(model_dir, method):
    method_dir = model_dir / "test" / method
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    return {
        "renders": len(list(renders_dir.glob("*.png"))) if renders_dir.is_dir() else 0,
        "gt": len(list(gt_dir.glob("*.png"))) if gt_dir.is_dir() else 0,
    }


def read_stage_artifacts(model_dir, stage_methods):
    if not stage_methods:
        return {}
    return {
        method: {
            "rendered_images": count_rendered_images(model_dir, method),
            "has_results": read_metrics(model_dir, method) is not None,
            "has_per_view": read_per_view(model_dir, method) is not None,
        }
        for method in stage_methods
    }


def read_case_artifacts(model_dir, method):
    final_iterations = find_final_iterations(model_dir)
    latest_iteration = final_iterations[-1] if final_iterations else None
    point_cloud_size_bytes = None
    if latest_iteration is not None:
        point_path = model_dir / "point_cloud" / f"iteration_{latest_iteration}" / "point_cloud.ply"
        if point_path.is_file():
            point_cloud_size_bytes = int(point_path.stat().st_size)
    return {
        "model_dir": str(model_dir),
        "done_marker": read_done_marker(model_dir),
        "has_results_json": (model_dir / "results.json").is_file(),
        "has_per_view_json": (model_dir / "per_view.json").is_file(),
        "final_iterations": final_iterations,
        "latest_iteration": latest_iteration,
        "point_cloud_size_bytes": point_cloud_size_bytes,
        "rendered_images": count_rendered_images(model_dir, method),
    }


def fmt_size(num_bytes):
    if num_bytes is None:
        return "missing"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def metric_delta(value, baseline, metric):
    delta = float(value) - float(baseline)
    return -delta if metric == "LPIPS" else delta


def compare_per_view(case_per_view, baseline_per_view):
    summary = {}
    for metric in METRICS:
        case_values = case_per_view.get(metric, {})
        base_values = baseline_per_view.get(metric, {})
        improved = []
        regressed = []
        for image_name, value in case_values.items():
            if image_name not in base_values:
                continue
            delta = float(value) - float(base_values[image_name])
            is_improved = delta < 0 if metric == "LPIPS" else delta > 0
            item = {
                "image": image_name,
                "delta": delta,
                "case": float(value),
                "baseline": float(base_values[image_name]),
            }
            if is_improved:
                improved.append(item)
            elif delta != 0:
                regressed.append(item)
        improved.sort(key=lambda item: abs(item["delta"]), reverse=True)
        regressed.sort(key=lambda item: abs(item["delta"]), reverse=True)
        summary[metric] = {
            "improved_count": len(improved),
            "regressed_count": len(regressed),
            "top_improvements": improved[:8],
            "top_regressions": regressed[:8],
        }
    return summary


def score_case(metrics, baseline_metrics, per_view_cmp):
    if not metrics:
        return -1e9
    score = 0.0
    score += 8.0 * metric_delta(metrics["SSIM"], baseline_metrics["SSIM"], "SSIM")
    score += 0.6 * metric_delta(metrics["PSNR"], baseline_metrics["PSNR"], "PSNR")
    score += 8.0 * metric_delta(metrics["LPIPS"], baseline_metrics["LPIPS"], "LPIPS")

    # Penalize broad regressions; a candidate must improve globally without damaging many views.
    for metric in METRICS:
        regressions = per_view_cmp.get(metric, {}).get("regressed_count", 0)
        improvements = per_view_cmp.get(metric, {}).get("improved_count", 0)
        score += 0.002 * (improvements - regressions)
    return score


def chronic_view_stats(case_per_view, baseline_per_view, chronic_views):
    stats = {
        "views": {},
        "max_psnr_drop": 0.0,
        "max_lpips_rise": 0.0,
        "avg_psnr_drop": 0.0,
        "avg_lpips_delta": 0.0,
        "failed_views": [],
    }
    psnr_drops = []
    lpips_deltas = []
    for image_name in chronic_views:
        psnr_case = case_per_view.get("PSNR", {}).get(image_name)
        psnr_base = baseline_per_view.get("PSNR", {}).get(image_name)
        lpips_case = case_per_view.get("LPIPS", {}).get(image_name)
        lpips_base = baseline_per_view.get("LPIPS", {}).get(image_name)
        if psnr_case is None or psnr_base is None or lpips_case is None or lpips_base is None:
            continue

        psnr_delta = float(psnr_case) - float(psnr_base)
        lpips_delta = float(lpips_case) - float(lpips_base)
        psnr_drop = max(0.0, -psnr_delta)
        lpips_rise = max(0.0, lpips_delta)
        stats["views"][image_name] = {
            "psnr_delta": psnr_delta,
            "lpips_delta": lpips_delta,
            "psnr_drop": psnr_drop,
            "lpips_rise": lpips_rise,
        }
        psnr_drops.append(psnr_drop)
        lpips_deltas.append(lpips_delta)
        stats["max_psnr_drop"] = max(stats["max_psnr_drop"], psnr_drop)
        stats["max_lpips_rise"] = max(stats["max_lpips_rise"], lpips_rise)
    if psnr_drops:
        stats["avg_psnr_drop"] = sum(psnr_drops) / len(psnr_drops)
    if lpips_deltas:
        stats["avg_lpips_delta"] = sum(lpips_deltas) / len(lpips_deltas)
    return stats


def evaluate_gates(metrics, baseline_metrics, per_view_cmp, chronic_stats, args):
    checks = {
        "global_psnr": float(metrics["PSNR"]) >= float(baseline_metrics["PSNR"]) - float(args.gate_psnr_drop),
        "global_ssim": float(metrics["SSIM"]) >= float(baseline_metrics["SSIM"]) - float(args.gate_ssim_drop),
        "global_lpips": float(metrics["LPIPS"]) <= float(baseline_metrics["LPIPS"]) + float(args.gate_lpips_rise),
        "psnr_regressed_views": per_view_cmp["PSNR"]["regressed_count"] <= int(args.gate_psnr_regressed_views),
        "lpips_regressed_views": per_view_cmp["LPIPS"]["regressed_count"] <= int(args.gate_lpips_regressed_views),
        "chronic_psnr_drop": chronic_stats["max_psnr_drop"] <= float(args.gate_chronic_psnr_drop),
        "chronic_avg_lpips_delta": chronic_stats["avg_lpips_delta"] <= float(args.gate_chronic_avg_lpips_delta),
    }
    return {"passed": all(checks.values()), "checks": checks}


def append_per_view_extremes(lines, metric, metric_info, title, key):
    rows = metric_info.get(key, [])
    lines.append(f"#### {metric} {title}")
    if not rows:
        lines.append("- None")
        lines.append("")
        return

    lines.append("| Image | Case | Baseline | Raw delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for item in rows[:5]:
        lines.append(
            f"| `{item['image']}` | {fmt_float(item['case'])} | "
            f"{fmt_float(item['baseline'])} | {fmt_signed(item['delta'])} |"
        )
    lines.append("")


def build_markdown(args, summary, ranked_items, missing_cases):
    lines = [
        "# v4_restore Ablation Summary",
        "",
        "## Run Overview",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Output root: `{args.output_root}`",
        f"- Method: `{args.method}`",
        f"- Baseline: `{args.baseline_case}`",
        f"- Gate reference: `{summary['gate_reference_case']}`",
        f"- Best case: `{summary['best_case']}`",
        f"- Completed cases: `{len(ranked_items)}/{len(args.cases)}`",
        f"- Missing or incomplete cases: `{len(missing_cases)}`",
        "",
        "## Baseline Metrics",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric in METRICS:
        lines.append(f"| {metric} | {fmt_float(summary['baseline_metrics'][metric])} |")
    if summary["gate_reference_case"] != summary["baseline_case"]:
        lines.extend([
            "",
            "## Gate Reference Metrics",
            "| Metric | Value |",
            "| --- | ---: |",
        ])
        for metric in METRICS:
            lines.append(f"| {metric} | {fmt_float(summary['gate_reference_metrics'][metric])} |")

    lines.extend([
        "",
        "## Ranking",
        "| Rank | Case | Gate | Score | SSIM | PSNR | LPIPS | delta SSIM | delta PSNR | delta LPIPS | gate delta SSIM | gate delta PSNR | gate delta LPIPS | View balance | Chronic max PSNR drop | Chronic avg LPIPS delta | Point cloud size |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ])
    for idx, (name, info) in enumerate(ranked_items, start=1):
        metrics = info["metrics"]
        deltas = info["delta_vs_baseline"]
        gate_deltas = info["delta_vs_gate_reference"]
        artifacts = info["artifacts"]
        gate_label = "pass" if info["gate"]["passed"] else "fail"
        view_balance = ", ".join(
            f"{metric}: +{info['per_view'][metric]['improved_count']}/"
            f"-{info['per_view'][metric]['regressed_count']}"
            for metric in METRICS
        )
        lines.append(
            f"| {idx} | `{name}` | `{gate_label}` | {fmt_float(info['score'])} | "
            f"{fmt_float(metrics['SSIM'])} | {fmt_float(metrics['PSNR'])} | "
            f"{fmt_float(metrics['LPIPS'])} | {fmt_signed(deltas['SSIM'])} | "
            f"{fmt_signed(deltas['PSNR'])} | {fmt_signed(deltas['LPIPS'])} | "
            f"{fmt_signed(gate_deltas['SSIM'])} | {fmt_signed(gate_deltas['PSNR'])} | "
            f"{fmt_signed(gate_deltas['LPIPS'])} | {view_balance} | "
            f"{fmt_float(info['chronic_stats']['max_psnr_drop'])} | "
            f"{fmt_signed(info['chronic_stats']['avg_lpips_delta'])} | "
            f"{fmt_size(artifacts.get('point_cloud_size_bytes'))} |"
        )

    if missing_cases:
        lines.extend([
            "",
            "## Missing Or Incomplete Cases",
            "| Case | Model dir | Results | Per-view | Latest iteration | Rendered test images |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ])
        for name in missing_cases:
            artifacts = summary["cases"][name]["artifacts"]
            rendered = artifacts["rendered_images"]
            lines.append(
                f"| `{name}` | `{artifacts['model_dir']}` | "
                f"{artifacts['has_results_json']} | {artifacts['has_per_view_json']} | "
                f"{artifacts['latest_iteration'] or 'missing'} | "
                f"{rendered['renders']}/{rendered['gt']} |"
            )

    lines.extend([
        "",
        "## Case Details",
    ])
    for name in args.cases:
        info = summary["cases"][name]
        artifacts = info["artifacts"]
        lines.extend([
            "",
            f"### {name}",
            f"- Model dir: `{artifacts['model_dir']}`",
            f"- Done marker: `{'yes' if artifacts['done_marker'] else 'no'}`",
            f"- Final iterations: `{artifacts['final_iterations'] or 'missing'}`",
            f"- Point cloud size: `{fmt_size(artifacts.get('point_cloud_size_bytes'))}`",
            (
                "- Rendered test images: "
                f"`renders={artifacts['rendered_images']['renders']}, "
                f"gt={artifacts['rendered_images']['gt']}`"
            ),
        ])

        if info.get("missing"):
            lines.append("- Status: `missing metrics or per_view data`")
            continue

        failed_checks = [
            check_name for check_name, passed in info["gate"]["checks"].items()
            if not bool(passed)
        ]
        lines.extend([
            f"- Metrics: `{metric_label(info['metrics'])}`",
            f"- Raw delta vs baseline: `{metric_delta_label(info['delta_vs_baseline'])}`",
            f"- Delta vs gate reference: `{metric_delta_label(info['delta_vs_gate_reference'])}`",
            f"- Score: `{fmt_float(info['score'])}`",
            f"- Gate: `{'pass' if info['gate']['passed'] else 'fail'}`",
            f"- Gate failed checks: `{', '.join(failed_checks) if failed_checks else 'none'}`",
            "",
            "| Metric | Improved views | Regressed views | Top improvement | Top regression |",
            "| --- | ---: | ---: | --- | --- |",
        ])
        for metric in METRICS:
            metric_info = info["per_view"][metric]
            top_improvement = metric_info["top_improvements"][0] if metric_info["top_improvements"] else None
            top_regression = metric_info["top_regressions"][0] if metric_info["top_regressions"] else None
            improvement_label = (
                f"`{top_improvement['image']}` ({fmt_signed(top_improvement['delta'])})"
                if top_improvement else "None"
            )
            regression_label = (
                f"`{top_regression['image']}` ({fmt_signed(top_regression['delta'])})"
                if top_regression else "None"
            )
            lines.append(
                f"| {metric} | {metric_info['improved_count']} | "
                f"{metric_info['regressed_count']} | {improvement_label} | "
                f"{regression_label} |"
            )

        lines.append("")
        for metric in METRICS:
            metric_info = info["per_view"][metric]
            append_per_view_extremes(lines, metric, metric_info, "Top Improvements", "top_improvements")
            append_per_view_extremes(lines, metric, metric_info, "Top Regressions", "top_regressions")

        lines.extend([
            "#### Chronic View Guard",
            "| View | PSNR delta | LPIPS delta |",
            "| --- | ---: | ---: |",
        ])
        for view_name in args.chronic_views:
            view_stats = info["chronic_stats"]["views"].get(view_name)
            if not view_stats:
                lines.append(f"| `{view_name}` | missing | missing |")
                continue
            lines.append(
                f"| `{view_name}` | {fmt_signed(view_stats['psnr_delta'])} | {fmt_signed(view_stats['lpips_delta'])} |"
            )
        lines.append("")
        if artifacts.get("stage_methods"):
            lines.extend([
                "#### Stage Checkpoints",
                "| Method | Rendered test images | Results | Per-view |",
                "| --- | ---: | --- | --- |",
            ])
            for method, stage_info in artifacts["stage_methods"].items():
                rendered = stage_info["rendered_images"]
                lines.append(
                    f"| `{method}` | {rendered['renders']}/{rendered['gt']} | "
                    f"{stage_info['has_results']} | {stage_info['has_per_view']} |"
                )
            lines.append("")

    lines.extend([
        "## Selection Rule",
        f"- Gate reference case: `{summary['gate_reference_case']}`.",
        f"- Gate global PSNR: `>= gate reference - {args.gate_psnr_drop} dB`.",
        f"- Gate global SSIM: `>= gate reference - {args.gate_ssim_drop}`.",
        f"- Gate global LPIPS: `<= gate reference + {args.gate_lpips_rise}`.",
        f"- Gate regressed views: `PSNR <= {args.gate_psnr_regressed_views}`, `LPIPS <= {args.gate_lpips_regressed_views}`.",
        f"- Gate chronic views ({', '.join(args.chronic_views)}): max PSNR drop `<= {args.gate_chronic_psnr_drop} dB`.",
        f"- Gate chronic average LPIPS delta: `<= {args.gate_chronic_avg_lpips_delta}`.",
        "- Ranking prefers gate-passing cases, then score.",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize ablations against the v4_restore reference.")
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--baseline_case", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--method", default="ours_30000")
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    parser.add_argument("--gate_reference_case", default=None)
    parser.add_argument("--stage_methods", nargs="*", default=[])
    parser.add_argument("--chronic_views", nargs="+", default=list(DEFAULT_CHRONIC_VIEWS))
    parser.add_argument("--gate_psnr_drop", type=float, default=0.01)
    parser.add_argument("--gate_ssim_drop", type=float, default=0.0002)
    parser.add_argument("--gate_lpips_rise", type=float, default=0.0005)
    parser.add_argument("--gate_psnr_regressed_views", type=int, default=9)
    parser.add_argument("--gate_lpips_regressed_views", type=int, default=12)
    parser.add_argument("--gate_chronic_psnr_drop", type=float, default=0.23)
    parser.add_argument("--gate_chronic_avg_lpips_delta", type=float, default=1e9)
    args = parser.parse_args()

    baseline_dir = args.output_root / args.baseline_case
    baseline_metrics = read_metrics(baseline_dir, args.method)
    baseline_per_view = read_per_view(baseline_dir, args.method)
    if baseline_metrics is None or baseline_per_view is None:
        raise SystemExit(f"Missing baseline metrics/per_view for {baseline_dir}")
    gate_reference_case = args.gate_reference_case or args.baseline_case
    gate_reference_dir = args.output_root / gate_reference_case
    gate_reference_metrics = read_metrics(gate_reference_dir, args.method)
    gate_reference_per_view = read_per_view(gate_reference_dir, args.method)
    if gate_reference_metrics is None or gate_reference_per_view is None:
        raise SystemExit(f"Missing gate reference metrics/per_view for {gate_reference_dir}")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(args.output_root),
        "method": args.method,
        "baseline_case": args.baseline_case,
        "baseline_metrics": baseline_metrics,
        "gate_reference_case": gate_reference_case,
        "gate_reference_metrics": gate_reference_metrics,
        "baseline_artifacts": read_case_artifacts(baseline_dir, args.method),
        "cases": {},
    }
    for case in args.cases:
        model_dir = args.output_root / case
        artifacts = read_case_artifacts(model_dir, args.method)
        artifacts["stage_methods"] = read_stage_artifacts(model_dir, args.stage_methods)
        metrics = read_metrics(model_dir, args.method)
        per_view = read_per_view(model_dir, args.method)
        if metrics is None or per_view is None:
            summary["cases"][case] = {"missing": True, "artifacts": artifacts}
            continue
        deltas = {metric: float(metrics[metric]) - float(baseline_metrics[metric]) for metric in METRICS}
        gate_deltas = {metric: float(metrics[metric]) - float(gate_reference_metrics[metric]) for metric in METRICS}
        per_view_cmp = compare_per_view(per_view, gate_reference_per_view)
        chronic_stats = chronic_view_stats(per_view, gate_reference_per_view, args.chronic_views)
        gate = evaluate_gates(metrics, gate_reference_metrics, per_view_cmp, chronic_stats, args)
        summary["cases"][case] = {
            "metrics": metrics,
            "delta_vs_baseline": deltas,
            "delta_vs_gate_reference": gate_deltas,
            "per_view": per_view_cmp,
            "score": score_case(metrics, gate_reference_metrics, per_view_cmp),
            "chronic_stats": chronic_stats,
            "gate": gate,
            "artifacts": artifacts,
        }

    ranked_items = [
        (name, info)
        for name, info in summary["cases"].items()
        if not info.get("missing")
    ]
    ranked_items.sort(key=lambda item: (not item[1]["gate"]["passed"], -item[1]["score"]))
    summary["ranked"] = [name for name, _ in ranked_items]
    summary["best_case"] = ranked_items[0][0] if ranked_items else None
    missing_cases = [
        name for name, info in summary["cases"].items()
        if info.get("missing")
    ]

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            build_markdown(args, summary, ranked_items, missing_cases),
            encoding="utf-8",
        )

    print(json.dumps({"best_case": summary["best_case"], "ranked": summary["ranked"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
