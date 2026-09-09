#!/usr/bin/env python3
"""Summarize v15 stage checkpoints for regression and chronic-view checks."""

import argparse
import json
import re
from pathlib import Path


DEFAULT_CHRONIC_VIEWS = ["00008.png", "00011.png", "00016.png", "00018.png", "00021.png"]
DEFAULT_PSNR_DROP_THRESHOLD = 0.05
DEFAULT_PERCEPTUAL_PSNR_BUDGET = 0.10
DEFAULT_CHRONIC_TRIPLE_LIMIT = 1


def read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def method_iter(method: str) -> int:
    return int(method.split("_", 1)[1])


def delta(results, ref_method, cur_method, metric):
    return float(results[cur_method][metric]) - float(results[ref_method][metric])


def per_view_delta(per_view, ref_method, cur_method, metric, view):
    return float(per_view[cur_method][metric][view]) - float(per_view[ref_method][metric][view])


def count_regressed(per_view, ref_method, cur_method, metric):
    if per_view is None or ref_method not in per_view or cur_method not in per_view:
        return None
    views = sorted(per_view[ref_method][metric].keys())
    if metric == "LPIPS":
        return sum(
            1
            for view in views
            if per_view[cur_method][metric][view] > per_view[ref_method][metric][view]
        )
    return sum(
        1
        for view in views
        if per_view[cur_method][metric][view] < per_view[ref_method][metric][view]
    )


def notes_for(method, best_psnr, best_ssim, best_lpips):
    notes = []
    if method == best_psnr:
        notes.append("PSNR peak")
    if method == best_ssim:
        notes.append("SSIM peak")
    if method == best_lpips:
        notes.append("LPIPS best")
    return ", ".join(notes)


def best_quality_method(methods, results):
    return max(methods, key=lambda method: (float(results[method]["PSNR"]), float(results[method]["SSIM"])))


def parse_role(config_label: str) -> str:
    match = re.search(r"role=([^,]+)", config_label or "")
    return match.group(1) if match else "unknown"


def metrics_snapshot(results, method):
    return {
        "method": method,
        "iter": method_iter(method),
        "PSNR": float(results[method]["PSNR"]),
        "SSIM": float(results[method]["SSIM"]),
        "LPIPS": float(results[method]["LPIPS"]),
    }


def count_chronic_triple_regressions(per_view, ref_method, cur_method, chronic_views):
    if not per_view or ref_method not in per_view or cur_method not in per_view:
        return None
    count = 0
    available_views = set(per_view[ref_method]["PSNR"].keys()) & set(per_view[cur_method]["PSNR"].keys())
    for view in chronic_views:
        if view not in available_views:
            continue
        d_psnr = per_view_delta(per_view, ref_method, cur_method, "PSNR", view)
        d_ssim = per_view_delta(per_view, ref_method, cur_method, "SSIM", view)
        d_lpips = per_view_delta(per_view, ref_method, cur_method, "LPIPS", view)
        if d_psnr < 0 and d_ssim < 0 and d_lpips > 0:
            count += 1
    return count


def select_quality_recommended(best_quality, final, final_vs_quality_psnr, psnr_drop_threshold):
    if best_quality == final:
        return final, "Final checkpoint matches quality peak (PSNR + SSIM tie-break)."
    if final_vs_quality_psnr < -psnr_drop_threshold:
        return (
            best_quality,
            (
                f"Final PSNR dropped {abs(final_vs_quality_psnr):.4f} dB vs quality peak; "
                f"promote `{best_quality}` instead of `{final}`."
            ),
        )
    return best_quality, "Quality-primary checkpoint selected (PSNR peak, SSIM tie-break)."


def select_perceptual_recommended(
    methods,
    results,
    per_view,
    quality_method,
    best_lpips,
    chronic_views,
    perceptual_psnr_budget,
    chronic_triple_limit,
):
    """Pick LPIPS-best among methods within PSNR budget of quality peak and chronic guard."""
    quality_psnr = float(results[quality_method]["PSNR"])
    candidates = []
    for method in methods:
        psnr_drop = quality_psnr - float(results[method]["PSNR"])
        if psnr_drop > perceptual_psnr_budget:
            continue
        chronic = count_chronic_triple_regressions(per_view, quality_method, method, chronic_views)
        if chronic is not None and chronic > chronic_triple_limit:
            continue
        candidates.append(method)

    if not candidates:
        return (
            quality_method,
            (
                f"No perceptual candidate within PSNR budget {perceptual_psnr_budget:.2f} dB "
                f"and chronic-triple limit {chronic_triple_limit}; fall back to quality peak."
            ),
        )

    selected = min(candidates, key=lambda method: float(results[method]["LPIPS"]))
    if selected == quality_method:
        return selected, "Quality peak is also the best in-budget perceptual checkpoint."
    if selected == best_lpips:
        return (
            selected,
            (
                f"LPIPS-best `{selected}` stays within PSNR budget "
                f"{perceptual_psnr_budget:.2f} dB of quality peak."
            ),
        )
    return (
        selected,
        (
            f"Selected bounded perceptual `{selected}` (global LPIPS-best `{best_lpips}` "
            f"exceeded PSNR budget or chronic guard)."
        ),
    )


def role_default_track(role: str) -> str:
    if role == "perceptual_branch":
        return "perceptual"
    return "quality"


def build_decision(
    results,
    per_view,
    methods,
    baseline,
    final,
    chronic_views,
    psnr_drop_threshold,
    config_label,
    model_dir,
    perceptual_psnr_budget=DEFAULT_PERCEPTUAL_PSNR_BUDGET,
    chronic_triple_limit=DEFAULT_CHRONIC_TRIPLE_LIMIT,
):
    best_psnr = max(methods, key=lambda method: results[method]["PSNR"])
    best_ssim = max(methods, key=lambda method: results[method]["SSIM"])
    best_lpips = min(methods, key=lambda method: results[method]["LPIPS"])
    best_quality = best_quality_method(methods, results)
    role = parse_role(config_label)

    final_psnr = float(results[final]["PSNR"])
    quality_psnr = float(results[best_quality]["PSNR"])
    final_vs_quality_psnr = final_psnr - quality_psnr

    quality_recommended, quality_reason = select_quality_recommended(
        best_quality, final, final_vs_quality_psnr, psnr_drop_threshold
    )
    perceptual_recommended, perceptual_reason = select_perceptual_recommended(
        methods=methods,
        results=results,
        per_view=per_view,
        quality_method=best_quality,
        best_lpips=best_lpips,
        chronic_views=chronic_views,
        perceptual_psnr_budget=perceptual_psnr_budget,
        chronic_triple_limit=chronic_triple_limit,
    )

    default_track = role_default_track(role)
    if default_track == "perceptual":
        recommended = perceptual_recommended
        reason = perceptual_reason
    else:
        recommended = quality_recommended
        reason = quality_reason

    chronic_triple_regressions = None
    if per_view and recommended != final:
        chronic_triple_regressions = count_chronic_triple_regressions(
            per_view, recommended, final, chronic_views
        )

    quality_metrics = metrics_snapshot(results, quality_recommended)
    perceptual_metrics = metrics_snapshot(results, perceptual_recommended)
    recommended_metrics = metrics_snapshot(results, recommended)

    return {
        "case": model_dir.name,
        "model_dir": str(model_dir),
        "role": role,
        "config_label": config_label,
        "baseline_method": baseline,
        "final_method": final,
        "best_quality_method": best_quality,
        "best_quality_iter": method_iter(best_quality),
        "best_perceptual_method": best_lpips,
        "best_perceptual_iter": method_iter(best_lpips),
        "best_psnr_method": best_psnr,
        "best_ssim_method": best_ssim,
        "quality_recommended_method": quality_recommended,
        "quality_recommended_iter": method_iter(quality_recommended),
        "quality_recommendation_reason": quality_reason,
        "quality_recommended_metrics": quality_metrics,
        "perceptual_recommended_method": perceptual_recommended,
        "perceptual_recommended_iter": method_iter(perceptual_recommended),
        "perceptual_recommendation_reason": perceptual_reason,
        "perceptual_recommended_metrics": perceptual_metrics,
        "default_track": default_track,
        "recommended_method": recommended,
        "recommended_iter": method_iter(recommended),
        "recommendation_reason": reason,
        "recommended_metrics": recommended_metrics,
        "psnr_drop_threshold_db": psnr_drop_threshold,
        "perceptual_psnr_budget_db": perceptual_psnr_budget,
        "chronic_triple_limit": chronic_triple_limit,
        "quality_to_final_delta": {
            "PSNR": delta(results, best_quality, final, "PSNR"),
            "SSIM": delta(results, best_quality, final, "SSIM"),
            "LPIPS": delta(results, best_quality, final, "LPIPS"),
        },
        "quality_to_perceptual_delta": {
            "PSNR": delta(results, best_quality, best_lpips, "PSNR"),
            "SSIM": delta(results, best_quality, best_lpips, "SSIM"),
            "LPIPS": delta(results, best_quality, best_lpips, "LPIPS"),
        },
        "quality_to_perceptual_recommended_delta": {
            "PSNR": delta(results, quality_recommended, perceptual_recommended, "PSNR"),
            "SSIM": delta(results, quality_recommended, perceptual_recommended, "SSIM"),
            "LPIPS": delta(results, quality_recommended, perceptual_recommended, "LPIPS"),
        },
        "final_vs_quality_psnr_drop_db": final_vs_quality_psnr,
        "chronic_triple_regressions_recommended_vs_final": chronic_triple_regressions,
        "metrics_by_method": {
            method: {
                "iter": method_iter(method),
                "PSNR": float(results[method]["PSNR"]),
                "SSIM": float(results[method]["SSIM"]),
                "LPIPS": float(results[method]["LPIPS"]),
            }
            for method in methods
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--out_md", required=True, type=Path)
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--config_label", default="")
    parser.add_argument("--baseline_method", default="")
    parser.add_argument("--final_method", default="")
    parser.add_argument("--chronic_views", default=",".join(DEFAULT_CHRONIC_VIEWS))
    parser.add_argument(
        "--psnr_drop_threshold",
        type=float,
        default=DEFAULT_PSNR_DROP_THRESHOLD,
        help="Recommend quality peak over final when final PSNR drops more than this (dB).",
    )
    parser.add_argument(
        "--perceptual_psnr_budget",
        type=float,
        default=DEFAULT_PERCEPTUAL_PSNR_BUDGET,
        help="Max PSNR drop vs quality peak allowed for perceptual-track promotion (dB).",
    )
    parser.add_argument(
        "--chronic_triple_limit",
        type=int,
        default=DEFAULT_CHRONIC_TRIPLE_LIMIT,
        help="Max chronic-view triple regressions allowed for perceptual-track candidates.",
    )
    args = parser.parse_args()

    results = read_json(args.model_dir / "results.json")
    if not results:
        raise SystemExit(f"Missing results.json under {args.model_dir}")
    per_view = read_json(args.model_dir / "per_view.json")

    methods = sorted(results.keys(), key=method_iter)
    best_psnr = max(methods, key=lambda method: results[method]["PSNR"])
    best_ssim = max(methods, key=lambda method: results[method]["SSIM"])
    best_lpips = min(methods, key=lambda method: results[method]["LPIPS"])
    best_quality = best_quality_method(methods, results)
    baseline = args.baseline_method if args.baseline_method in results else methods[0]
    final = args.final_method if args.final_method in results else methods[-1]
    chronic_views = [view.strip() for view in args.chronic_views.split(",") if view.strip()]
    decision = build_decision(
        results=results,
        per_view=per_view,
        methods=methods,
        baseline=baseline,
        final=final,
        chronic_views=chronic_views,
        psnr_drop_threshold=args.psnr_drop_threshold,
        config_label=args.config_label,
        model_dir=args.model_dir,
        perceptual_psnr_budget=args.perceptual_psnr_budget,
        chronic_triple_limit=args.chronic_triple_limit,
    )

    lines = [
        "# v15 Stage Selection Summary",
        "",
        f"Case: `{args.model_dir.name}`",
    ]
    if args.config_label:
        lines.append(f"Config: `{args.config_label}`")
    lines.extend(
        [
            "",
            "## 1. Stage global metrics",
            "",
            "| Stage | PSNR | SSIM | LPIPS | Notes |",
            "| ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for method in methods:
        row = results[method]
        lines.append(
            f"| {method_iter(method)} | {row['PSNR']:.4f} | {row['SSIM']:.6f} | "
            f"{row['LPIPS']:.6f} | {notes_for(method, best_psnr, best_ssim, best_lpips)} |"
        )

    lines.extend(
        [
            "",
            f"- PSNR peak: `{best_psnr}` ({results[best_psnr]['PSNR']:.4f})",
            f"- SSIM peak: `{best_ssim}` ({results[best_ssim]['SSIM']:.6f})",
            f"- LPIPS best: `{best_lpips}` ({results[best_lpips]['LPIPS']:.6f})",
            f"- Quality-primary: `{best_quality}` ({results[best_quality]['PSNR']:.4f} PSNR)",
            "",
            "## 2. Stage-to-stage deltas",
            "",
            "| From -> To | dPSNR | dSSIM | dLPIPS |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for prev, cur in zip(methods, methods[1:]):
        lines.append(
            f"| {method_iter(prev)} -> {method_iter(cur)} | "
            f"{delta(results, prev, cur, 'PSNR'):+.4f} | "
            f"{delta(results, prev, cur, 'SSIM'):+.6f} | "
            f"{delta(results, prev, cur, 'LPIPS'):+.6f} |"
        )

    if per_view:
        lines.extend(
            [
                "",
                "## 3. Regressed view counts vs previous stage",
                "",
                "| Stage | vs prev PSNR down | vs prev SSIM down | vs prev LPIPS up |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for prev, cur in zip(methods, methods[1:]):
            lines.append(
                f"| {method_iter(cur)} | "
                f"{count_regressed(per_view, prev, cur, 'PSNR')} | "
                f"{count_regressed(per_view, prev, cur, 'SSIM')} | "
                f"{count_regressed(per_view, prev, cur, 'LPIPS')} |"
            )

    lines.extend(
        [
            "",
            f"## 4. Baseline to final: `{baseline}` -> `{final}`",
            "",
            "| Metric | Baseline | Final | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric in ("PSNR", "SSIM", "LPIPS"):
        lines.append(
            f"| {metric} | {results[baseline][metric]:.6f} | {results[final][metric]:.6f} | "
            f"{delta(results, baseline, final, metric):+.6f} |"
        )

    if best_psnr != best_lpips:
        lines.extend(
            [
                "",
                f"## 5. Quality to perceptual: `{best_psnr}` -> `{best_lpips}`",
                "",
                "| Metric | Quality-primary | Perceptual-primary | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for metric in ("PSNR", "SSIM", "LPIPS"):
            lines.append(
                f"| {metric} | {results[best_psnr][metric]:.6f} | {results[best_lpips][metric]:.6f} | "
                f"{delta(results, best_psnr, best_lpips, metric):+.6f} |"
            )
        if per_view:
            lines.extend(
                [
                    "",
                    f"- PSNR down: {count_regressed(per_view, best_psnr, best_lpips, 'PSNR')}/25",
                    f"- SSIM down: {count_regressed(per_view, best_psnr, best_lpips, 'SSIM')}/25",
                    f"- LPIPS up: {count_regressed(per_view, best_psnr, best_lpips, 'LPIPS')}/25",
                ]
            )

    if per_view:
        lines.extend(
            [
                "",
                "## 6. Chronic views",
                "",
                f"Reference: `{baseline}` -> `{final}`",
                "",
                "| View | dPSNR | dSSIM | dLPIPS | Triple regression? |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        available_views = set(per_view[baseline]["PSNR"].keys()) & set(per_view[final]["PSNR"].keys())
        for view in chronic_views:
            if view not in available_views:
                continue
            d_psnr = per_view_delta(per_view, baseline, final, "PSNR", view)
            d_ssim = per_view_delta(per_view, baseline, final, "SSIM", view)
            d_lpips = per_view_delta(per_view, baseline, final, "LPIPS", view)
            triple = d_psnr < 0 and d_ssim < 0 and d_lpips > 0
            lines.append(f"| {view} | {d_psnr:+.4f} | {d_ssim:+.6f} | {d_lpips:+.6f} | {triple} |")

    lines.extend(
        [
            "",
            "## 7. Decision hints",
            "",
        ]
    )
    if best_psnr == best_lpips:
        lines.append("- Single checkpoint is supported by global PSNR and LPIPS peaks.")
    else:
        gap_psnr = delta(results, best_psnr, best_lpips, "PSNR")
        gap_ssim = delta(results, best_psnr, best_lpips, "SSIM")
        gap_lpips = delta(results, best_psnr, best_lpips, "LPIPS")
        lines.append(
            f"- Quality/perceptual split remains: dPSNR {gap_psnr:+.4f}, "
            f"dSSIM {gap_ssim:+.6f}, dLPIPS {gap_lpips:+.6f}."
        )
    lines.append("- Use chronic-view triple regressions as a guardrail before promoting a candidate.")
    lines.extend(
        [
            "",
            "## 8. Recommended checkpoints",
            "",
            f"- **Default track (`{decision['default_track']}`):** "
            f"`{decision['recommended_method']}` (iter {decision['recommended_iter']})",
            f"- Reason: {decision['recommendation_reason']}",
            (
                f"- Default metrics: PSNR {decision['recommended_metrics']['PSNR']:.4f}, "
                f"SSIM {decision['recommended_metrics']['SSIM']:.6f}, "
                f"LPIPS {decision['recommended_metrics']['LPIPS']:.6f}"
            ),
            (
                f"- **Quality track:** `{decision['quality_recommended_method']}` "
                f"(iter {decision['quality_recommended_iter']}) — "
                f"{decision['quality_recommendation_reason']}"
            ),
            (
                f"- **Perceptual track:** `{decision['perceptual_recommended_method']}` "
                f"(iter {decision['perceptual_recommended_iter']}) — "
                f"{decision['perceptual_recommendation_reason']}"
            ),
            (
                f"- Quality→perceptual recommended Δ: "
                f"dPSNR {decision['quality_to_perceptual_recommended_delta']['PSNR']:+.4f}, "
                f"dSSIM {decision['quality_to_perceptual_recommended_delta']['SSIM']:+.6f}, "
                f"dLPIPS {decision['quality_to_perceptual_recommended_delta']['LPIPS']:+.6f}"
            ),
            f"- Final vs quality dPSNR: {decision['quality_to_final_delta']['PSNR']:+.4f}",
            (
                f"- Perceptual PSNR budget: {decision['perceptual_psnr_budget_db']:.2f} dB; "
                f"chronic triple limit: {decision['chronic_triple_limit']}"
            ),
        ]
    )
    if decision["chronic_triple_regressions_recommended_vs_final"] is not None:
        lines.append(
            "- Chronic triple regressions (recommended vs final): "
            f"{decision['chronic_triple_regressions_recommended_vs_final']}/{len(chronic_views)}"
        )
    lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.out_md)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        print(args.out_json)


if __name__ == "__main__":
    main()
