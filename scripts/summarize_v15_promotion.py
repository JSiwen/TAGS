#!/usr/bin/env python3
"""Aggregate v15 stage-selection JSON decisions into a promotion summary."""

import argparse
import json
from pathlib import Path


ROLE_LABELS = {
    "mainline": "Quality-primary (mainline)",
    "probe": "Quality-protect probe",
    "perceptual_branch": "Perceptual branch",
}

ROLE_SLOT = {
    "mainline": "default_promote",
    "probe": "quality_candidate",
    "perceptual_branch": "perceptual_candidate",
}


def read_decision(path: Path):
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["source_json"] = str(path)
    return data


def track_metrics(decision, track):
    if not decision:
        return None
    if track == "perceptual":
        return decision.get("perceptual_recommended_metrics") or decision.get("recommended_metrics")
    if track == "quality":
        return decision.get("quality_recommended_metrics") or decision.get("recommended_metrics")
    return decision.get("recommended_metrics")


def pick_role(decisions, role, track="quality"):
    """Pick best decision for a role when round-1/round-2 duplicates exist."""
    matches = [d for d in decisions if d.get("role") == role]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    def sort_key(decision):
        metrics = track_metrics(decision, track) or {}
        psnr = float(metrics.get("PSNR", -1e9))
        ssim = float(metrics.get("SSIM", -1e9))
        lpips = float(metrics.get("LPIPS", 1e9))
        if track == "perceptual":
            # Lower LPIPS first; keep PSNR as soft preference.
            return (lpips, -psnr, -ssim)
        return (-psnr, -ssim, lpips)

    return sorted(matches, key=sort_key)[0]


def track_method(decision, track):
    if not decision:
        return None, None, None
    if track == "perceptual":
        method = decision.get("perceptual_recommended_method") or decision.get("recommended_method")
        iter_ = decision.get("perceptual_recommended_iter") or decision.get("recommended_iter")
        reason = decision.get("perceptual_recommendation_reason") or decision.get("recommendation_reason")
        return method, iter_, reason
    method = decision.get("quality_recommended_method") or decision.get("recommended_method")
    iter_ = decision.get("quality_recommended_iter") or decision.get("recommended_iter")
    reason = decision.get("quality_recommendation_reason") or decision.get("recommendation_reason")
    return method, iter_, reason


def metrics_line(decision, track="default"):
    if track == "default":
        metrics = track_metrics(decision, "default")
    else:
        metrics = track_metrics(decision, track)
    if not metrics:
        return None
    if not isinstance(metrics.get("PSNR"), (int, float)):
        return None
    return (
        f"PSNR {metrics.get('PSNR', 'n/a'):.4f}, "
        f"SSIM {metrics.get('SSIM', 'n/a'):.6f}, "
        f"LPIPS {metrics.get('LPIPS', 'n/a'):.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Summarize v15 checkpoint promotion decisions.")
    parser.add_argument("--stage_jsons", nargs="+", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    args = parser.parse_args()

    decisions = []
    for path in args.stage_jsons:
        decision = read_decision(path)
        if decision:
            decisions.append(decision)

    if not decisions:
        raise SystemExit("No stage selection JSON files found.")

    mainline = pick_role(decisions, "mainline", track="quality")
    probe = pick_role(decisions, "probe", track="quality")
    perceptual = pick_role(decisions, "perceptual_branch", track="perceptual")

    mainline_method, mainline_iter, mainline_reason = track_method(mainline, "quality")
    probe_method, probe_iter, probe_reason = track_method(probe, "quality")
    perceptual_method, perceptual_iter, perceptual_reason = track_method(perceptual, "perceptual")

    # Prefer probe as quality_candidate when available; otherwise fall back to mainline quality track.
    if probe_method:
        quality_candidate_case = probe
        quality_candidate_method = probe_method
        quality_candidate_iter = probe_iter
        quality_candidate_reason = probe_reason
    else:
        quality_candidate_case = mainline
        quality_candidate_method = mainline_method
        quality_candidate_iter = mainline_iter
        quality_candidate_reason = mainline_reason

    slots = {
        "default_promote": {
            "case": mainline.get("case") if mainline else None,
            "role": "mainline" if mainline else None,
            "method": mainline_method,
            "iter": mainline_iter,
            "reason": mainline_reason,
            "metrics": (mainline or {}).get("quality_recommended_metrics")
            or (mainline or {}).get("recommended_metrics"),
        },
        "quality_candidate": {
            "case": quality_candidate_case.get("case") if quality_candidate_case else None,
            "role": quality_candidate_case.get("role") if quality_candidate_case else None,
            "method": quality_candidate_method,
            "iter": quality_candidate_iter,
            "reason": quality_candidate_reason,
            "metrics": (quality_candidate_case or {}).get("quality_recommended_metrics")
            or (quality_candidate_case or {}).get("recommended_metrics"),
        },
        "perceptual_candidate": {
            "case": perceptual.get("case") if perceptual else None,
            "role": "perceptual_branch" if perceptual else None,
            "method": perceptual_method,
            "iter": perceptual_iter,
            "reason": perceptual_reason,
            "metrics": (perceptual or {}).get("perceptual_recommended_metrics")
            or (perceptual or {}).get("recommended_metrics"),
        },
    }

    summary = {
        "cases": decisions,
        "slots": slots,
        "mainline_recommended_method": mainline_method,
        "mainline_recommended_iter": mainline_iter,
        "probe_recommended_method": probe_method,
        "probe_recommended_iter": probe_iter,
        "perceptual_recommended_method": perceptual_method,
        "perceptual_recommended_iter": perceptual_iter,
        "default_promote_method": slots["default_promote"]["method"],
        "default_promote_iter": slots["default_promote"]["iter"],
        "quality_candidate_method": slots["quality_candidate"]["method"],
        "quality_candidate_iter": slots["quality_candidate"]["iter"],
        "perceptual_candidate_method": slots["perceptual_candidate"]["method"],
        "perceptual_candidate_iter": slots["perceptual_candidate"]["iter"],
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.out_md:
        lines = [
            "# v15 Promotion Summary",
            "",
            "Checkpoint promotion decisions aggregated from stage-selection JSON outputs.",
            "",
            "| Case | Role | Slot | Quality | Perceptual | Default | Reason |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for decision in decisions:
            role = decision.get("role", "n/a")
            slot = ROLE_SLOT.get(role, "n/a")
            q_method = decision.get("quality_recommended_method") or decision.get("recommended_method")
            q_iter = decision.get("quality_recommended_iter") or decision.get("recommended_iter")
            p_method = decision.get("perceptual_recommended_method") or decision.get("recommended_method")
            p_iter = decision.get("perceptual_recommended_iter") or decision.get("recommended_iter")
            d_method = decision.get("recommended_method", "n/a")
            d_iter = decision.get("recommended_iter", "n/a")
            reason = decision.get("recommendation_reason", "n/a")
            lines.append(
                f"| `{decision.get('case', 'n/a')}` | {role} | `{slot}` | "
                f"`{q_method}` @ {q_iter} | `{p_method}` @ {p_iter} | "
                f"`{d_method}` @ {d_iter} | {reason} |"
            )

        lines.extend(["", "## Promotion guidance", ""])

        if mainline:
            q_metrics = metrics_line(mainline, "quality")
            lines.append(
                f"- **`default_promote` / {ROLE_LABELS['mainline']}:** "
                f"promote `{mainline_method}` @ iter {mainline_iter}."
            )
            if q_metrics:
                lines.append(f"  - Metrics: {q_metrics}")
            if mainline_reason:
                lines.append(f"  - Reason: {mainline_reason}")

        if probe:
            p_metrics = metrics_line(probe, "quality")
            lines.append(
                f"- **`quality_candidate` / {ROLE_LABELS['probe']}:** "
                f"`{probe_method}` @ iter {probe_iter} as quality-protect candidate."
            )
            if p_metrics:
                lines.append(f"  - Metrics: {p_metrics}")
            if probe_reason:
                lines.append(f"  - Reason: {probe_reason}")
        elif mainline:
            lines.append(
                "- **`quality_candidate`:** no probe case; reuse mainline quality track "
                f"`{mainline_method}` @ iter {mainline_iter}."
            )

        if perceptual:
            pe_metrics = metrics_line(perceptual, "perceptual")
            lines.append(
                f"- **`perceptual_candidate` / {ROLE_LABELS['perceptual_branch']}:** "
                f"`{perceptual_method}` @ iter {perceptual_iter} for LPIPS-first delivery."
            )
            if pe_metrics:
                lines.append(f"  - Metrics: {pe_metrics}")
            if perceptual_reason:
                lines.append(f"  - Reason: {perceptual_reason}")

        lines.append("- Do not promote structure/floor variants from the legacy ablation as defaults.")
        lines.append(
            "- Rank / A100 handoff should use these three slots, not the final-iteration ablation ranking alone."
        )
        lines.append("")

        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(lines), encoding="utf-8")

    if args.out_md:
        print(args.out_md)
    elif args.out_json:
        print(args.out_json)


if __name__ == "__main__":
    main()
