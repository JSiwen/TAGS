#!/usr/bin/env python3
"""Summarize vr_room E2 quality-versus-Gaussian-budget runs."""

import argparse
import csv
import json
import re
from pathlib import Path


LABEL_TO_BUDGET = {
    "1m": 1_000_000,
    "1.5m": 1_500_000,
    "2m": 2_000_000,
    "3m": 3_000_000,
    "unlimited": 0,
}


def read_jsonl(path):
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def budget_from_case(case_dir):
    prefix = "vr_room_e2_budget_"
    label = case_dir.name[len(prefix) :] if case_dir.name.startswith(prefix) else case_dir.name
    if label in LABEL_TO_BUDGET:
        return LABEL_TO_BUDGET[label], label
    match = re.fullmatch(r"(\d+(?:\.\d+)?)m", label)
    if match:
        return int(float(match.group(1)) * 1_000_000), label
    return None, label


def summarize_case(case_dir, budget_override=None, label_override=None):
    budget, label = budget_from_case(case_dir)
    if budget_override is not None:
        budget = budget_override
    if label_override is not None:
        label = label_override
    rows = read_jsonl(case_dir / "live_logs" / "stage_snapshots.jsonl")
    evaluated = [row for row in rows if (row.get("eval") or {}).get("test")]
    if not evaluated:
        return {
            "case": case_dir.name,
            "label": label,
            "budget": budget,
            "status": "missing_final_evaluation",
        }
    final_iteration = max(int(row.get("iteration", 0)) for row in evaluated)
    final = [row for row in evaluated if int(row.get("iteration", 0)) == final_iteration][-1]
    test = final["eval"]["test"]
    return {
        "case": case_dir.name,
        "label": label,
        "budget": budget,
        "status": "complete" if final_iteration >= 30_000 else "partial",
        "iteration": final_iteration,
        "num_points": int(final.get("num_points", 0)),
        "psnr": float(test["psnr"]),
        "ssim": float(test["ssim"]),
        "lpips": float(test["lpips"]),
        "l1": float(test["l1"]),
    }


def load_dn_reference(path):
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("psnr", "ssim", "lpips")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"DN reference is missing keys: {', '.join(missing)}")
    return {key: payload[key] for key in payload}


def analyze_curve(cases, dn_reference, saturation_threshold):
    fixed = sorted(
        (case for case in cases if case.get("status") == "complete" and case.get("budget", 0) > 0),
        key=lambda case: case["budget"],
    )
    saturation_budget = None
    for previous, current in zip(fixed, fixed[1:]):
        current["delta_psnr_from_previous_budget"] = current["psnr"] - previous["psnr"]
        current["delta_lpips_from_previous_budget"] = current["lpips"] - previous["lpips"]
        if (
            saturation_budget is None
            and current["delta_psnr_from_previous_budget"] < saturation_threshold
        ):
            saturation_budget = current["budget"]

    if dn_reference:
        for case in cases:
            if case.get("psnr") is None:
                continue
            case["beats_dn_psnr"] = case["psnr"] > float(dn_reference["psnr"])
            case["beats_dn_ssim"] = case["ssim"] > float(dn_reference["ssim"])
            case["beats_dn_lpips"] = case["lpips"] < float(dn_reference["lpips"])

    lpips_steps = [
        case["delta_lpips_from_previous_budget"]
        for case in fixed
        if "delta_lpips_from_previous_budget" in case
    ]
    if lpips_steps and all(delta < 0 for delta in lpips_steps):
        lpips_trend = "improves_with_more_gaussians"
    elif lpips_steps and all(delta > 0 for delta in lpips_steps):
        lpips_trend = "worsens_with_more_gaussians"
    else:
        lpips_trend = "mixed_or_insufficient"

    return {
        "psnr_saturation_threshold_db": saturation_threshold,
        "first_saturated_budget": saturation_budget,
        "lpips_trend": lpips_trend,
        "dn_reference": dn_reference,
        "cases": cases,
    }


def write_csv(path, cases):
    fields = [
        "case",
        "label",
        "budget",
        "status",
        "iteration",
        "num_points",
        "psnr",
        "ssim",
        "lpips",
        "l1",
        "delta_psnr_from_previous_budget",
        "delta_lpips_from_previous_budget",
        "beats_dn_psnr",
        "beats_dn_ssim",
        "beats_dn_lpips",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output")
    parser.add_argument(
        "--unlimited-case",
        default="output/vr_room_indoor_sensor_depth",
        help="Existing uncapped baseline directory; pass an empty string to exclude it.",
    )
    parser.add_argument("--dn-reference", default="")
    parser.add_argument("--saturation-threshold-db", type=float, default=0.05)
    parser.add_argument("--json-out", default="output/vr_room_e2_summary.json")
    parser.add_argument("--csv-out", default="output/vr_room_e2_summary.csv")
    args = parser.parse_args()

    root = Path(args.output_root)
    case_dirs = sorted(path for path in root.glob("vr_room_e2_budget_*") if path.is_dir())
    cases = [summarize_case(path) for path in case_dirs]
    if args.unlimited_case and Path(args.unlimited_case).is_dir():
        cases.append(
            summarize_case(
                Path(args.unlimited_case),
                budget_override=0,
                label_override="unlimited_existing_baseline",
            )
        )
    summary = analyze_curve(
        cases,
        load_dn_reference(args.dn_reference),
        args.saturation_threshold_db,
    )

    json_out = Path(args.json_out)
    csv_out = Path(args.csv_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_out, cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {json_out} and {csv_out}")


if __name__ == "__main__":
    main()
