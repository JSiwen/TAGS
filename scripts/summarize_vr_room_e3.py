#!/usr/bin/env python3
"""Summarize vr_room E3 prune ablation against pass criteria."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_DN = {
    "num_points": 1_120_000,
    "psnr": 26.4,
    "ssim": 0.86,
    "lpips": 0.134,
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


def load_json(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8-sig"))


def summarize_case(case_dir):
    rows = read_jsonl(case_dir / "live_logs" / "stage_snapshots.jsonl")
    evaluated = [row for row in rows if (row.get("eval") or {}).get("test")]
    profile = load_json(case_dir / "e3_profile.json") or {}
    fps_payload = load_json(case_dir / "fps_benchmark.json")
    if not evaluated:
        return {
            "case": case_dir.name,
            "status": "missing_final_evaluation",
            "profile": profile.get("profile"),
            "seed": profile.get("seed"),
        }
    final_iteration = max(int(row.get("iteration", 0)) for row in evaluated)
    final = [row for row in evaluated if int(row.get("iteration", 0)) == final_iteration][-1]
    test = final["eval"]["test"]
    fps_ms_p95 = None
    fps_std = None
    if fps_payload:
        fps_ms_p95 = fps_payload.get("ms_per_frame_p95")
        fps_std = fps_payload.get("fps_std")
    return {
        "case": case_dir.name,
        "status": "complete" if final_iteration >= 30_000 else "partial",
        "profile": profile.get("profile"),
        "seed": profile.get("seed"),
        "settings": profile.get("settings"),
        "iteration": final_iteration,
        "num_points": int(final.get("num_points", 0)),
        "psnr": float(test["psnr"]),
        "ssim": float(test["ssim"]),
        "lpips": float(test["lpips"]),
        "l1": float(test["l1"]),
        "fps": None if not fps_payload else float(fps_payload.get("fps")),
        "fps_std": None if fps_std is None else float(fps_std),
        "fps_ms_per_frame": None if not fps_payload else float(fps_payload.get("ms_per_frame")),
        "fps_ms_per_frame_p95": None if fps_ms_p95 is None else float(fps_ms_p95),
        "fps_peak_memory_mb": None
        if not fps_payload or fps_payload.get("peak_memory_mb") is None
        else float(fps_payload["peak_memory_mb"]),
    }


def collect_case_dirs(root, glob_spec):
    seen = set()
    dirs = []
    for pattern in [item.strip() for item in str(glob_spec).split(",") if item.strip()]:
        for path in sorted(root.glob(pattern)):
            if path.is_dir() and path.resolve() not in seen:
                seen.add(path.resolve())
                dirs.append(path)
    return dirs


def measure_fps(case_dir, python_bin, warmup, repeats):
    out = case_dir / "fps_benchmark.json"
    cmd = [
        python_bin,
        "scripts/benchmark_render_fps.py",
        "-m",
        str(case_dir),
        "--iteration",
        "30000",
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--json-out",
        str(out),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(REPO_ROOT)]
    )
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["TAGGS_CPU_KNN_INIT"] = env.get("TAGGS_CPU_KNN_INIT", "1")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    return load_json(out)


def evaluate_gates(case, unlimited, dn, max_points, max_psnr_drop, min_fps_speedup, baseline_fps):
    if case.get("psnr") is None:
        case["quality_pass"] = False
        case["full_pass"] = False
        case["pass"] = False
        case["fps_status"] = "pending"
        case["fail_reasons"] = ["missing_metrics"]
        return case
    reasons = []
    points_ok = case["num_points"] <= max_points
    psnr_drop = unlimited["psnr"] - case["psnr"]
    psnr_ok = psnr_drop <= max_psnr_drop
    dn_ok = case["psnr"] > dn["psnr"]
    quality_pass = points_ok and psnr_ok and dn_ok
    fps = case.get("fps")
    fps_speedup = None
    fps_ok = None
    fps_status = "pending"
    if fps is not None and baseline_fps and baseline_fps > 0:
        fps_speedup = fps / baseline_fps
        fps_ok = fps_speedup >= min_fps_speedup
        fps_status = "pass" if fps_ok else "fail"
    elif fps is None:
        reasons.append("fps_pending")
    if not points_ok:
        reasons.append(f"points>{max_points}")
    if not psnr_ok:
        reasons.append(f"psnr_drop>{max_psnr_drop}")
    if not dn_ok:
        reasons.append("psnr_not_above_dn")
    if fps_ok is False:
        reasons.append(f"fps_speedup<{min_fps_speedup}")
    full_pass = quality_pass and bool(fps_ok)
    case["psnr_drop_vs_unlimited"] = psnr_drop
    case["beats_dn_psnr"] = dn_ok
    case["fps_speedup_vs_unlimited"] = fps_speedup
    case["quality_pass"] = quality_pass
    case["full_pass"] = full_pass
    case["fps_status"] = fps_status
    case["gates"] = {
        "points_le_2m": points_ok,
        "psnr_drop_le_0_3db": psnr_ok,
        "psnr_above_dn": dn_ok,
        "fps_speedup_ge_1_5x": fps_ok,
    }
    case["pass"] = full_pass
    case["fail_reasons"] = reasons
    return case


def recommend_top_profiles(cases, limit=2):
    ranked = sorted(
        (
            case
            for case in cases
            if case.get("psnr") is not None and case.get("status") == "complete"
        ),
        key=lambda case: (
            0 if case.get("quality_pass") else 1,
            0 if case.get("full_pass") else 1,
            case.get("psnr_drop_vs_unlimited", 99),
            case.get("num_points", 10**12),
        ),
    )
    unique = []
    seen = set()
    for case in ranked:
        profile = case.get("profile") or case.get("case")
        if profile in seen:
            continue
        seen.add(profile)
        unique.append(profile)
        if len(unique) >= limit:
            break
    return unique


def write_csv(path, cases):
    fields = [
        "case",
        "profile",
        "seed",
        "status",
        "num_points",
        "psnr",
        "ssim",
        "lpips",
        "psnr_drop_vs_unlimited",
        "beats_dn_psnr",
        "fps",
        "fps_std",
        "fps_ms_per_frame_p95",
        "fps_speedup_vs_unlimited",
        "quality_pass",
        "full_pass",
        "fps_status",
        "pass",
        "fail_reasons",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            row = dict(case)
            row["fail_reasons"] = ";".join(case.get("fail_reasons") or [])
            writer.writerow(row)


def resolve_unlimited_metrics(args):
    unlimited_dir = Path(args.unlimited_case)
    if unlimited_dir.is_dir():
        unlimited_case = summarize_case(unlimited_dir)
        if unlimited_case.get("psnr") is not None:
            return {
                "num_points": unlimited_case["num_points"],
                "psnr": unlimited_case["psnr"],
                "ssim": unlimited_case["ssim"],
                "lpips": unlimited_case["lpips"],
            }, unlimited_dir
    ref = load_json(args.unlimited_reference)
    if ref and ref.get("psnr") is not None:
        print(
            f"[warn] unlimited case missing live metrics; using {args.unlimited_reference}",
            file=sys.stderr,
        )
        return {
            "num_points": int(ref.get("num_points", 0)),
            "psnr": float(ref["psnr"]),
            "ssim": float(ref.get("ssim", 0.0)),
            "lpips": float(ref.get("lpips", 0.0)),
        }, unlimited_dir if unlimited_dir.is_dir() else None
    raise SystemExit(
        "Missing unlimited baseline. Provide a completed --unlimited-case "
        "with stage snapshots or a JSON --unlimited-reference."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output")
    parser.add_argument(
        "--case-glob",
        default="vr_room_e3_v2_*,vr_room_e3_v3cap_*",
        help="Comma-separated globs under --output-root",
    )
    parser.add_argument("--unlimited-reference", default="")
    parser.add_argument("--dn-reference", default="output/vr_room_dn_reference.json")
    parser.add_argument("--unlimited-case", default="output/vr_room_indoor_sensor_depth")
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--max-psnr-drop", type=float, default=0.3)
    parser.add_argument("--min-fps-speedup", type=float, default=1.5)
    parser.add_argument("--measure-fps", type=int, default=0)
    parser.add_argument("--fps-warmup", type=int, default=10)
    parser.add_argument("--fps-repeats", type=int, default=30)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--json-out", default="output/vr_room_e3_summary.json")
    parser.add_argument("--csv-out", default="output/vr_room_e3_summary.csv")
    args = parser.parse_args()

    root = Path(args.output_root)
    case_dirs = collect_case_dirs(root, args.case_glob)
    unlimited_metrics, unlimited_dir = resolve_unlimited_metrics(args)

    dn_metrics = dict(DEFAULT_DN)
    dn_ref = load_json(args.dn_reference)
    if dn_ref and "psnr" in dn_ref:
        dn_metrics.update({k: dn_ref[k] for k in ("num_points", "psnr", "ssim", "lpips") if k in dn_ref})
    else:
        print(
            f"[warn] DN reference missing at {args.dn_reference}; using documented defaults. "
            "LPIPS is not a strict gate.",
            file=sys.stderr,
        )

    baseline_fps = None
    if args.measure_fps:
        if unlimited_dir is None or not unlimited_dir.is_dir():
            raise SystemExit("FPS measurement requires --unlimited-case directory.")
        fps_payload = measure_fps(unlimited_dir, args.python_bin, args.fps_warmup, args.fps_repeats)
        baseline_fps = float(fps_payload["fps"])
        (unlimited_dir / "fps_benchmark.json").write_text(
            json.dumps(fps_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif unlimited_dir is not None:
        fps_payload = load_json(unlimited_dir / "fps_benchmark.json")
        if fps_payload and fps_payload.get("fps"):
            baseline_fps = float(fps_payload["fps"])

    cases = []
    for case_dir in case_dirs:
        if args.measure_fps:
            try:
                measure_fps(case_dir, args.python_bin, args.fps_warmup, args.fps_repeats)
            except subprocess.CalledProcessError as exc:
                print(f"[warn] FPS failed for {case_dir.name}: {exc}", file=sys.stderr)
        case = summarize_case(case_dir)
        evaluate_gates(
            case,
            unlimited_metrics,
            dn_metrics,
            args.max_points,
            args.max_psnr_drop,
            args.min_fps_speedup,
            baseline_fps,
        )
        cases.append(case)

    summary = {
        "pass_criteria": {
            "max_points": args.max_points,
            "max_psnr_drop_db": args.max_psnr_drop,
            "dn_psnr": dn_metrics["psnr"],
            "min_fps_speedup": args.min_fps_speedup,
            "unlimited_psnr": unlimited_metrics["psnr"],
            "unlimited_points": unlimited_metrics["num_points"],
            "baseline_fps": baseline_fps,
            "dn_lpips_note": "DN LPIPS is an external reference, not a strict gate.",
        },
        "quality_passing_cases": [c["case"] for c in cases if c.get("quality_pass")],
        "passing_cases": [c["case"] for c in cases if c.get("full_pass")],
        "recommended_top2_for_multiseed": recommend_top_profiles(cases, 2),
        "cases": cases,
    }

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
