#!/usr/bin/env python3
"""Summarize vr_room E4 perceptual ablation against paired gates."""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_E3_REFERENCE = {
    "psnr": 27.46973387400309,
    "ssim": 0.8791310569596669,
    "lpips": 0.222192958588638,
    "fps": 167.21602489556497,
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


def summarize_case(case_dir, final_iteration=30000):
    rows = read_jsonl(case_dir / "live_logs" / "stage_snapshots.jsonl")
    evaluated = [row for row in rows if (row.get("eval") or {}).get("test")]
    profile = load_json(case_dir / "e4_profile.json") or {}
    fps_payload = load_json(case_dir / "fps_benchmark.json")
    depth_payload = load_json(case_dir / "depth_per_view.json")
    if not evaluated:
        return {
            "case": case_dir.name,
            "status": "missing_final_evaluation",
            "profile": profile.get("profile"),
            "seed": profile.get("seed"),
        }
    final_rows = [row for row in evaluated if int(row.get("iteration", 0)) == final_iteration]
    if not final_rows:
        final_iteration = max(int(row.get("iteration", 0)) for row in evaluated)
        final_rows = [row for row in evaluated if int(row.get("iteration", 0)) == final_iteration]
    final = final_rows[-1]
    test = final["eval"]["test"]
    train_snapshot = final.get("train_snapshot") or {}
    geo_stats = aggregate_geo_logs(case_dir)
    return {
        "case": case_dir.name,
        "status": "complete" if final_iteration >= 30_000 else "partial",
        "profile": profile.get("profile"),
        "seed": profile.get("seed"),
        "perceptual": profile.get("perceptual"),
        "structure": profile.get("structure"),
        "iteration": final_iteration,
        "num_points": int(final.get("num_points", 0)),
        "psnr": float(test["psnr"]),
        "ssim": float(test["ssim"]),
        "lpips": float(test["lpips"]),
        "l1": float(test["l1"]),
        "fps": None if not fps_payload else float(fps_payload.get("fps")),
        "fps_std": None if not fps_payload else float(fps_payload.get("fps_std", 0.0) or 0.0),
        "fps_ms_per_frame_p95": None
        if not fps_payload
        else float(fps_payload.get("ms_per_frame_p95", 0.0) or 0.0),
        "fps_peak_memory_mb": None
        if not fps_payload or fps_payload.get("peak_memory_mb") is None
        else float(fps_payload["peak_memory_mb"]),
        "geo_train": geo_stats,
        "depth_prior": depth_payload.get("summary") if depth_payload else None,
        "final_geo_snapshot": {
            key: float(train_snapshot[key])
            for key in train_snapshot
            if key.startswith("geo/")
        },
    }


def aggregate_geo_logs(case_dir):
    rows = read_jsonl(case_dir / "live_logs" / "train_iter_metrics.jsonl")
    if not rows:
        rows = read_jsonl(case_dir / "live_logs" / "stage_snapshots.jsonl")
        rows = [
            {
                "iteration": row.get("iteration"),
                "stats": row.get("train_snapshot") or {},
            }
            for row in rows
        ]
    keys = (
        "geo/depth_loss",
        "geo/depth_weight",
        "geo/depth_mask_fraction",
        "geo/normal_loss",
        "geo/normal_weight",
        "geo/normal_mask_fraction",
    )
    buckets = {key: [] for key in keys}
    for row in rows:
        iteration = int(row.get("iteration", 0))
        if iteration > 20_000:
            continue
        stats = row.get("stats") or row.get("operation_losses") or row.get("train_snapshot") or {}
        for key in keys:
            if key in stats:
                buckets[key].append(float(stats[key]))
    out = {}
    for key, values in buckets.items():
        if not values:
            continue
        out[key] = {
            "mean": statistics.mean(values),
            "max": max(values),
            "min": min(values),
        }
    return out


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


def aggregate_profile(cases):
    complete = [case for case in cases if case.get("psnr") is not None]
    if not complete:
        return None
    seeds = sorted({int(case.get("seed", 0)) for case in complete})
    metrics = ("psnr", "ssim", "lpips", "num_points", "fps")
    agg = {
        "profile": complete[0].get("profile"),
        "seeds_present": seeds,
        "num_complete": len(complete),
    }
    for metric in metrics:
        values = [case[metric] for case in complete if case.get(metric) is not None]
        if not values:
            continue
        agg[f"{metric}_mean"] = statistics.mean(values)
        agg[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        if metric == "lpips":
            agg[f"{metric}_best"] = min(values)
            agg[f"{metric}_worst"] = max(values)
        else:
            agg[f"{metric}_best"] = max(values)
            agg[f"{metric}_worst"] = min(values)
    return agg


def compare_to_baseline(case, baseline):
    if case.get("psnr") is None or baseline.get("psnr") is None:
        return None
    lpips_improve_ratio = (baseline["lpips"] - case["lpips"]) / baseline["lpips"]
    return {
        "psnr_drop": baseline["psnr"] - case["psnr"],
        "ssim_drop": baseline["ssim"] - case["ssim"],
        "lpips_improve_ratio": lpips_improve_ratio,
        "lpips_improve_abs": baseline["lpips"] - case["lpips"],
        "fps_ratio": None
        if case.get("fps") is None or baseline.get("fps") in (None, 0)
        else case["fps"] / baseline["fps"],
    }


def evaluate_prescreen(delta, args):
    checks = {
        "lpips_improve_ge_3pct": delta["lpips_improve_ratio"] >= args.prescreen_lpips_improve,
        "psnr_drop_le_0_25db": delta["psnr_drop"] <= args.prescreen_psnr_drop,
        "ssim_drop_le_0_003": delta["ssim_drop"] <= args.prescreen_ssim_drop,
    }
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_final(case, baseline_case, delta, unlimited, args):
    reasons = []
    points_ok = case["num_points"] <= args.max_points
    if not points_ok:
        reasons.append(f"points>{args.max_points}")
    if delta is None:
        return {
            "pass": False,
            "quality_pass": False,
            "fail_reasons": ["missing_baseline_pair"],
        }

    lpips_ok = (
        delta["lpips_improve_ratio"] >= args.gate_lpips_improve
        and delta["lpips_improve_abs"] >= 0
    )
    psnr_ok = delta["psnr_drop"] <= args.gate_psnr_drop
    ssim_ok = delta["ssim_drop"] <= args.gate_ssim_drop
    if not lpips_ok:
        reasons.append("lpips_improve<5pct_or_worse")
    if not psnr_ok:
        reasons.append(f"psnr_drop>{args.gate_psnr_drop}")
    if not ssim_ok:
        reasons.append(f"ssim_drop>{args.gate_ssim_drop}")

    fps = case.get("fps")
    fps_speedup = None
    fps_ok = None
    if fps is not None and args.baseline_fps and args.baseline_fps > 0:
        fps_speedup = fps / args.baseline_fps
        fps_ok = fps_speedup >= args.min_fps_speedup
        if fps_ok is False:
            reasons.append(f"fps_speedup<{args.min_fps_speedup}")
    elif fps is None:
        reasons.append("fps_pending")

    baseline_fps = baseline_case.get("fps")
    fps_soft_warn = False
    if fps is not None and baseline_fps:
        if fps < 0.9 * baseline_fps:
            fps_soft_warn = True
            reasons.append("fps_drop_gt_10pct_vs_rgb_ssim")

    quality_pass = points_ok and lpips_ok and psnr_ok and ssim_ok
    full_pass = quality_pass and bool(fps_ok)
    return {
        "pass": full_pass,
        "quality_pass": quality_pass,
        "fps_status": "pending" if fps_ok is None else ("pass" if fps_ok else "fail"),
        "fps_speedup_vs_unlimited": fps_speedup,
        "fps_soft_warn": fps_soft_warn,
        "fail_reasons": reasons,
        "delta_vs_rgb_ssim": delta,
    }


def evaluate_multiseed(profile_cases, baseline_by_seed, args):
    complete = [case for case in profile_cases if case.get("psnr") is not None]
    if len(complete) < 3:
        return {"passed": False, "reason": "incomplete_seeds"}
    deltas = []
    for case in complete:
        seed = int(case.get("seed", 0))
        baseline = baseline_by_seed.get(seed)
        if baseline is None:
            return {"passed": False, "reason": f"missing_baseline_seed_{seed}"}
        deltas.append(compare_to_baseline(case, baseline))
    lpips_ratios = [delta["lpips_improve_ratio"] for delta in deltas]
    psnr_drops = [delta["psnr_drop"] for delta in deltas]
    ssim_drops = [delta["ssim_drop"] for delta in deltas]
    checks = {
        "all_seeds_lpips_not_worse": all(delta["lpips_improve_abs"] >= 0 for delta in deltas),
        "mean_lpips_improve_ge_5pct": statistics.mean(lpips_ratios) >= args.gate_lpips_improve,
        "mean_psnr_drop_le_0_2db": statistics.mean(psnr_drops) <= args.gate_psnr_drop,
        "worst_psnr_drop_le_0_25db": max(psnr_drops) <= args.gate_psnr_drop_worst,
        "mean_ssim_drop_le_0_002": statistics.mean(ssim_drops) <= args.gate_ssim_drop,
        "worst_ssim_drop_le_0_003": max(ssim_drops) <= args.gate_ssim_drop_worst,
        "all_points_le_2m": all(case["num_points"] <= args.max_points for case in complete),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "lpips_improve_ratio_mean": statistics.mean(lpips_ratios),
        "psnr_drop_mean": statistics.mean(psnr_drops),
        "ssim_drop_mean": statistics.mean(ssim_drops),
    }


def recommend_top_profiles(profile_aggs, baseline_name="rgb_ssim", limit=1, multiscale=False):
    ranked = []
    for profile, agg in profile_aggs.items():
        if profile == baseline_name:
            continue
        name = profile or ""
        is_ms = name.startswith("lpips_ms_")
        if multiscale != is_ms:
            continue
        if not agg.get("prescreen_pass"):
            continue
        ranked.append(
            (
                profile,
                agg.get("lpips_improve_ratio_mean", -999),
                agg.get("psnr_drop_mean", 999),
            )
        )
    ranked.sort(key=lambda item: (-item[1], item[2]))
    return [item[0] for item in ranked[:limit]]


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
        "psnr_drop_vs_rgb_ssim",
        "ssim_drop_vs_rgb_ssim",
        "lpips_improve_ratio_vs_rgb_ssim",
        "fps",
        "fps_speedup_vs_unlimited",
        "quality_pass",
        "pass",
        "fps_status",
        "fail_reasons",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            row = dict(case)
            delta = case.get("delta_vs_rgb_ssim") or {}
            row["psnr_drop_vs_rgb_ssim"] = delta.get("psnr_drop")
            row["ssim_drop_vs_rgb_ssim"] = delta.get("ssim_drop")
            row["lpips_improve_ratio_vs_rgb_ssim"] = delta.get("lpips_improve_ratio")
            row["fail_reasons"] = ";".join(case.get("fail_reasons") or [])
            writer.writerow(row)


def resolve_unlimited_fps(args):
    unlimited_dir = Path(args.unlimited_case)
    if args.measure_fps:
        if not unlimited_dir.is_dir():
            raise SystemExit("FPS measurement requires --unlimited-case directory.")
        fps_payload = measure_fps(unlimited_dir, args.python_bin, args.fps_warmup, args.fps_repeats)
        return float(fps_payload["fps"])
    fps_payload = load_json(unlimited_dir / "fps_benchmark.json")
    if fps_payload and fps_payload.get("fps"):
        return float(fps_payload["fps"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--case-glob", default="vr_room_e4_*")
    parser.add_argument("--baseline-profile", default="rgb_ssim")
    parser.add_argument("--e3-reference-json", default="")
    parser.add_argument("--unlimited-case", default="output/vr_room_indoor_sensor_depth")
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--min-fps-speedup", type=float, default=1.5)
    parser.add_argument("--gate-lpips-improve", type=float, default=0.05)
    parser.add_argument("--gate-psnr-drop", type=float, default=0.2)
    parser.add_argument("--gate-psnr-drop-worst", type=float, default=0.25)
    parser.add_argument("--gate-ssim-drop", type=float, default=0.002)
    parser.add_argument("--gate-ssim-drop-worst", type=float, default=0.003)
    parser.add_argument("--prescreen-lpips-improve", type=float, default=0.03)
    parser.add_argument("--prescreen-psnr-drop", type=float, default=0.25)
    parser.add_argument("--prescreen-ssim-drop", type=float, default=0.003)
    parser.add_argument("--measure-fps", type=int, default=0)
    parser.add_argument("--fps-warmup", type=int, default=10)
    parser.add_argument("--fps-repeats", type=int, default=30)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--json-out", default="output/vr_room_e4_summary.json")
    parser.add_argument("--csv-out", default="output/vr_room_e4_summary.csv")
    args = parser.parse_args()

    root = Path(args.output_root)
    case_dirs = collect_case_dirs(root, args.case_glob)
    baseline_fps = resolve_unlimited_fps(args)

    e3_ref = dict(DEFAULT_E3_REFERENCE)
    if args.e3_reference_json:
        loaded = load_json(args.e3_reference_json)
        if loaded:
            e3_ref.update({k: loaded[k] for k in ("psnr", "ssim", "lpips", "fps") if k in loaded})

    cases = []
    for case_dir in case_dirs:
        if args.measure_fps:
            try:
                measure_fps(case_dir, args.python_bin, args.fps_warmup, args.fps_repeats)
            except subprocess.CalledProcessError as exc:
                print(f"[warn] FPS failed for {case_dir.name}: {exc}", file=sys.stderr)
        cases.append(summarize_case(case_dir))

    by_profile = {}
    for case in cases:
        profile = case.get("profile") or case.get("case")
        by_profile.setdefault(profile, []).append(case)

    baseline_by_seed = {}
    for case in by_profile.get(args.baseline_profile, []):
        baseline_by_seed[int(case.get("seed", 0))] = case

    enriched = []
    for case in cases:
        seed = int(case.get("seed", 0))
        baseline_case = baseline_by_seed.get(seed)
        delta = compare_to_baseline(case, baseline_case) if baseline_case else None
        if delta is not None:
            case["delta_vs_rgb_ssim"] = delta
        if case.get("profile") != args.baseline_profile and delta is not None:
            prescreen = evaluate_prescreen(delta, args)
            case["prescreen"] = prescreen
        gate = evaluate_final(case, baseline_case or {}, delta, e3_ref, args)
        case.update({k: gate[k] for k in ("pass", "quality_pass", "fps_status", "fps_speedup_vs_unlimited", "fail_reasons") if k in gate})
        if gate.get("fps_soft_warn"):
            case.setdefault("fail_reasons", []).append("fps_soft_warn")
        enriched.append(case)

    profile_aggs = {}
    for profile, profile_cases in by_profile.items():
        agg = aggregate_profile(profile_cases) or {"profile": profile}
        if profile != args.baseline_profile:
            multiseed = evaluate_multiseed(profile_cases, baseline_by_seed, args)
            agg["multiseed_gate"] = multiseed
            deltas = [
                case["delta_vs_rgb_ssim"]
                for case in profile_cases
                if case.get("delta_vs_rgb_ssim") is not None
            ]
            if deltas:
                agg["lpips_improve_ratio_mean"] = statistics.mean(d["lpips_improve_ratio"] for d in deltas)
                agg["psnr_drop_mean"] = statistics.mean(d["psnr_drop"] for d in deltas)
                agg["ssim_drop_mean"] = statistics.mean(d["ssim_drop"] for d in deltas)
            prescreens = [case.get("prescreen", {}).get("passed") for case in profile_cases if case.get("prescreen")]
            agg["prescreen_pass"] = bool(prescreens) and all(prescreens)
        profile_aggs[profile] = agg

    summary = {
        "pass_criteria": {
            "baseline_profile": args.baseline_profile,
            "gate_lpips_improve_ratio": args.gate_lpips_improve,
            "gate_psnr_drop_db": args.gate_psnr_drop,
            "gate_psnr_drop_worst_db": args.gate_psnr_drop_worst,
            "gate_ssim_drop": args.gate_ssim_drop,
            "gate_ssim_drop_worst": args.gate_ssim_drop_worst,
            "max_points": args.max_points,
            "min_fps_speedup_vs_unlimited": args.min_fps_speedup,
            "prescreen_lpips_improve_ratio": args.prescreen_lpips_improve,
            "prescreen_psnr_drop_db": args.prescreen_psnr_drop,
            "prescreen_ssim_drop": args.prescreen_ssim_drop,
            "e3_reference": e3_ref,
            "note": "E4 gates are relative to rgb_ssim baseline per seed, not unlimited.",
        },
        "quality_passing_cases": [c["case"] for c in enriched if c.get("quality_pass")],
        "passing_cases": [c["case"] for c in enriched if c.get("pass")],
        "recommended_top_single_scale": recommend_top_profiles(profile_aggs, multiscale=False),
        "recommended_top_multiscale": recommend_top_profiles(profile_aggs, multiscale=True),
        "recommended_multiseed_set": [
            args.baseline_profile,
            "rgb_l1",
            *recommend_top_profiles(profile_aggs, multiscale=False, limit=1),
            *recommend_top_profiles(profile_aggs, multiscale=True, limit=1),
        ],
        "profile_aggregates": profile_aggs,
        "cases": enriched,
    }

    json_out = Path(args.json_out)
    csv_out = Path(args.csv_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_out, enriched)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {json_out} and {csv_out}")


if __name__ == "__main__":
    main()
