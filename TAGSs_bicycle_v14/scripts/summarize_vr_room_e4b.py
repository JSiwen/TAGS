#!/usr/bin/env python3
"""Summarize vr_room E4b geo-overlap runs against the frozen E4 winner."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_vr_room_e4 as e4sum

FREEZE_PROFILE = "decay20k"
RGB_PROFILE = "rgb_ssim"
E4_WINNER_PROFILE = "lpips_ms_w005"
GEO_MARKERS = (21000, 24000, 27000, 30000)


def load_profile_meta(case_dir):
    return e4sum.load_json(case_dir / "e4b_profile.json") or e4sum.load_json(case_dir / "e4_profile.json") or {}


def remap_case_profile(case, case_dir):
    meta = load_profile_meta(case_dir)
    stored = case.get("profile") or meta.get("profile")
    name = case_dir.name
    if name.startswith("vr_room_e4b_"):
        case["profile"] = stored or name.replace("vr_room_e4b_", "").split("_seed")[0]
        case["source"] = "e4b"
    elif name.startswith("vr_room_e4_lpips_ms_w005") or stored == E4_WINNER_PROFILE:
        case["profile"] = FREEZE_PROFILE
        case["e4_source_profile"] = E4_WINNER_PROFILE
        case["source"] = "e4_freeze"
    elif name.startswith("vr_room_e4_rgb_ssim") or stored == RGB_PROFILE:
        case["profile"] = RGB_PROFILE
        case["source"] = "e4_rgb"
    if case.get("seed") is None:
        case["seed"] = meta.get("seed", 0)
    case["geo"] = meta.get("geo")
    return case


def nearest_stat(rows, target, key):
    best = None
    best_dist = None
    for row in rows:
        iteration = int(row.get("iteration", 0))
        stats = row.get("stats") or row.get("operation_losses") or row.get("train_snapshot") or {}
        if key not in stats:
            continue
        dist = abs(iteration - target)
        if best_dist is None or dist < best_dist:
            best = {"iteration": iteration, "value": float(stats[key])}
            best_dist = dist
    return best


def extract_geo_overlap(case_dir):
    rows = e4sum.read_jsonl(case_dir / "live_logs" / "train_iter_metrics.jsonl")
    if not rows:
        rows = e4sum.read_jsonl(case_dir / "live_logs" / "stage_snapshots.jsonl")
        rows = [
            {"iteration": row.get("iteration"), "stats": row.get("train_snapshot") or {}}
            for row in rows
        ]
    overlap_depth = []
    overlap_normal = []
    for row in rows:
        iteration = int(row.get("iteration", 0))
        if iteration < 21000 or iteration > 27000:
            continue
        stats = row.get("stats") or row.get("operation_losses") or row.get("train_snapshot") or {}
        if "geo/depth_weight" in stats:
            overlap_depth.append(float(stats["geo/depth_weight"]))
        if "geo/normal_weight" in stats:
            overlap_normal.append(float(stats["geo/normal_weight"]))
    markers = {}
    for target in GEO_MARKERS:
        markers[str(target)] = {
            "depth_weight": nearest_stat(rows, target, "geo/depth_weight"),
            "normal_weight": nearest_stat(rows, target, "geo/normal_weight"),
        }
    out = {"markers": markers}
    if overlap_depth:
        out["overlap_21k_27k_depth_weight_mean"] = statistics.mean(overlap_depth)
    if overlap_normal:
        out["overlap_21k_27k_normal_weight_mean"] = statistics.mean(overlap_normal)
    return out


def per_view_improve_counts(case_payload, baseline_payload):
    if not case_payload or not baseline_payload:
        return None
    case_views = {row["image_name"]: row for row in case_payload.get("views") or []}
    base_views = {row["image_name"]: row for row in baseline_payload.get("views") or []}
    names = sorted(set(case_views) & set(base_views))
    if not names:
        return None
    counts = {
        "num_views": len(names),
        "lpips_improved": 0,
        "psnr_improved": 0,
        "ssim_improved": 0,
    }
    for name in names:
        case_row = case_views[name]
        base_row = base_views[name]
        if case_row["lpips"] < base_row["lpips"]:
            counts["lpips_improved"] += 1
        if case_row["psnr"] > base_row["psnr"]:
            counts["psnr_improved"] += 1
        if case_row["ssim"] > base_row["ssim"]:
            counts["ssim_improved"] += 1
    return counts


def depth_delta(case_summary, baseline_summary):
    if not case_summary or not baseline_summary:
        return None
    out = {}
    for metric in ("pearson", "spearman", "log_l1"):
        case_mean = ((case_summary.get(metric) or {}).get("mean"))
        base_mean = ((baseline_summary.get(metric) or {}).get("mean"))
        if case_mean is None or base_mean is None:
            continue
        out[f"{metric}_drop"] = base_mean - case_mean
        if abs(base_mean) > 1e-12:
            if metric == "log_l1":
                out[f"{metric}_improve_ratio"] = (base_mean - case_mean) / base_mean
            else:
                out[f"{metric}_improve_ratio"] = (case_mean - base_mean) / abs(base_mean)
    return out


def compare_bundle(case, freeze, rgb):
    bundle = {
        "vs_freeze": e4sum.compare_to_baseline(case, freeze) if freeze else None,
        "vs_rgb_ssim": e4sum.compare_to_baseline(case, rgb) if rgb else None,
        "depth_vs_freeze": depth_delta(case.get("depth_prior"), (freeze or {}).get("depth_prior")),
        "per_view_vs_rgb_ssim": case.get("per_view_vs_rgb_ssim"),
        "per_view_vs_freeze": case.get("per_view_vs_freeze"),
    }
    return bundle


def evaluate_prescreen(bundle, args):
    vs_freeze = bundle.get("vs_freeze") or {}
    vs_rgb = bundle.get("vs_rgb_ssim") or {}
    depth = bundle.get("depth_vs_freeze") or {}
    per_view = bundle.get("per_view_vs_rgb_ssim") or {}
    checks = {
        "lpips_vs_rgb_ge_12pct": vs_rgb.get("lpips_improve_ratio", -1) >= args.prescreen_lpips_vs_rgb,
        "psnr_drop_vs_freeze_le": vs_freeze.get("psnr_drop", 999) <= args.prescreen_psnr_drop,
        "ssim_drop_vs_freeze_le": vs_freeze.get("ssim_drop", 999) <= args.prescreen_ssim_drop,
        "depth_log_l1_not_worse": depth.get("log_l1_improve_ratio", 0) >= 0 if depth else True,
    }
    if per_view:
        checks["ssim_improved_views_ge_min"] = per_view.get("ssim_improved", 0) >= args.prescreen_ssim_improved_views
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_final(case, bundle, args, unlimited_fps=None):
    reasons = []
    vs_freeze = bundle.get("vs_freeze")
    vs_rgb = bundle.get("vs_rgb_ssim")
    depth = bundle.get("depth_vs_freeze") or {}
    if vs_freeze is None or vs_rgb is None:
        return {
            "pass": False,
            "quality_pass": False,
            "fail_reasons": ["missing_baseline_pair"],
        }

    points_ok = case["num_points"] <= args.max_points
    lpips_ok = vs_rgb["lpips_improve_ratio"] >= args.gate_lpips_vs_rgb and vs_rgb["lpips_improve_abs"] >= 0
    lpips_vs_freeze_ok = vs_freeze["lpips_improve_abs"] >= -abs(args.gate_lpips_vs_freeze_abs_slack)
    psnr_ok = vs_freeze["psnr_drop"] <= args.gate_psnr_drop
    ssim_ok = vs_freeze["ssim_drop"] <= args.gate_ssim_drop
    depth_l1_ok = True if not depth else depth.get("log_l1_improve_ratio", -1) >= args.gate_depth_log_l1_improve
    spearman_ok = True if not depth else depth.get("spearman_improve_ratio", -1) >= 0
    if not points_ok:
        reasons.append(f"points>{args.max_points}")
    if not lpips_ok:
        reasons.append("lpips_vs_rgb<12pct_or_worse")
    if not lpips_vs_freeze_ok:
        reasons.append("lpips_vs_freeze_regressed")
    if not psnr_ok:
        reasons.append(f"psnr_drop_vs_freeze>{args.gate_psnr_drop}")
    if not ssim_ok:
        reasons.append(f"ssim_drop_vs_freeze>{args.gate_ssim_drop}")
    if depth and not depth_l1_ok:
        reasons.append("depth_log_l1_improve<3pct")
    if depth and not spearman_ok:
        reasons.append("spearman_worse_vs_freeze")

    fps = case.get("fps")
    fps_speedup = None
    fps_ok = None
    if fps is not None and unlimited_fps and unlimited_fps > 0:
        fps_speedup = fps / unlimited_fps
        fps_ok = fps_speedup >= args.min_fps_speedup
        if fps_ok is False:
            reasons.append(f"fps_speedup<{args.min_fps_speedup}")
    elif fps is None:
        reasons.append("fps_pending")

    quality_pass = (
        points_ok and lpips_ok and lpips_vs_freeze_ok and psnr_ok and ssim_ok and depth_l1_ok and spearman_ok
    )
    return {
        "pass": quality_pass and bool(fps_ok),
        "quality_pass": quality_pass,
        "fps_status": "pending" if fps_ok is None else ("pass" if fps_ok else "fail"),
        "fps_speedup_vs_unlimited": fps_speedup,
        "fail_reasons": reasons,
    }


def evaluate_multiseed(profile_cases, freeze_by_seed, rgb_by_seed, args, unlimited_fps=None):
    complete = [case for case in profile_cases if case.get("psnr") is not None]
    if len(complete) < 3:
        return {"passed": False, "reason": "incomplete_seeds", "quality_passed": False, "fps_passed": False}
    freeze_deltas = []
    rgb_deltas = []
    depth_deltas = []
    for case in complete:
        seed = int(case.get("seed", 0))
        freeze = freeze_by_seed.get(seed)
        rgb = rgb_by_seed.get(seed)
        if freeze is None or rgb is None:
            return {
                "passed": False,
                "reason": f"missing_baseline_seed_{seed}",
                "quality_passed": False,
                "fps_passed": False,
            }
        freeze_deltas.append(e4sum.compare_to_baseline(case, freeze))
        rgb_deltas.append(e4sum.compare_to_baseline(case, rgb))
        depth = depth_delta(case.get("depth_prior"), freeze.get("depth_prior"))
        if depth:
            depth_deltas.append(depth)

    quality_checks = {
        "all_seeds_lpips_vs_rgb_not_worse": all(delta["lpips_improve_abs"] >= 0 for delta in rgb_deltas),
        "mean_lpips_vs_rgb_ge_12pct": statistics.mean(d["lpips_improve_ratio"] for d in rgb_deltas) >= args.gate_lpips_vs_rgb,
        "mean_psnr_drop_vs_freeze_le": statistics.mean(d["psnr_drop"] for d in freeze_deltas) <= args.gate_psnr_drop,
        "worst_psnr_drop_vs_freeze_le": max(d["psnr_drop"] for d in freeze_deltas) <= args.gate_psnr_drop_worst,
        "mean_ssim_drop_vs_freeze_le": statistics.mean(d["ssim_drop"] for d in freeze_deltas) <= args.gate_ssim_drop,
        "worst_ssim_drop_vs_freeze_le": max(d["ssim_drop"] for d in freeze_deltas) <= args.gate_ssim_drop_worst,
        "all_points_le_2m": all(case["num_points"] <= args.max_points for case in complete),
    }
    if depth_deltas:
        quality_checks["mean_depth_log_l1_improve_ge_3pct"] = (
            statistics.mean(d.get("log_l1_improve_ratio", -1) for d in depth_deltas) >= args.gate_depth_log_l1_improve
        )
        quality_checks["mean_spearman_not_worse"] = (
            statistics.mean(d.get("spearman_improve_ratio", -1) for d in depth_deltas) >= 0
        )
    fps_values = [case.get("fps") for case in complete]
    if unlimited_fps and unlimited_fps > 0:
        fps_checks = {
            "all_seeds_fps_present": all(fps is not None for fps in fps_values),
            "all_seeds_fps_speedup_ge_min": all(
                fps / unlimited_fps >= args.min_fps_speedup for fps in fps_values if fps is not None
            )
            if any(fps is not None for fps in fps_values)
            else False,
        }
        fps_speedups = [fps / unlimited_fps for fps in fps_values if fps is not None]
    else:
        fps_checks = {"all_seeds_fps_present": False, "all_seeds_fps_speedup_ge_min": False}
        fps_speedups = []
    quality_passed = all(quality_checks.values())
    fps_passed = all(fps_checks.values()) if unlimited_fps else False
    return {
        "passed": quality_passed and fps_passed,
        "quality_passed": quality_passed,
        "fps_passed": fps_passed,
        "checks": {**quality_checks, **fps_checks},
        "quality_checks": quality_checks,
        "fps_checks": fps_checks,
        "lpips_improve_vs_rgb_mean": statistics.mean(d["lpips_improve_ratio"] for d in rgb_deltas),
        "psnr_drop_vs_freeze_mean": statistics.mean(d["psnr_drop"] for d in freeze_deltas),
        "ssim_drop_vs_freeze_mean": statistics.mean(d["ssim_drop"] for d in freeze_deltas),
        "depth_log_l1_improve_mean": (
            statistics.mean(d["log_l1_improve_ratio"] for d in depth_deltas) if depth_deltas else None
        ),
        "fps_speedup_mean": statistics.mean(fps_speedups) if fps_speedups else None,
    }


def recommend_top_profiles(profile_aggs, limit=1):
    ranked = []
    for profile, agg in profile_aggs.items():
        if profile in (FREEZE_PROFILE, RGB_PROFILE):
            continue
        if not agg.get("prescreen_pass"):
            continue
        ranked.append(
            (
                profile,
                agg.get("ssim_drop_vs_freeze_mean", 999),
                -(agg.get("depth_log_l1_improve_mean") or -999),
                -(agg.get("lpips_improve_vs_rgb_mean") or -999),
            )
        )
    ranked.sort(key=lambda item: (item[1], item[2], item[3]))
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
        "psnr_drop_vs_freeze",
        "ssim_drop_vs_freeze",
        "lpips_improve_ratio_vs_rgb_ssim",
        "depth_log_l1_improve_vs_freeze",
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
            vs_freeze = case.get("delta_vs_freeze") or {}
            vs_rgb = case.get("delta_vs_rgb_ssim") or {}
            depth = case.get("depth_vs_freeze") or {}
            row["psnr_drop_vs_freeze"] = vs_freeze.get("psnr_drop")
            row["ssim_drop_vs_freeze"] = vs_freeze.get("ssim_drop")
            row["lpips_improve_ratio_vs_rgb_ssim"] = vs_rgb.get("lpips_improve_ratio")
            row["depth_log_l1_improve_vs_freeze"] = depth.get("log_l1_improve_ratio")
            row["fail_reasons"] = ";".join(case.get("fail_reasons") or [])
            writer.writerow(row)


def index_by_seed(cases):
    out = {}
    for case in cases:
        if case.get("psnr") is None:
            continue
        out[int(case.get("seed", 0))] = case
    return out


def summarize_one(case_dir):
    case = e4sum.summarize_case(case_dir)
    case = remap_case_profile(case, case_dir)
    per_view = e4sum.load_json(case_dir / "per_view.json")
    if per_view:
        case["per_view_summary"] = per_view.get("summary")
        case["per_view_views"] = per_view.get("views")
    case["geo_overlap"] = extract_geo_overlap(case_dir)
    return case


def collect_cases(args):
    root = Path(args.output_root)
    e4b_dirs = e4sum.collect_case_dirs(root, args.case_glob)
    cases = [summarize_one(path) for path in e4b_dirs]
    has_freeze = any(case.get("profile") == FREEZE_PROFILE and case.get("psnr") is not None for case in cases)
    if not has_freeze:
        for path in e4sum.collect_case_dirs(root, args.freeze_glob):
            cases.append(summarize_one(path))
    existing_rgb = {int(case.get("seed", 0)) for case in cases if case.get("profile") == RGB_PROFILE}
    for path in e4sum.collect_case_dirs(root, args.rgb_glob):
        rgb_case = summarize_one(path)
        seed = int(rgb_case.get("seed", 0))
        if seed not in existing_rgb:
            cases.append(rgb_case)
            existing_rgb.add(seed)
    return cases


def attach_per_view_counts(cases, freeze_by_seed, rgb_by_seed):
    for case in cases:
        seed = int(case.get("seed", 0))
        case_payload = {"views": case.get("per_view_views") or []}
        freeze = freeze_by_seed.get(seed) or {}
        rgb = rgb_by_seed.get(seed) or {}
        case["per_view_vs_freeze"] = per_view_improve_counts(
            case_payload, {"views": freeze.get("per_view_views") or []}
        )
        case["per_view_vs_rgb_ssim"] = per_view_improve_counts(
            case_payload, {"views": rgb.get("per_view_views") or []}
        )
        case.pop("per_view_views", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--case-glob", default="vr_room_e4b_*")
    parser.add_argument("--freeze-glob", default="vr_room_e4_lpips_ms_w005*")
    parser.add_argument("--rgb-glob", default="vr_room_e4_rgb_ssim*")
    parser.add_argument("--unlimited-case", default="output/vr_room_indoor_sensor_depth")
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--min-fps-speedup", type=float, default=1.5)
    parser.add_argument("--gate-lpips-vs-rgb", type=float, default=0.12)
    parser.add_argument("--gate-lpips-vs-freeze-abs-slack", type=float, default=0.002)
    parser.add_argument("--gate-psnr-drop", type=float, default=0.05)
    parser.add_argument("--gate-psnr-drop-worst", type=float, default=0.08)
    parser.add_argument("--gate-ssim-drop", type=float, default=0.0002)
    parser.add_argument("--gate-ssim-drop-worst", type=float, default=0.001)
    parser.add_argument("--gate-depth-log-l1-improve", type=float, default=0.03)
    parser.add_argument("--prescreen-lpips-vs-rgb", type=float, default=0.12)
    parser.add_argument("--prescreen-psnr-drop", type=float, default=0.08)
    parser.add_argument("--prescreen-ssim-drop", type=float, default=0.001)
    parser.add_argument("--prescreen-ssim-improved-views", type=int, default=18)
    parser.add_argument("--measure-fps", type=int, default=0)
    parser.add_argument("--fps-warmup", type=int, default=10)
    parser.add_argument("--fps-repeats", type=int, default=30)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--json-out", default="output/vr_room_e4b_summary.json")
    parser.add_argument("--csv-out", default="output/vr_room_e4b_summary.csv")
    args = parser.parse_args()

    unlimited_fps = e4sum.resolve_unlimited_fps(args)
    cases = collect_cases(args)
    if args.measure_fps:
        for case in cases:
            case_dir = Path(args.output_root) / case["case"]
            if not case_dir.is_dir():
                continue
            try:
                e4sum.measure_fps(case_dir, args.python_bin, args.fps_warmup, args.fps_repeats)
                fps_payload = e4sum.load_json(case_dir / "fps_benchmark.json")
                if fps_payload and fps_payload.get("fps") is not None:
                    case["fps"] = float(fps_payload["fps"])
            except subprocess.CalledProcessError as exc:
                print(f"[warn] FPS failed for {case_dir.name}: {exc}", file=sys.stderr)

    by_profile = {}
    for case in cases:
        by_profile.setdefault(case.get("profile") or case.get("case"), []).append(case)
    freeze_by_seed = index_by_seed(by_profile.get(FREEZE_PROFILE, []))
    rgb_by_seed = index_by_seed(by_profile.get(RGB_PROFILE, []))
    attach_per_view_counts(cases, freeze_by_seed, rgb_by_seed)

    enriched = []
    for case in cases:
        seed = int(case.get("seed", 0))
        bundle = compare_bundle(case, freeze_by_seed.get(seed), rgb_by_seed.get(seed))
        if bundle["vs_freeze"] is not None:
            case["delta_vs_freeze"] = bundle["vs_freeze"]
        if bundle["vs_rgb_ssim"] is not None:
            case["delta_vs_rgb_ssim"] = bundle["vs_rgb_ssim"]
        if bundle["depth_vs_freeze"] is not None:
            case["depth_vs_freeze"] = bundle["depth_vs_freeze"]
        if case.get("profile") not in (FREEZE_PROFILE, RGB_PROFILE):
            case["prescreen"] = evaluate_prescreen(bundle, args)
            gate = evaluate_final(case, bundle, args, unlimited_fps=unlimited_fps)
            case.update(
                {
                    k: gate[k]
                    for k in ("pass", "quality_pass", "fps_status", "fps_speedup_vs_unlimited", "fail_reasons")
                    if k in gate
                }
            )
        enriched.append(case)

    profile_aggs = {}
    for profile, profile_cases in by_profile.items():
        agg = e4sum.aggregate_profile(profile_cases) or {"profile": profile}
        if profile not in (FREEZE_PROFILE, RGB_PROFILE):
            freeze_deltas = [case["delta_vs_freeze"] for case in profile_cases if case.get("delta_vs_freeze")]
            rgb_deltas = [case["delta_vs_rgb_ssim"] for case in profile_cases if case.get("delta_vs_rgb_ssim")]
            depth_deltas = [case["depth_vs_freeze"] for case in profile_cases if case.get("depth_vs_freeze")]
            if freeze_deltas:
                agg["psnr_drop_vs_freeze_mean"] = statistics.mean(d["psnr_drop"] for d in freeze_deltas)
                agg["ssim_drop_vs_freeze_mean"] = statistics.mean(d["ssim_drop"] for d in freeze_deltas)
            if rgb_deltas:
                agg["lpips_improve_vs_rgb_mean"] = statistics.mean(d["lpips_improve_ratio"] for d in rgb_deltas)
            if depth_deltas:
                agg["depth_log_l1_improve_mean"] = statistics.mean(
                    d.get("log_l1_improve_ratio", 0) for d in depth_deltas
                )
            prescreens = [case.get("prescreen", {}).get("passed") for case in profile_cases if case.get("prescreen")]
            agg["prescreen_pass"] = bool(prescreens) and all(prescreens)
            agg["multiseed_gate"] = evaluate_multiseed(
                profile_cases, freeze_by_seed, rgb_by_seed, args, unlimited_fps=unlimited_fps
            )
        profile_aggs[profile] = agg

    recommended = recommend_top_profiles(profile_aggs)
    summary = {
        "pass_criteria": {
            "freeze_profile": FREEZE_PROFILE,
            "rgb_profile": RGB_PROFILE,
            "gate_lpips_vs_rgb": args.gate_lpips_vs_rgb,
            "gate_psnr_drop_vs_freeze_db": args.gate_psnr_drop,
            "gate_psnr_drop_worst_db": args.gate_psnr_drop_worst,
            "gate_ssim_drop_vs_freeze": args.gate_ssim_drop,
            "gate_ssim_drop_worst": args.gate_ssim_drop_worst,
            "gate_depth_log_l1_improve": args.gate_depth_log_l1_improve,
            "max_points": args.max_points,
            "min_fps_speedup_vs_unlimited": args.min_fps_speedup,
            "unlimited_fps": unlimited_fps,
            "note": (
                "LPIPS is still scored against E4 rgb_ssim. "
                "PSNR/SSIM/depth gates are relative to the frozen E4 winner (decay20k / lpips_ms_w005). "
                "FPS gates are relative to unlimited_fps."
            ),
        },
        "quality_passing_profiles": [
            profile
            for profile, agg in profile_aggs.items()
            if profile not in (FREEZE_PROFILE, RGB_PROFILE)
            and agg.get("multiseed_gate", {}).get("quality_passed")
        ],
        "quality_passing_cases": [c["case"] for c in enriched if c.get("quality_pass")],
        "passing_cases": [c["case"] for c in enriched if c.get("pass")],
        "recommended_top": recommended,
        "recommended_multiseed_set": recommended,
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
