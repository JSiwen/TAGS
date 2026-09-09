#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None


DEFAULT_CHRONIC_VIEWS = ("00008.png", "00011.png", "00016.png", "00018.png", "00021.png")
DEFAULT_HARD_VIEWS = ("00004.png", "00002.png", "00003.png", "00001.png")


def read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def method_to_iter(method_name: str):
    if not method_name.startswith("ours_"):
        return None
    try:
        return int(method_name.split("_", 1)[1])
    except Exception:
        return None


def chronic_stats(case_per_view, baseline_per_view, chronic_views):
    max_psnr_drop = 0.0
    lpips_deltas = []
    for view_name in chronic_views:
        psnr_case = case_per_view.get("PSNR", {}).get(view_name)
        psnr_base = baseline_per_view.get("PSNR", {}).get(view_name)
        lpips_case = case_per_view.get("LPIPS", {}).get(view_name)
        lpips_base = baseline_per_view.get("LPIPS", {}).get(view_name)
        if psnr_case is None or psnr_base is None or lpips_case is None or lpips_base is None:
            continue
        psnr_drop = max(0.0, float(psnr_base) - float(psnr_case))
        max_psnr_drop = max(max_psnr_drop, psnr_drop)
        lpips_deltas.append(float(lpips_case) - float(lpips_base))
    avg_lpips_delta = sum(lpips_deltas) / len(lpips_deltas) if lpips_deltas else 0.0
    return {
        "max_psnr_drop": max_psnr_drop,
        "avg_lpips_delta": avg_lpips_delta,
    }


def regressed_count(case_per_view, baseline_per_view, metric):
    case_values = case_per_view.get(metric, {})
    base_values = baseline_per_view.get(metric, {})
    count = 0
    for image_name, value in case_values.items():
        if image_name not in base_values:
            continue
        delta = float(value) - float(base_values[image_name])
        if metric == "LPIPS":
            if delta > 0:
                count += 1
        elif delta < 0:
            count += 1
    return count


def write_tensorboard(
    case_dir: Path,
    gate_reference_dir: Path,
    stage_methods,
    final_method: str,
    chronic_views,
    hard_views,
    gate_psnr_drop: float,
    gate_ssim_drop: float,
    gate_lpips_rise: float,
    gate_psnr_regressed_views: int,
    gate_lpips_regressed_views: int,
    gate_chronic_psnr_drop: float,
    gate_chronic_avg_lpips_delta: float,
):
    if SummaryWriter is None:
        raise SystemExit("TensorBoard is unavailable: install tensorboard to enable stage export.")

    results = read_json(case_dir / "results.json") or {}
    per_view = read_json(case_dir / "per_view.json") or {}
    gate_results = read_json(gate_reference_dir / "results.json") or {}
    gate_per_view = read_json(gate_reference_dir / "per_view.json") or {}
    if final_method not in gate_results or final_method not in gate_per_view:
        raise SystemExit(f"Gate reference missing method {final_method}: {gate_reference_dir}")

    gate_metrics = gate_results[final_method]
    writer = SummaryWriter(log_dir=str(case_dir))
    wrote_any = False

    for method in stage_methods:
        if method not in results:
            continue
        step = method_to_iter(method)
        if step is None:
            continue
        metrics = results[method]
        writer.add_scalar("stage/SSIM", float(metrics.get("SSIM", 0.0)), step)
        writer.add_scalar("stage/PSNR", float(metrics.get("PSNR", 0.0)), step)
        writer.add_scalar("stage/LPIPS", float(metrics.get("LPIPS", 0.0)), step)
        writer.add_scalar("stage/delta_vs_gate_ssim", float(metrics.get("SSIM", 0.0)) - float(gate_metrics.get("SSIM", 0.0)), step)
        writer.add_scalar("stage/delta_vs_gate_psnr", float(metrics.get("PSNR", 0.0)) - float(gate_metrics.get("PSNR", 0.0)), step)
        writer.add_scalar("stage/delta_vs_gate_lpips", float(metrics.get("LPIPS", 0.0)) - float(gate_metrics.get("LPIPS", 0.0)), step)
        wrote_any = True

        if method not in per_view or final_method not in gate_per_view:
            continue
        case_per_view = per_view[method]
        baseline_per_view = gate_per_view[final_method]
        psnr_regressed = regressed_count(case_per_view, baseline_per_view, "PSNR")
        lpips_regressed = regressed_count(case_per_view, baseline_per_view, "LPIPS")
        chronic = chronic_stats(case_per_view, baseline_per_view, chronic_views)

        writer.add_scalar("stage/psnr_regressed_views", float(psnr_regressed), step)
        writer.add_scalar("stage/lpips_regressed_views", float(lpips_regressed), step)
        writer.add_scalar("stage/chronic_max_psnr_drop", float(chronic["max_psnr_drop"]), step)
        writer.add_scalar("stage/chronic_avg_lpips_delta", float(chronic["avg_lpips_delta"]), step)

        for view_name in hard_views:
            psnr_case = case_per_view.get("PSNR", {}).get(view_name)
            psnr_base = baseline_per_view.get("PSNR", {}).get(view_name)
            lpips_case = case_per_view.get("LPIPS", {}).get(view_name)
            lpips_base = baseline_per_view.get("LPIPS", {}).get(view_name)
            if psnr_case is not None and psnr_base is not None:
                writer.add_scalar(
                    f"hard_views/{view_name}/psnr_delta_vs_gate",
                    float(psnr_case) - float(psnr_base),
                    step,
                )
            if lpips_case is not None and lpips_base is not None:
                writer.add_scalar(
                    f"hard_views/{view_name}/lpips_delta_vs_gate",
                    float(lpips_case) - float(lpips_base),
                    step,
                )

        gate_checks = {
            "global_psnr": float(metrics.get("PSNR", 0.0)) >= float(gate_metrics.get("PSNR", 0.0)) - gate_psnr_drop,
            "global_ssim": float(metrics.get("SSIM", 0.0)) >= float(gate_metrics.get("SSIM", 0.0)) - gate_ssim_drop,
            "global_lpips": float(metrics.get("LPIPS", 0.0)) <= float(gate_metrics.get("LPIPS", 0.0)) + gate_lpips_rise,
            "psnr_regressed_views": psnr_regressed <= gate_psnr_regressed_views,
            "lpips_regressed_views": lpips_regressed <= gate_lpips_regressed_views,
            "chronic_psnr_drop": chronic["max_psnr_drop"] <= gate_chronic_psnr_drop,
            "chronic_avg_lpips_delta": chronic["avg_lpips_delta"] <= gate_chronic_avg_lpips_delta,
        }
        gate_pass = all(gate_checks.values())
        writer.add_scalar("stage/gate_pass", 1.0 if gate_pass else 0.0, step)
        for check_name, passed in gate_checks.items():
            writer.add_scalar(f"stage/gate_{check_name}", 1.0 if passed else 0.0, step)

    if not wrote_any:
        writer.close()
        raise SystemExit(f"No stage metrics found in {case_dir}/results.json for methods: {stage_methods}")
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Export stage metrics/per-view deltas to TensorBoard.")
    parser.add_argument("--case_dir", required=True, type=Path)
    parser.add_argument("--gate_reference_dir", required=True, type=Path)
    parser.add_argument("--final_method", default="ours_30000")
    parser.add_argument("--stage_methods", nargs="+", required=True)
    parser.add_argument("--chronic_views", nargs="+", default=list(DEFAULT_CHRONIC_VIEWS))
    parser.add_argument("--hard_views", nargs="+", default=list(DEFAULT_HARD_VIEWS))
    parser.add_argument("--gate_psnr_drop", type=float, default=0.01)
    parser.add_argument("--gate_ssim_drop", type=float, default=0.0002)
    parser.add_argument("--gate_lpips_rise", type=float, default=0.0005)
    parser.add_argument("--gate_psnr_regressed_views", type=int, default=10)
    parser.add_argument("--gate_lpips_regressed_views", type=int, default=12)
    parser.add_argument("--gate_chronic_psnr_drop", type=float, default=0.12)
    parser.add_argument("--gate_chronic_avg_lpips_delta", type=float, default=0.0015)
    args = parser.parse_args()

    write_tensorboard(
        case_dir=args.case_dir,
        gate_reference_dir=args.gate_reference_dir,
        stage_methods=args.stage_methods,
        final_method=args.final_method,
        chronic_views=args.chronic_views,
        hard_views=args.hard_views,
        gate_psnr_drop=args.gate_psnr_drop,
        gate_ssim_drop=args.gate_ssim_drop,
        gate_lpips_rise=args.gate_lpips_rise,
        gate_psnr_regressed_views=args.gate_psnr_regressed_views,
        gate_lpips_regressed_views=args.gate_lpips_regressed_views,
        gate_chronic_psnr_drop=args.gate_chronic_psnr_drop,
        gate_chronic_avg_lpips_delta=args.gate_chronic_avg_lpips_delta,
    )


if __name__ == "__main__":
    main()
