#!/usr/bin/env python3
"""Run multiple independent FPS benchmark trials with shuffled model order."""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path):
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8-sig"))


def run_trial(model_dir, python_bin, iteration, warmup, repeats, split, trial_idx):
    out = Path(model_dir) / f"fps_benchmark_trial{trial_idx}.json"
    cmd = [
        python_bin,
        str(ROOT / "scripts" / "benchmark_render_fps.py"),
        "-m",
        str(model_dir),
        "--iteration",
        str(iteration),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--split",
        split,
        "--json-out",
        str(out),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    payload = load_json(out)
    payload["trial"] = trial_idx
    return payload


def aggregate(trials):
    fps_values = [float(t["fps"]) for t in trials]
    ms_values = [float(t["ms_per_frame"]) for t in trials]
    n = len(fps_values)
    mean_fps = sum(fps_values) / n
    mean_ms = sum(ms_values) / n
    var_fps = sum((v - mean_fps) ** 2 for v in fps_values) / max(n - 1, 1)
    var_ms = sum((v - mean_ms) ** 2 for v in ms_values) / max(n - 1, 1)
    sorted_ms = sorted(ms_values)

    def pct(values, q):
        if len(values) == 1:
            return values[0]
        rank = (len(values) - 1) * q
        lo = int(rank)
        hi = min(lo + 1, len(values) - 1)
        if lo == hi:
            return values[lo]
        w = rank - lo
        return values[lo] * (1 - w) + values[hi] * w

    return {
        "num_trials": n,
        "fps_mean": mean_fps,
        "fps_std": var_fps**0.5 if n > 1 else 0.0,
        "fps_min": min(fps_values),
        "fps_max": max(fps_values),
        "ms_per_frame_mean": mean_ms,
        "ms_per_frame_std": var_ms**0.5 if n > 1 else 0.0,
        "ms_per_frame_p50": pct(sorted_ms, 0.50),
        "ms_per_frame_p95": pct(sorted_ms, 0.95),
        "trial_fps": fps_values,
        "trial_ms_per_frame": ms_values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="Comma-separated model dirs")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    model_dirs = [Path(p.strip()) for p in args.models.split(",") if p.strip()]
    if not model_dirs:
        raise SystemExit("No model directories provided.")

    summary = {}
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            print(f"[warn] skip missing {model_dir}", file=sys.stderr)
            continue
        order = list(range(args.trials))
        rng = random.Random(args.seed + hash(model_dir.name) % 10_000)
        rng.shuffle(order)
        trial_payloads = []
        for trial in order:
            print(f"[fps-trials] {model_dir.name} trial {trial + 1}/{args.trials}")
            trial_payloads.append(
                run_trial(
                    model_dir,
                    args.python_bin,
                    args.iteration,
                    args.warmup,
                    args.repeats,
                    args.split,
                    trial + 1,
                )
            )
        agg = aggregate(trial_payloads)
        payload = {
            "model_path": str(model_dir),
            "iteration": args.iteration,
            "split": args.split,
            "warmup": args.warmup,
            "repeats_per_trial": args.repeats,
            "trials": trial_payloads,
            **agg,
        }
        out = model_dir / "fps_benchmark_trials.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # Keep backward-compatible single benchmark file for summarizer.
        canonical = {
            **trial_payloads[-1],
            **agg,
            "metric": "test_split_throughput_trials",
        }
        (model_dir / "fps_benchmark.json").write_text(
            json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary[model_dir.name] = agg
        print(f"Wrote {out}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
