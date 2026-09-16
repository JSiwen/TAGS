#!/usr/bin/env python3
"""Benchmark TAGS render FPS on the eval camera split."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401

    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def render_split(views, gaussians, pipeline, background, use_trained_exp):
    for view in views:
        render(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=use_trained_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
        )


def measure_fps(views, gaussians, pipeline, background, warmup, repeats, use_trained_exp):
    if not views:
        raise RuntimeError("No cameras available for FPS measurement.")

    n_views = len(views)
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            render_split(views, gaussians, pipeline, background, use_trained_exp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        trial_ms = []
        trial_fps = []
        frames = 0
        total_elapsed = 0.0
        for _ in range(max(1, repeats)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            render_split(views, gaussians, pipeline, background, use_trained_exp)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - start, 1e-8)
            total_elapsed += elapsed
            frames += n_views
            trial_ms.append(1000.0 * elapsed / n_views)
            trial_fps.append(n_views / elapsed)

    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    # statistics.fmean is 3.8+; gaussian_splatting env is 3.7.
    mean_ms = statistics.mean(trial_ms)
    return {
        "frames": frames,
        "elapsed_sec": total_elapsed,
        "fps": frames / max(total_elapsed, 1e-8),
        "fps_mean": statistics.mean(trial_fps),
        "fps_std": statistics.pstdev(trial_fps) if len(trial_fps) > 1 else 0.0,
        "ms_per_frame": mean_ms,
        "ms_per_frame_mean": mean_ms,
        "ms_per_frame_std": statistics.pstdev(trial_ms) if len(trial_ms) > 1 else 0.0,
        "ms_per_frame_p50": percentile(trial_ms, 0.50),
        "ms_per_frame_p95": percentile(trial_ms, 0.95),
        "trial_ms_per_frame": trial_ms,
        "trial_fps": trial_fps,
        "peak_memory_mb": peak_mem_mb,
        "num_cameras": n_views,
        "repeats": repeats,
        "warmup": warmup,
    }


def pipeline_snapshot(pipe):
    keys = (
        "convert_SHs_python",
        "compute_cov3D_python",
        "debug",
        "antialiasing",
    )
    return {key: getattr(pipe, key, None) for key in keys}


def main():
    parser = argparse.ArgumentParser(description="Benchmark render FPS for a trained TAGS model.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--repeats", default=30, type=int)
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json-out", default="")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor(
            [1, 1, 1] if dataset.white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        views = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
        stats = measure_fps(
            views,
            gaussians,
            pipe,
            bg,
            warmup=args.warmup,
            repeats=args.repeats,
            use_trained_exp=dataset.train_test_exp,
        )

    first = views[0] if views else None
    resolution = None
    if first is not None:
        image = getattr(first, "original_image", None)
        if image is not None:
            resolution = [int(image.shape[-1]), int(image.shape[-2])]

    payload = {
        "model_path": dataset.model_path,
        "iteration": int(scene.loaded_iter),
        "split": args.split,
        "num_points": int(gaussians.get_xyz.shape[0]),
        "metric": "test_split_throughput",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": getattr(torch.version, "cuda", None),
        "resolution_xy": resolution,
        "pipeline": pipeline_snapshot(pipe),
        **stats,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
