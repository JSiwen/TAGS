#!/usr/bin/env python3
"""Benchmark TAGS render FPS on the eval camera split."""

import argparse
import json
import time
from pathlib import Path

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


def measure_fps(views, gaussians, pipeline, background, warmup, repeats, use_trained_exp):
    if not views:
        raise RuntimeError("No cameras available for FPS measurement.")

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            for view in views:
                render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    use_trained_exp=use_trained_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        frames = 0
        for _ in range(max(1, repeats)):
            for view in views:
                render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    use_trained_exp=use_trained_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                )
                frames += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - start, 1e-8)

    return {
        "frames": frames,
        "elapsed_sec": elapsed,
        "fps": frames / elapsed,
        "ms_per_frame": 1000.0 * elapsed / frames,
        "num_cameras": len(views),
        "repeats": repeats,
        "warmup": warmup,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark render FPS for a trained TAGS model.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--repeats", default=5, type=int)
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

    payload = {
        "model_path": dataset.model_path,
        "iteration": int(scene.loaded_iter),
        "split": args.split,
        "num_points": int(gaussians.get_xyz.shape[0]),
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
