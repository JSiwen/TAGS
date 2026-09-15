#!/usr/bin/env python3
"""Dump per-view NVS metrics for a trained TAGS model."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch import lpips
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401

    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def evaluate_views(views, gaussians, pipeline, background, use_trained_exp):
    rows = []
    with torch.no_grad():
        for view in views:
            image = torch.clamp(
                render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    use_trained_exp=use_trained_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                )["render"],
                0.0,
                1.0,
            )
            gt = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            if use_trained_exp:
                image = image[..., image.shape[-1] // 2 :]
                gt = gt[..., gt.shape[-1] // 2 :]
            rows.append(
                {
                    "image_name": view.image_name,
                    "l1": float(l1_loss(image, gt).mean().item()),
                    "psnr": float(psnr(image, gt).mean().item()),
                    "ssim": float(ssim(image.unsqueeze(0), gt.unsqueeze(0)).item()),
                    "lpips": float(lpips(image.unsqueeze(0), gt.unsqueeze(0), net_type="vgg").item()),
                }
            )
    return rows


def summarize(rows):
    if not rows:
        return {}
    metrics = ("l1", "psnr", "ssim", "lpips")
    out = {"num_cameras": len(rows)}
    for metric in metrics:
        values = [row[metric] for row in rows]
        ranked = sorted(range(len(values)), key=lambda i: values[i])
        worse_first = ranked if metric in ("l1", "lpips") else list(reversed(ranked))
        out[metric] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "worst_views": [rows[i]["image_name"] for i in worse_first[:5]],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-view NVS metrics.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json-out", default="")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    views = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
    rows = evaluate_views(views, gaussians, pipe, bg, dataset.train_test_exp)
    payload = {
        "model_path": dataset.model_path,
        "iteration": int(scene.loaded_iter),
        "split": args.split,
        "num_points": int(gaussians.get_xyz.shape[0]),
        "summary": summarize(rows),
        "views": rows,
    }
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    out = Path(args.json_out) if args.json_out else Path(dataset.model_path) / "per_view.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
