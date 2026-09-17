#!/usr/bin/env python3
"""Evaluate per-view depth-prior consistency for a trained TAGS model."""

import argparse
import json
import math
import sys
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


def masked_mean(values, mask):
    valid = mask > 0
    if int(valid.sum().item()) < 2:
        return torch.tensor(float("nan"), device=values.device)
    return values[valid].mean()


def pearson_corr_1d(a, b):
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum() * (b * b).sum()).clamp_min(1e-12)
    return float((a * b).sum().div(denom).item())


def pearson_corr(pred, gt, mask):
    valid = mask > 0
    if int(valid.sum().item()) < 2:
        return float("nan")
    return pearson_corr_1d(pred[valid].reshape(-1), gt[valid].reshape(-1))


def spearman_corr(pred, gt, mask):
    valid = mask > 0
    if int(valid.sum().item()) < 2:
        return float("nan")
    p = pred[valid].reshape(-1)
    g = gt[valid].reshape(-1)
    p_rank = p.argsort().argsort().float()
    g_rank = g.argsort().argsort().float()
    return pearson_corr_1d(p_rank, g_rank)


def edge_aware_log_l1(pred, gt, rgb, mask):
    gray = rgb.mean(dim=0, keepdim=True)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[..., :, 1:-1] = 0.5 * torch.abs(gray[..., :, 2:] - gray[..., :, :-2])
    gy[..., 1:-1, :] = 0.5 * torch.abs(gray[..., 2:, :] - gray[..., :-2, :])
    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    threshold = torch.quantile(grad_mag.reshape(-1), 0.95)
    edge_weight = torch.exp(-grad_mag / (threshold + 1e-6))
    diff = torch.abs(torch.log(pred.clamp_min(1e-6)) - torch.log(gt.clamp_min(1e-6)))
    weighted = diff * edge_weight * mask
    denom = (edge_weight * mask).sum().clamp_min(1.0)
    return float((weighted.sum() / denom).item())


def evaluate_views(views, gaussians, pipeline, background, use_trained_exp):
    rows = []
    with torch.no_grad():
        for view in views:
            if not getattr(view, "depth_reliable", False):
                continue
            if getattr(view, "invdepthmap", None) is None:
                continue
            render_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                use_trained_exp=use_trained_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )
            pred = render_pkg["depth"][0]
            gt = view.invdepthmap.cuda()[0]
            rgb = view.original_image.cuda()
            base_mask = view.depth_mask.cuda()[0] if getattr(view, "depth_mask", None) is not None else torch.ones_like(gt)
            valid = base_mask > 0
            valid = valid & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0)
            mask = valid.float()
            coverage = float(valid.float().mean().item())
            if coverage <= 0:
                continue
            rows.append(
                {
                    "image_name": view.image_name,
                    "coverage": coverage,
                    "pearson": pearson_corr(pred, gt, mask),
                    "spearman": spearman_corr(pred, gt, mask),
                    "log_l1": edge_aware_log_l1(pred, gt, rgb, mask),
                }
            )
    return rows


def summarize(rows):
    if not rows:
        return {}
    metrics = ("coverage", "pearson", "spearman", "log_l1")
    out = {"num_cameras": len(rows)}
    for metric in metrics:
        values = [row[metric] for row in rows if not math.isnan(row[metric])]
        if not values:
            continue
        ranked = sorted(range(len(values)), key=lambda i: values[i])
        worse_first = ranked if metric in ("log_l1",) else list(reversed(ranked))
        out[metric] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "worst_views": [rows[i]["image_name"] for i in worse_first[:5]],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-view depth-prior consistency.")
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
        "note": "Metrics measure consistency with monocular inverse-depth priors, not absolute depth accuracy.",
    }
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    out = Path(args.json_out) if args.json_out else Path(dataset.model_path) / "depth_per_view.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
