#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


DEFAULT_BOXES = {
    "full": [0.00, 0.00, 1.00, 1.00],
    "subject": [0.12, 0.20, 0.86, 0.90],
    "background": [0.00, 0.00, 1.00, 0.45],
    "wheel_front": [0.56, 0.52, 0.83, 0.88],
    "wheel_rear": [0.17, 0.49, 0.44, 0.87],
    "frame_mid": [0.33, 0.33, 0.64, 0.62],
    "bg_highfreq": [0.67, 0.05, 0.95, 0.35],
}


def to_tensor_rgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return tf.to_tensor(img).unsqueeze(0)[:, :3, :, :]


def box_to_px(box, h, w):
    x1, y1, x2, y2 = box
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def crop_tensor(t: torch.Tensor, box):
    _, _, h, w = t.shape
    x1, y1, x2, y2 = box_to_px(box, h, w)
    return t[:, :, y1:y2, x1:x2]


def evaluate_case(model_dir: Path, split: str, method: str, boxes: dict, device: torch.device):
    method_dir = model_dir / split / method
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    names = sorted([p.name for p in renders_dir.glob("*.png") if (gt_dir / p.name).is_file()])
    if not names:
        return {}

    per_crop = {
        crop_name: {"SSIM": [], "PSNR": [], "LPIPS": []}
        for crop_name in boxes
    }

    for name in names:
        pred = to_tensor_rgb(renders_dir / name).to(device)
        gt = to_tensor_rgb(gt_dir / name).to(device)
        for crop_name, box in boxes.items():
            pred_crop = crop_tensor(pred, box)
            gt_crop = crop_tensor(gt, box)
            per_crop[crop_name]["SSIM"].append(float(ssim(pred_crop, gt_crop).item()))
            per_crop[crop_name]["PSNR"].append(float(psnr(pred_crop, gt_crop).mean().item()))
            per_crop[crop_name]["LPIPS"].append(float(lpips(pred_crop, gt_crop, net_type="vgg").item()))

    out = {}
    for crop_name, metric_lists in per_crop.items():
        out[crop_name] = {
            metric: float(np.mean(values)) if values else float("nan")
            for metric, values in metric_lists.items()
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Compute local crop SSIM/PSNR/LPIPS for rendered results.")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=Path)
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--method", default="ours_30000")
    parser.add_argument("--boxes_json", type=Path, default=None, help="Optional custom crop boxes JSON.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    boxes = DEFAULT_BOXES
    if args.boxes_json is not None:
        boxes = json.loads(args.boxes_json.read_text(encoding="utf-8"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    report = {}
    for model_dir in args.model_paths:
        report[str(model_dir)] = evaluate_case(model_dir, args.split, args.method, boxes, device)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
