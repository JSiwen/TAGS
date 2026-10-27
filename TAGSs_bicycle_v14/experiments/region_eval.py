import json
from pathlib import Path

import torch

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


DEFAULT_REGION_BOXES = {
    "subject": [0.12, 0.20, 0.86, 0.90],
    "background": [0.00, 0.00, 1.00, 0.45],
}


class RegionEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.boxes = self._load_boxes()

    def _load_boxes(self):
        path = str(getattr(self.cfg, "region_eval_boxes_json", "")).strip()
        if not path:
            return DEFAULT_REGION_BOXES
        p = Path(path)
        if not p.is_file():
            print(f"[RegionEval] Box file not found: {p}, fallback to defaults.")
            return DEFAULT_REGION_BOXES
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:
            print(f"[RegionEval] Failed to parse {p}: {exc}, fallback to defaults.")
        return DEFAULT_REGION_BOXES

    def _crop(self, tensor, box):
        _, h, w = tensor.shape
        x1 = max(0, min(w - 1, int(float(box[0]) * w)))
        y1 = max(0, min(h - 1, int(float(box[1]) * h)))
        x2 = max(x1 + 1, min(w, int(float(box[2]) * w)))
        y2 = max(y1 + 1, min(h, int(float(box[3]) * h)))
        return tensor[:, y1:y2, x1:x2]

    def evaluate(self, pred, gt):
        region_metrics = {}
        for name, box in self.boxes.items():
            pred_crop = self._crop(pred, box)
            gt_crop = self._crop(gt, box)
            region_metrics[f"SSIM_{name}"] = float(ssim(pred_crop.unsqueeze(0), gt_crop.unsqueeze(0)).item())
            region_metrics[f"PSNR_{name}"] = float(psnr(pred_crop.unsqueeze(0), gt_crop.unsqueeze(0)).mean().item())
            region_metrics[f"LPIPS_{name}"] = float(
                lpips(pred_crop.unsqueeze(0), gt_crop.unsqueeze(0), net_type="vgg").item()
            )
            region_metrics[f"L1_{name}"] = float(
                l1_loss(pred_crop.unsqueeze(0), gt_crop.unsqueeze(0)).mean().item()
            )
        return region_metrics
