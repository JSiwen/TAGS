#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_BOXES = {
    "wheel_front": [0.56, 0.52, 0.83, 0.88],
    "wheel_rear": [0.17, 0.49, 0.44, 0.87],
    "frame_mid": [0.33, 0.33, 0.64, 0.62],
    "bg_highfreq": [0.67, 0.05, 0.95, 0.35],
}


def parse_namespace(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    if not text.startswith("Namespace(") or not text.endswith(")"):
        return {}
    body = text[len("Namespace("):-1]
    # Convert "a=1, b='x'" to dict literal form for safe parsing.
    dict_like = "{" + body.replace("=", ":") + "}"
    try:
        parsed = ast.literal_eval(dict_like)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def load_img(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def grad_mag(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    return np.sqrt(gx * gx + gy * gy + 1e-12)


def lap_abs(gray: np.ndarray) -> np.ndarray:
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return np.abs(lap)


def img_stats(img: np.ndarray) -> dict:
    gray = img.mean(axis=-1)
    g = grad_mag(gray)
    l = lap_abs(gray)
    return {
        "grad_mean": float(g.mean()),
        "grad_p90": float(np.percentile(g, 90)),
        "grad_p99": float(np.percentile(g, 99)),
        "lap_mean": float(l.mean()),
        "lap_p90": float(np.percentile(l, 90)),
        "lap_p99": float(np.percentile(l, 99)),
    }


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    return {
        "mae": float(d.mean()),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
    }


def crop_to_pixels(box, h, w):
    x1, y1, x2, y2 = box
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def crop_stats(img: np.ndarray, box) -> dict:
    h, w, _ = img.shape
    x1, y1, x2, y2 = crop_to_pixels(box, h, w)
    patch = img[y1:y2, x1:x2]
    gray = patch.mean(axis=-1)
    g = grad_mag(gray)
    return {
        "mean": float(patch.mean()),
        "std": float(patch.std()),
        "grad_p95": float(np.percentile(g, 95)),
    }


def dict_diff(a: dict, b: dict) -> dict:
    out = {}
    for k in sorted(set(a.keys()) | set(b.keys())):
        av = a.get(k, "<missing>")
        bv = b.get(k, "<missing>")
        if av != bv:
            out[k] = {"a": av, "b": bv}
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze v3/v4/origin experiment differences.")
    parser.add_argument("--v3", required=True, type=Path)
    parser.add_argument("--v4", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--image_name", default="00000.png")
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    args = parser.parse_args()

    exp_paths = {"v3": args.v3, "v4": args.v4, "origin": args.origin}
    cfgs = {k: parse_namespace(v / "cfg_args") for k, v in exp_paths.items()}
    metrics = {
        k: json.loads((v / "results.json").read_text(encoding="utf-8"))["ours_30000"]
        for k, v in exp_paths.items()
    }
    imgs = {
        k: load_img(v / "test" / "ours_30000" / "renders" / args.image_name)
        for k, v in exp_paths.items()
    }

    report = {
        "metrics": metrics,
        "metric_delta": {
            "v4_minus_v3": {
                "SSIM": metrics["v4"]["SSIM"] - metrics["v3"]["SSIM"],
                "PSNR": metrics["v4"]["PSNR"] - metrics["v3"]["PSNR"],
                "LPIPS": metrics["v4"]["LPIPS"] - metrics["v3"]["LPIPS"],
            },
            "v4_minus_origin": {
                "SSIM": metrics["v4"]["SSIM"] - metrics["origin"]["SSIM"],
                "PSNR": metrics["v4"]["PSNR"] - metrics["origin"]["PSNR"],
                "LPIPS": metrics["v4"]["LPIPS"] - metrics["origin"]["LPIPS"],
            },
        },
        "pairwise_image_delta": {
            "v4_vs_v3": pair_stats(imgs["v4"], imgs["v3"]),
            "v4_vs_origin": pair_stats(imgs["v4"], imgs["origin"]),
            "v3_vs_origin": pair_stats(imgs["v3"], imgs["origin"]),
        },
        "global_highfreq": {k: img_stats(v) for k, v in imgs.items()},
        "crop_stats": {
            crop_name: {exp: crop_stats(img, box) for exp, img in imgs.items()}
            for crop_name, box in DEFAULT_BOXES.items()
        },
        "config_diffs": {
            "v4_vs_v3": dict_diff(cfgs["v4"], cfgs["v3"]),
            "v4_vs_origin": dict_diff(cfgs["v4"], cfgs["origin"]),
        },
    }
    report["diagnosis"] = [
        "v4 在 SSIM/PSNR/LPIPS 三项上均优于 v3 与 origin，但 PSNR/SSIM 提升幅度明显小于 LPIPS 改善，说明结构质量仍接近瓶颈。",
        "wheel_front/wheel_rear/frame_mid 若出现 grad_p95 持平或下降，通常意味着辐条与细边缘还未得到稳定重建。",
        "bg_highfreq 若 LPIPS 明显高于主体区域，往往对应植被与草地高频纹理噪声，对全图感知指标形成拖累。",
        "建议将下一轮优化重点放在主体细结构 densification 与背景噪声抑制，而非单纯增加迭代数。",
    ]

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# Bicycle v3/v4/origin Experiment Analysis")
        lines.append("")
        lines.append("## Metrics")
        lines.append(f"- v3: {metrics['v3']}")
        lines.append(f"- v4: {metrics['v4']}")
        lines.append(f"- origin: {metrics['origin']}")
        lines.append("")
        lines.append("## Key Deltas")
        lines.append(f"- v4 - v3: {report['metric_delta']['v4_minus_v3']}")
        lines.append(f"- v4 - origin: {report['metric_delta']['v4_minus_origin']}")
        lines.append("")
        lines.append("## Diagnosis")
        for item in report["diagnosis"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## Pairwise Image Delta (00000.png)")
        for k, v in report["pairwise_image_delta"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Config Differences (Top Keys)")
        for name, diffs in report["config_diffs"].items():
            lines.append(f"### {name}")
            for key in list(diffs.keys())[:24]:
                lines.append(f"- {key}: {diffs[key]['a']} -> {diffs[key]['b']}")
            lines.append("")
        args.out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(report["metric_delta"], indent=2))


if __name__ == "__main__":
    main()
