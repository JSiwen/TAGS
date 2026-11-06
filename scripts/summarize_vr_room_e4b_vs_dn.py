#!/usr/bin/env python3
"""Score DN-Splatter vr_room renders against the E4b TAGS summary.

DN renders live under output/vr_room_dn_renders/eval_seed{0,1,2}/eval/{gt,pred}/rgb.
The TAGS side is output/vr_room_e4b_summary.json, which stores per-case means
plus per_view.json names inside each case dir.

IMPORTANT: the two sides use different eval splits.
  TAGS e4b : frame_00011, 00021, ... (42 views, every 10th starting at 11)
  DN       : frame_00001, 00009, ... (53 views, every 8th starting at 1)
Only 10 views overlap. This script therefore reports both:
  * "as_is"        - each side's own mean over its own split (comparable to
                      the existing e4b summary numbers, but NOT a fair
                      head-to-head)
  * "intersection" - both sides restricted to the 10 shared views, which is
                      the only strictly paired comparison
Gaussian point counts come from the ply header (element vertex N), so the
195 MB files are never parsed.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import cv2

METRICS = ("psnr", "ssim", "lpips")


def load_json(path):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[warn] bad json {p}: {exc}", file=sys.stderr)
        return None


def psnr(gt_u8, pred_u8):
    mse = float(np.mean((gt_u8.astype(np.float64) - pred_u8.astype(np.float64)) ** 2))
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10((255.0**2) / mse))


def ssim(gt_u8, pred_u8):
    """Grayscale SSIM with the standard 11x11 gaussian window."""
    gray = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    a, b = gray(gt_u8), gray(pred_u8)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu_a = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
    mu_b = cv2.filter2D(b, -1, window)[5:-5, 5:-5]
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = cv2.filter2D(a * a, -1, window)[5:-5, 5:-5] - mu_a2
    sigma_b2 = cv2.filter2D(b * b, -1, window)[5:-5, 5:-5] - mu_b2
    sigma_ab = cv2.filter2D(a * b, -1, window)[5:-5, 5:-5] - mu_ab
    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return float(np.mean(num / den))


_LPIPS = {}


def lpips_metric(gt_u8, pred_u8, net="vgg"):
    if net not in _LPIPS:
        import torch
        import lpips as lpips_mod

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _LPIPS[net] = (lpips_mod.LPIPS(net=net).to(device).eval(), device)
    model, device = _LPIPS[net]
    import torch

    def to_t(img):
        t = torch.from_numpy(img[:, :, ::-1].copy()).float().permute(2, 0, 1)[None]
        return (t / 255.0) * 2 - 1

    with torch.no_grad():
        a = to_t(gt_u8).to(device)
        b = to_t(pred_u8).to(device)
        return float(model(a, b).item())


PLY_VERTEX_RE = re.compile(rb"element\s+vertex\s+(\d+)")


def ply_vertex_count(path):
    """Read the ply header only; never load the 195 MB body."""
    p = Path(path)
    if not p.is_file():
        return None
    with p.open("rb") as handle:
        head = handle.read(4096)
    match = PLY_VERTEX_RE.search(head)
    if not match:
        return None
    return int(match.group(1))


def image_pairs(eval_dir):
    """Yield (stem, gt_path, pred_path) for one eval_seedN dir."""
    eval_dir = Path(eval_dir)
    gt_dir = eval_dir / "gt" / "rgb"
    pred_dir = eval_dir / "pred" / "rgb"
    if not gt_dir.is_dir() or not pred_dir.is_dir():
        return []
    gts = {p.stem: p for p in gt_dir.glob("*.png")}
    preds = {p.stem: p for p in pred_dir.glob("*.png")}
    shared = sorted(set(gts) & set(preds))
    return [(stem, gts[stem], preds[stem]) for stem in shared]


def score_seed(eval_dir, use_lpips=True, lpips_net="vgg"):
    rows = []
    for stem, gt_path, pred_path in image_pairs(eval_dir):
        gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
        if gt is None or pred is None:
            print(f"[warn] unreadable in {eval_dir.name}: {stem}", file=sys.stderr)
            continue
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        row = {"image_name": f"{stem}.jpg", "stem": stem, "psnr": psnr(gt, pred), "ssim": ssim(gt, pred)}
        if use_lpips:
            row["lpips"] = lpips_metric(gt, pred, net=lpips_net)
        rows.append(row)
    return rows


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def aggregate_seed_rows(rows):
    out = {}
    for metric in METRICS:
        if metric in (rows[0] if rows else {}):
            out[metric] = mean_std([r.get(metric) for r in rows])
    return out


def tags_per_view_map(output_root):
    """Map stem -> {case: per-view metric dict} from every e4b per_view.json."""
    root = Path(output_root)
    per_case = {}
    for case_dir in sorted(root.glob("vr_room_e4b_*")) + sorted(root.glob("vr_room_e4_*")):
        pv = load_json(case_dir / "per_view.json")
        if not pv:
            continue
        views = pv.get("views") or []
        if not views:
            continue
        entry = {"case": case_dir.name, "views": {}}
        for view in views:
            stem = Path(str(view.get("image_name", ""))).stem
            entry["views"][stem] = view
        per_case[case_dir.name] = entry
    return per_case


def tags_split_names(output_root, case_name):
    pv = load_json(Path(output_root) / case_name / "per_view.json")
    if not pv:
        return set()
    return {Path(str(v.get("image_name", ""))).stem for v in pv.get("views") or []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders-root", default="output/vr_room_dn_renders")
    parser.add_argument("--ply-root", default="output/vr_room_dn_splatter")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--e4b-summary", default="output/vr_room_e4b_summary.json")
    parser.add_argument("--lpips-net", default="vgg", choices=["vgg", "alex"])
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--label", default="dn_main", help="label for this run in the report")
    parser.add_argument("--json-out", default="output/vr_room_e4b_vs_dn.json")
    parser.add_argument("--csv-out", default="output/vr_room_e4b_vs_dn.csv")
    args = parser.parse_args()

    use_lpips = not args.no_lpips
    renders_root = Path(args.renders_root)
    ply_root = Path(args.ply_root)

    seed_dirs = sorted(p for p in renders_root.glob("eval_seed*") if p.is_dir())
    if not seed_dirs:
        raise SystemExit(f"no eval_seedN dirs under {renders_root}")

    print(f"[vs-dn] scoring {len(seed_dirs)} DN seeds (lpips={'off' if not use_lpips else args.lpips_net})")
    seeds = {}
    for seed_dir in seed_dirs:
        seed_id = int(seed_dir.name.replace("eval_seed", ""))
        rows = score_seed(seed_dir / "eval", use_lpips=use_lpips, lpips_net=args.lpips_net)
        if not rows:
            print(f"[warn] no rows for {seed_dir.name}", file=sys.stderr)
            continue
        seeds[seed_id] = {
            "dir": seed_dir.name,
            "num_views": len(rows),
            "views": rows,
            "aggregate": aggregate_seed_rows(rows),
            "num_points": ply_vertex_count(ply_root / f"splat_seed{seed_id}" / "splat.ply"),
        }
        agg = seeds[seed_id]["aggregate"]
        parts = []
        for metric in METRICS:
            stat = agg.get(metric) or {}
            if stat.get("mean") is not None:
                parts.append(f"{metric}={stat['mean']:.5f}")
        print(f"  {seed_dir.name}: n={len(rows)} " + " ".join(parts))

    if not seeds:
        raise SystemExit("no DN seeds scored")

    # ---- DN aggregate across seeds (each seed keeps its own split, which is identical) ----
    dn_mean = {
        metric: mean_std([seeds[s]["aggregate"].get(metric, {}).get("mean") for s in seeds])
        for metric in METRICS
        if METRICS[0] in seeds[next(iter(seeds))]["aggregate"]
    }
    dn_points = mean_std([seeds[s]["num_points"] for s in seeds if seeds[s]["num_points"]])

    # ---- TAGS side ----
    summary = load_json(args.e4b_summary) or {}
    tags_cases = {}
    for case in summary.get("cases") or []:
        name = case.get("case")
        if not name:
            continue
        tags_cases[name] = case

    per_case = tags_per_view_map(args.output_root)
    reference_case = "vr_room_e4_lpips_ms_w005"
    ref_stems = tags_split_names(args.output_root, reference_case) or tags_split_names(
        args.output_root, "vr_room_e4b_decay24k"
    )

    dn_stems = {r["stem"] for s in seeds.values() for r in s["views"]}
    shared = sorted(dn_stems & ref_stems)

    # ---- paired comparison on shared views ----
    paired = {"num_shared_views": len(shared), "shared_views": shared, "cases": []}
    for name, entry in sorted(per_case.items()):
        views = entry["views"]
        shared_here = [stem for stem in shared if stem in views]
        if not shared_here:
            continue
        row = {"case": name, "num_paired_views": len(shared_here)}
        for metric in METRICS:
            dn_vals = []
            tags_vals = []
            for stem in shared_here:
                tags_view = views[stem]
                dn_view = next(
                    (r for r in seeds[next(iter(seeds))]["views"] if r["stem"] == stem),
                    None,
                )
                if dn_view is None:
                    continue
                if tags_view.get(metric) is None or dn_view.get(metric) is None:
                    continue
                tags_vals.append(float(tags_view[metric]))
                dn_vals.append(float(dn_view[metric]))
            if dn_vals and tags_vals:
                row[f"tags_{metric}"] = float(statistics.mean(tags_vals))
                row[f"dn_{metric}"] = float(statistics.mean(dn_vals))
                delta = row[f"dn_{metric}"] - row[f"tags_{metric}"]
                row[f"delta_{metric}"] = delta
                if metric == "lpips":
                    row["lpips_improve_ratio"] = (
                        (row["tags_lpips"] - row["dn_lpips"]) / row["tags_lpips"]
                        if row["tags_lpips"]
                        else None
                    )
                else:
                    row[f"{metric}_improve_ratio"] = (
                        delta / abs(row[f"tags_{metric}"]) if row[f"tags_{metric}"] else None
                    )
        paired["cases"].append(row)

    # ---- assemble ----
    out = {
        "label": args.label,
        "renders_root": str(renders_root),
        "ply_root": str(ply_root),
        "metric_definitions": {
            "psnr": "mean over views, RGB uint8, 10*log10(255^2/mse)",
            "ssim": "grayscale 11x11 gaussian SSIM, averaged over views",
            "lpips": f"LPIPS-{args.lpips_net} on RGB in [-1,1]",
            "num_points": "gaussian count; DN from ply header, TAGS from e4b summary",
        },
        "split_warning": (
            "TAGS e4b evaluates 42 views (every 10th from frame_00011); "
            "DN evaluates 53 views (every 8th from frame_00001). "
            "Only the 'paired' block is a strictly like-for-like comparison."
        ),
        "dn": {
            "seeds": seeds,
            "mean_over_seeds": dn_mean,
            "num_points": dn_points,
        },
        "tags_e4b": {
            "cases": tags_cases,
            "num_cases": len(tags_cases),
            "num_points_mean": mean_std(
                [c["num_points"] for c in tags_cases.values() if c.get("num_points")]
            ),
        },
        "paired": paired,
        "unpaired_as_is": [
            {
                "case": name,
                "seed": case.get("seed"),
                "tags_psnr": case.get("psnr"),
                "tags_ssim": case.get("ssim"),
                "tags_lpips": case.get("lpips"),
                "tags_num_points": case.get("num_points"),
                "dn_psnr": dn_mean.get("psnr", {}).get("mean"),
                "dn_ssim": dn_mean.get("ssim", {}).get("mean"),
                "dn_lpips": dn_mean.get("lpips", {}).get("mean"),
                "dn_num_points": dn_points.get("mean"),
            }
            for name, case in sorted(tags_cases.items())
        ],
    }

    json_out = Path(args.json_out)
    csv_out = Path(args.csv_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "tags_psnr",
                "dn_psnr",
                "delta_psnr",
                "tags_ssim",
                "dn_ssim",
                "delta_ssim",
                "tags_lpips",
                "dn_lpips",
                "delta_lpips",
                "tags_num_points",
                "dn_num_points",
            ]
        )
        for row in out["unpaired_as_is"]:
            writer.writerow(
                [
                    row["case"],
                    row["tags_psnr"],
                    row["dn_psnr"],
                    (row["dn_psnr"] - row["tags_psnr"]) if (row["dn_psnr"] is not None and row["tags_psnr"] is not None) else None,
                    row["tags_ssim"],
                    row["dn_ssim"],
                    (row["dn_ssim"] - row["tags_ssim"]) if (row["dn_ssim"] is not None and row["tags_ssim"] is not None) else None,
                    row["tags_lpips"],
                    row["dn_lpips"],
                    (row["dn_lpips"] - row["tags_lpips"]) if (row["dn_lpips"] is not None and row["tags_lpips"] is not None) else None,
                    row["tags_num_points"],
                    row["dn_num_points"],
                ]
            )

    print()
    psnr_stat = dn_mean.get("psnr") or {}
    ssim_stat = dn_mean.get("ssim") or {}
    lpips_stat = dn_mean.get("lpips") or {}
    parts = []
    if psnr_stat.get("mean") is not None:
        parts.append(f"psnr={psnr_stat['mean']:.3f}+-{psnr_stat.get('std') or 0:.3f}")
    if ssim_stat.get("mean") is not None:
        parts.append(f"ssim={ssim_stat['mean']:.5f}")
    if lpips_stat.get("mean") is not None:
        parts.append(f"lpips={lpips_stat['mean']:.5f}")
    print(f"[vs-dn] DN over seeds: {' '.join(parts)}")
    print(f"[vs-dn] DN gaussian count: {dn_points.get('mean')} (per-seed: "
          f"{ {s: seeds[s]['num_points'] for s in seeds} })")
    print(f"[vs-dn] paired views available: {len(shared)} -> {shared}")
    print(f"Wrote {json_out} and {csv_out}")


if __name__ == "__main__":
    main()