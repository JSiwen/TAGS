#!/usr/bin/env python3
"""Aggregate 3-seed TAGS metrics for the bicycle E4 winner and the DN baseline.

Reads:
    output/bicycle_e4_lpips_ms_w005/results.json + per_view.json
    output/bicycle_e4_lpips_ms_w005_seed1/results.json + per_view.json
    output/bicycle_e4_lpips_ms_w005_seed2/results.json + per_view.json
    output/bicycle_dn_splatter/results.json + per_view.json + splat.ply

Writes:
    output/bicycle_e4_vs_dn_summary_3seed.json
    output/bicycle_e4_vs_dn_summary_3seed.csv
    output/bicycle_e4_vs_dn_summary_3seed_per_view.csv

Mean / std are reported over the three seeds.
Per-view win counts require every TAGS seed to be strictly better than DN
on the same view (intersection of view stems across all seeds and DN).
"""

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TAGS_SEED_DIRS = {
    "seed0": "bicycle_e4_lpips_ms_w005",
    "seed1": "bicycle_e4_lpips_ms_w005_seed1",
    "seed2": "bicycle_e4_lpips_ms_w005_seed2",
}
DN_DIR = "bicycle_dn_splatter"
METHOD_KEY = "ours_30000"
DN_KEY = "dn-splatter"


def _candidates_for(model_dir: Path, name: str, key: str):
    return [
        model_dir / "test" / key / name,
        model_dir / name,
    ]


def _pick_newest(matches):
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def read_results(model_dir: Path, key: str, ply_path: Path = None):
    """Prefer ``<model>/test/<key>/results.json``; fall back to legacy
    ``<model>/results.json``. If multiple candidates exist, the newest is used.
    If ``ply_path`` is given, only candidates at least as new as the ply
    are accepted.
    """
    candidates = [p for p in _candidates_for(model_dir, "results.json", key) if p.is_file()]
    if ply_path is not None and ply_path.is_file():
        candidates = [p for p in candidates if p.stat().st_mtime >= ply_path.stat().st_mtime]
    chosen = _pick_newest(candidates)
    if chosen is None:
        return None
    payload = json.loads(chosen.read_text(encoding="utf-8"))
    block = payload.get(key)
    if isinstance(block, dict):
        return block
    return None


def read_per_view(model_dir: Path, key: str, ply_path: Path = None):
    candidates = [p for p in _candidates_for(model_dir, "per_view.json", key) if p.is_file()]
    if ply_path is not None and ply_path.is_file():
        candidates = [p for p in candidates if p.stat().st_mtime >= ply_path.stat().st_mtime]
    chosen = _pick_newest(candidates)
    if chosen is None:
        return {}
    payload = json.loads(chosen.read_text(encoding="utf-8"))
    block = payload.get(key)
    if isinstance(block, dict):
        return block
    return {}


def count_ply_vertices(path: Path):
    if not path or not path.is_file():
        return None
    with path.open("rb") as handle:
        for line in handle:
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("element vertex"):
                return int(text.split()[-1])
            if text == "end_header":
                return None
    return None


def stem_metric(per_view_block, metric_name):
    block = per_view_block.get(metric_name) if isinstance(per_view_block, dict) else None
    if not isinstance(block, dict):
        return {}
    out = {}
    for name, value in block.items():
        try:
            out[Path(name).stem] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def per_view_wins(tags_blocks, dn_block):
    """tags_blocks: list of per_view blocks, one per seed.

    The DN baseline and the TAGS seeds use different view stems (DN was
    rendered against filenames like _DSC8679.png, TAGS against 00000.png).
    We therefore align by sorted-stem index, not by stem name, and report
    the alignment in ``view_alignment``.
    """
    dn_psnr = stem_metric(dn_block, "PSNR")
    dn_ssim = stem_metric(dn_block, "SSIM")
    dn_lpips = stem_metric(dn_block, "LPIPS")
    dn_names = sorted(dn_psnr)

    if not dn_names:
        return None

    # If all TAGS seeds and DN share the same set of stems, align by stem.
    common = set(dn_names)
    for block in tags_blocks:
        common &= set(stem_metric(block, "PSNR"))
    if common:
        common = sorted(common)
        alignment = "by_stem"
    else:
        # Otherwise pair by sorted index. This is a "view-rank" alignment
        # only; it is NOT the same view identity, so it must be reported
        # as a limitation.
        n_dn = len(dn_names)
        per_seed_sorted = [sorted(stem_metric(b, "PSNR")) for b in tags_blocks]
        per_seed_sorted = [s for s in per_seed_sorted if s]
        n = min(n_dn, *(len(s) for s in per_seed_sorted)) if per_seed_sorted else 0
        if n == 0:
            return None
        common = [f"rank_{i:02d}" for i in range(n)]
        alignment = "by_sorted_index"
    psnr_better = 0
    ssim_better = 0
    lpips_better = 0
    view_records = []
    for idx, name in enumerate(common):
        if alignment == "by_stem":
            dn_name = name
        else:
            dn_name = dn_names[idx]
        dn_p = dn_psnr[dn_name]
        dn_s = dn_ssim[dn_name]
        dn_l = dn_lpips[dn_name]
        tags_psnrs = []
        tags_ssims = []
        tags_lpipss = []
        for block in tags_blocks:
            ps = stem_metric(block, "PSNR")
            ss = stem_metric(block, "SSIM")
            lp = stem_metric(block, "LPIPS")
            if alignment == "by_stem":
                tags_psnrs.append(ps.get(name))
                tags_ssims.append(ss.get(name))
                tags_lpipss.append(lp.get(name))
            else:
                sorted_ps_vals = sorted(ps.values())
                sorted_ss_vals = sorted(ss.values())
                sorted_lp_vals = sorted(lp.values())
                if idx < len(sorted_ps_vals):
                    tags_psnrs.append(sorted_ps_vals[idx])
                if idx < len(sorted_ss_vals):
                    tags_ssims.append(sorted_ss_vals[idx])
                if idx < len(sorted_lp_vals):
                    tags_lpipss.append(sorted_lp_vals[idx])
        if all(v is not None and v > dn_p for v in tags_psnrs):
            psnr_better += 1
        if all(v is not None and v > dn_s for v in tags_ssims):
            ssim_better += 1
        if all(v is not None and v < dn_l for v in tags_lpipss):
            lpips_better += 1
        view_records.append({
            "view": name,
            "dn_view": dn_name,
            "dn_psnr": dn_p,
            "dn_ssim": dn_s,
            "dn_lpips": dn_l,
            "tags_psnr_mean": statistics.mean([v for v in tags_psnrs if v is not None]),
            "tags_ssim_mean": statistics.mean([v for v in tags_ssims if v is not None]),
            "tags_lpips_mean": statistics.mean([v for v in tags_lpipss if v is not None]),
        })
    return {
        "num_views": len(common),
        "psnr_tags_better": psnr_better,
        "ssim_tags_better": ssim_better,
        "lpips_tags_better": lpips_better,
        "view_alignment": alignment,
        "per_view": view_records,
    }


def ms(xs):
    clean = [float(x) for x in xs if x is not None]
    if not clean:
        return None, None
    mean = statistics.mean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--json-out", default=str(ROOT / "output" / "bicycle_e4_vs_dn_summary_3seed.json"))
    parser.add_argument("--csv-out", default=str(ROOT / "output" / "bicycle_e4_vs_dn_summary_3seed.csv"))
    parser.add_argument("--csv-per-view", default=str(ROOT / "output" / "bicycle_e4_vs_dn_summary_3seed_per_view.csv"))
    args = parser.parse_args()

    root = Path(args.root)
    json_out = Path(args.json_out)
    csv_out = Path(args.csv_out)
    csv_per_view = Path(args.csv_per_view)

    tags_results = {}
    tags_per_view = {}
    missing = []
    for label, sub in TAGS_SEED_DIRS.items():
        mdir = root / "output" / sub
        ply = mdir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        block = read_results(mdir, METHOD_KEY, ply_path=ply)
        if block is None:
            missing.append(label)
            continue
        tags_results[label] = block
        tags_per_view[label] = read_per_view(mdir, METHOD_KEY, ply_path=ply)
    if missing:
        raise SystemExit(f"Missing TAGS results for: {', '.join(missing)}")
    if set(tags_results.keys()) != set(TAGS_SEED_DIRS.keys()):
        raise SystemExit(f"Only {len(tags_results)} of 3 TAGS seeds scored: {list(tags_results.keys())}")

    dn_model_dir = root / "output" / DN_DIR
    dn_ply = dn_model_dir / "splat.ply"
    dn_block = read_results(dn_model_dir, DN_KEY, ply_path=dn_ply)
    if dn_block is None:
        raise SystemExit(f"Missing DN results under {dn_model_dir}")
    dn_per_view = read_per_view(dn_model_dir, DN_KEY, ply_path=dn_ply)
    dn_ply = dn_model_dir / "splat.ply"
    dn_points = count_ply_vertices(dn_ply)

    psnr_mean, psnr_std = ms([tags_results[k]["PSNR"] for k in tags_results])
    ssim_mean, ssim_std = ms([tags_results[k]["SSIM"] for k in tags_results])
    lpips_mean, lpips_std = ms([tags_results[k]["LPIPS"] for k in tags_results])

    summary = {
        "protocol": {
            "scene": "bicycle",
            "tags_profile": "lpips_ms_w005",
            "images": "images_4",
            "tags_resolution": 1,
            "hold": 8,
            "seeds": list(tags_results.keys()),
            "dn_depth_loss": "PearsonDepth",
            "metrics": "metrics.py VGG LPIPS",
            "note": (
                "Delta is TAGS minus DN. Positive PSNR/SSIM favors TAGS; "
                "positive LPIPS favors DN. TAGS numbers are mean +- std over 3 seeds."
            ),
        },
        "tags_seeds": {
            label: {
                "psnr": tags_results[label]["PSNR"],
                "ssim": tags_results[label]["SSIM"],
                "lpips": tags_results[label]["LPIPS"],
            }
            for label in tags_results
        },
        "tags_3seed": {
            "psnr_mean": psnr_mean,
            "psnr_std": psnr_std,
            "ssim_mean": ssim_mean,
            "ssim_std": ssim_std,
            "lpips_mean": lpips_mean,
            "lpips_std": lpips_std,
        },
        "dn": {
            "psnr": dn_block["PSNR"],
            "ssim": dn_block["SSIM"],
            "lpips": dn_block["LPIPS"],
            "num_points": dn_points,
            "ply": str(dn_ply),
        },
        "delta_tags_minus_dn": {
            "psnr": float(psnr_mean) - float(dn_block["PSNR"]) if psnr_mean is not None else None,
            "ssim": float(ssim_mean) - float(dn_block["SSIM"]) if ssim_mean is not None else None,
            "lpips": float(lpips_mean) - float(dn_block["LPIPS"]) if lpips_mean is not None else None,
        },
        "per_view_tags_better": per_view_wins(
            [tags_per_view[k] for k in tags_results], dn_per_view
        ),
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "PSNR_mean", "PSNR_std", "SSIM_mean", "SSIM_std", "LPIPS_mean", "LPIPS_std"])
        writer.writerow([
            "TAGS_3seed",
            f"{psnr_mean:.4f}", f"{psnr_std:.4f}",
            f"{ssim_mean:.5f}", f"{ssim_std:.5f}",
            f"{lpips_mean:.5f}", f"{lpips_std:.5f}",
        ])
        writer.writerow([
            "DN-Splatter",
            f"{dn_block['PSNR']:.4f}", "0.0000",
            f"{dn_block['SSIM']:.5f}", "0.00000",
            f"{dn_block['LPIPS']:.5f}", "0.00000",
        ])
        d = summary["delta_tags_minus_dn"]
        writer.writerow([
            "Delta (TAGS 3seed - DN)",
            f"{d['psnr']:+.4f}", "",
            f"{d['ssim']:+.5f}", "",
            f"{d['lpips']:+.5f}", "",
        ])

    pv = (summary["per_view_tags_better"] or {})
    if pv and pv.get("per_view"):
        with csv_per_view.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "view", "dn_psnr", "dn_ssim", "dn_lpips",
                "tags_psnr_mean", "tags_ssim_mean", "tags_lpips_mean",
            ])
            for row in pv["per_view"]:
                writer.writerow([
                    row["view"],
                    f"{row['dn_psnr']:.4f}", f"{row['dn_ssim']:.5f}", f"{row['dn_lpips']:.5f}",
                    f"{row['tags_psnr_mean']:.4f}",
                    f"{row['tags_ssim_mean']:.5f}",
                    f"{row['tags_lpips_mean']:.5f}",
                ])

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {json_out}")
    print(f"Wrote {csv_out}")
    print(f"Wrote {csv_per_view}")


if __name__ == "__main__":
    main()
