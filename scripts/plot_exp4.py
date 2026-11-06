#!/usr/bin/env python3
"""v2 可视化: 误差棒散点 + 配对差异图 + 户外/室内对比箱线图

依赖: matplotlib, numpy
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available")

REPO_ROOT = Path(__file__).resolve().parents[1]

V2_SCENES_4 = ["bicycle", "garden", "bonsai", "room"]
INDOOR = {"bonsai", "room"}
OUTDOOR = {"bicycle", "garden"}


def _mean(values):
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _std(values, ddof=1):
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    m = sum(clean) / len(clean)
    var = sum((v - m) ** 2 for v in clean) / (len(clean) - ddof)
    return math.sqrt(var)


def load_seeds(output_root: Path, scene: str, group_id: str) -> List[Dict]:
    summary_path = output_root / scene / group_id / "summary.json"
    if not summary_path.is_file():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return list(payload.get("seeds", {}).values())


def plot_exp4_5_ablation(output_root: Path, stats_root: Path):
    """§4.5 误差棒散点: x=高斯数量, y=PSNR, 颜色=组."""
    if not HAS_MPL:
        return
    # 扫 §4.5 组
    groups_4_5 = []
    scene_dir = output_root / "exp4_bicycle"
    if not scene_dir.is_dir():
        return
    for d in sorted(scene_dir.iterdir()):
        if d.is_dir() and d.name.startswith("4_5_"):
            groups_4_5.append(d.name)
    if not groups_4_5:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups_4_5)))
    for g_id, color in zip(groups_4_5, colors):
        all_psnrs = []
        all_ns = []
        for scene in V2_SCENES_4:
            seeds = load_seeds(output_root, scene, g_id)
            for s in seeds:
                if s.get("psnr") is not None and s.get("num_points") is not None:
                    all_psnrs.append(s["psnr"])
                    all_ns.append(s["num_points"])
        if not all_psnrs:
            continue
        m = _mean(all_psnrs)
        sd = _std(all_psnrs)
        n_mean = _mean(all_ns)
        label = g_id.replace("4_5_", "")
        ax.errorbar(n_mean, m, xerr=_std(all_ns), yerr=sd, fmt="o", color=color,
                    label=label, capsize=3, markersize=8)
    ax.set_xlabel("Number of Gaussians (mean ± SD)")
    ax.set_ylabel("PSNR (mean ± SD)")
    ax.set_title("§4.5 4-Scene Ablation: PSNR vs Num Gaussians")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    out_path = stats_root / "exp4_5_ablation_scatter.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


def plot_exp4_5_indoor_outdoor_boxplot(output_root: Path, stats_root: Path):
    """§4.5 户外 vs 室内 箱线图."""
    if not HAS_MPL:
        return
    groups_4_5 = []
    scene_dir = output_root / "exp4_bicycle"
    if not scene_dir.is_dir():
        return
    for d in sorted(scene_dir.iterdir()):
        if d.is_dir() and d.name.startswith("4_5_"):
            groups_4_5.append(d.name)
    if not groups_4_5:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    positions_outdoor = []
    positions_indoor = []
    box_outdoor = []
    box_indoor = []
    labels = []
    for i, g_id in enumerate(groups_4_5):
        label = g_id.replace("4_5_", "")
        labels.append(label)
        outdoor_psnrs = []
        indoor_psnrs = []
        for scene in V2_SCENES_4:
            seeds = load_seeds(output_root, scene, g_id)
            psnrs = [s.get("psnr") for s in seeds if s.get("psnr") is not None]
            if scene in OUTDOOR:
                outdoor_psnrs.extend(psnrs)
            elif scene in INDOOR:
                indoor_psnrs.extend(psnrs)
        positions_outdoor.append(i - 0.2)
        positions_indoor.append(i + 0.2)
        box_outdoor.append(outdoor_psnrs or [0])
        box_indoor.append(indoor_psnrs or [0])

    if box_outdoor:
        bp_out = ax.boxplot(box_outdoor, positions=positions_outdoor, widths=0.35, patch_artist=True, showfliers=False)
        for patch in bp_out["boxes"]:
            patch.set_facecolor("#9ecae1")
    if box_indoor:
        bp_in = ax.boxplot(box_indoor, positions=positions_indoor, widths=0.35, patch_artist=True, showfliers=False)
        for patch in bp_in["boxes"]:
            patch.set_facecolor("#fdae6b")

    ax.set_xticks(range(len(groups_4_5)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("PSNR (per seed)")
    ax.set_title("§4.5 Outdoor (bicycle+garden) vs Indoor (bonsai+room) PSNR Distribution")
    ax.legend([bp_out["boxes"][0], bp_in["boxes"][0]], ["outdoor", "indoor"], loc="lower right")
    ax.grid(True, alpha=0.3)
    out_path = stats_root / "exp4_5_indoor_outdoor_boxplot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


def plot_exp4_4_paired_diff(output_root: Path, stats_root: Path):
    """§4.4 配对差异图: T3 vs T0, S1/S3 vs T0 等的配对差异 (跨 4 场景)."""
    if not HAS_MPL:
        return
    # 6 个独立组
    groups = ["T0", "T1", "T2", "T3", "S1", "S3"]
    group_paths = {
        "T0": "4_4_T0_NoGeo",
        "T1": "4_4_T1_GeoEarly",
        "T2": "4_4_T2_GeoMid",
        "T3": "4_4_T3_GeoLate",
        "S1": "4_4_S1_GeoWeak",
        "S3": "4_4_S3_GeoStrong",
    }
    # 对每对 (T1, T2, T3, S1, S3) vs T0 算每场景的 mean_diff
    fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)
    for ax, g_id in zip(axes, ["T1", "T2", "T3", "S1", "S3"]):
        t0_psnrs_per_scene = {}
        g_psnrs_per_scene = {}
        for scene in V2_SCENES_4:
            t0_seeds = load_seeds(output_root, scene, group_paths["T0"])
            g_seeds = load_seeds(output_root, scene, group_paths[g_id])
            t0_psnrs_per_scene[scene] = [s.get("psnr") for s in t0_seeds if s.get("psnr") is not None]
            g_psnrs_per_scene[scene] = [s.get("psnr") for s in g_seeds if s.get("psnr") is not None]
        # 配对: 同 seed 配对
        deltas = []
        for scene in V2_SCENES_4:
            pairs = list(zip(t0_psnrs_per_scene[scene], g_psnrs_per_scene[scene]))
            for t0, g in pairs:
                if t0 is not None and g is not None:
                    deltas.append(g - t0)
        if deltas:
            ax.hist(deltas, bins=10, color="#4292c6", edgecolor="black")
            ax.axvline(0, color="red", linestyle="--", label="zero")
            ax.axvline(sum(deltas)/len(deltas), color="green", linestyle="-", label=f"mean={sum(deltas)/len(deltas):.3f}")
            ax.set_title(f"{g_id} vs T0")
            ax.set_xlabel("ΔPSNR")
            ax.legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle("§4.4 Paired PSNR Differences (4.4.G vs 4.4.T0)")
    out_path = stats_root / "exp4_4_paired_diff.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="v2 可视化")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--stats-root", default="output/exp4_stats")
    parser.add_argument("--sections", default="4_4,4_5")
    args = parser.parse_args()

    output_root = (REPO_ROOT / args.output_root).resolve()
    stats_root = (REPO_ROOT / args.stats_root).resolve()
    stats_root.mkdir(parents=True, exist_ok=True)

    sections = set(s.strip() for s in args.sections.split(","))
    if not HAS_MPL:
        print("matplotlib not available, 跳过所有图")
        return
    if "4_5" in sections:
        print("\n[plot] §4.5 误差棒散点 + 箱线图")
        plot_exp4_5_ablation(output_root, stats_root)
        plot_exp4_5_indoor_outdoor_boxplot(output_root, stats_root)
    if "4_4" in sections:
        print("\n[plot] §4.4 配对差异图")
        plot_exp4_4_paired_diff(output_root, stats_root)


if __name__ == "__main__":
    main()
