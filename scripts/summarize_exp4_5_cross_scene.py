#!/usr/bin/env python3
"""v2 跨场景汇总 (4 场景 × 全部实验组) — output/exp4_5/cross_scene_all.csv

读所有 4 个 exp4 阶段 (4.2/4.3/4.4/4.5) 的 summary, 输出统一的跨场景对照表.

输出:
  output/exp4_stats/cross_scene_all.csv: 行=阶段×组, 列=4 场景均值+户外/室内均值+95%CI
  output/exp4_stats/cross_scene_4_5_only.csv: 只含 §4.5 主表
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

V2_SCENES_3 = ["bicycle", "garden", "bonsai"]
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


def load_summary_seeds(output_root: Path, scene: str, group_id: str) -> List[Dict]:
    summary_path = output_root / scene / group_id / "summary.json"
    if not summary_path.is_file():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return list(payload.get("seeds", {}).values())


def scan_4_5_groups(output_root: Path) -> List[str]:
    """从 output/exp4_bicycle/ 下扫出所有 4_5_* 子目录."""
    scene_dir = output_root / "exp4_bicycle"
    if not scene_dir.is_dir():
        return []
    groups = []
    for d in sorted(scene_dir.iterdir()):
        if d.is_dir() and d.name.startswith("4_5_"):
            groups.append(d.name)
    return groups


def main():
    parser = argparse.ArgumentParser(description="v2 跨场景汇总")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--stats-root", default="output/exp4_stats")
    args = parser.parse_args()

    output_root = (REPO_ROOT / args.output_root).resolve()
    stats_root = (REPO_ROOT / args.stats_root).resolve()
    stats_root.mkdir(parents=True, exist_ok=True)

    # 扫 §4.5 组 (从 bicycle 下扫)
    groups_4_5 = scan_4_5_groups(output_root)
    if not groups_4_5:
        print("no 4_5_* groups found")
    else:
        print(f"§4.5 groups: {len(groups_4_5)}")

    fieldnames = ["phase", "group", "label", "n_scenes", "bicycle", "garden", "bonsai", "room",
                  "outdoor_mean", "outdoor_std", "indoor_mean", "indoor_std",
                  "delta_indoor_minus_outdoor"]
    rows = []
    for g_id in groups_4_5:
        label = g_id[len("4_5_"):]
        per_scene = {}
        for scene in V2_SCENES_4:
            seeds = load_summary_seeds(output_root, scene, g_id)
            psnrs = [s.get("psnr") for s in seeds if s.get("psnr") is not None]
            per_scene[scene] = psnrs

        # 各场景均值
        col_vals = {}
        for scene in V2_SCENES_4:
            m = _mean(per_scene[scene])
            col_vals[scene] = f"{m:.4f} (n={len(per_scene[scene])})" if m else "n/a"

        outdoor_all = []
        indoor_all = []
        for scene, psnrs in per_scene.items():
            if scene in OUTDOOR:
                outdoor_all.extend(psnrs)
            elif scene in INDOOR:
                indoor_all.extend(psnrs)

        outdoor_m = _mean(outdoor_all)
        outdoor_sd = _std(outdoor_all)
        indoor_m = _mean(indoor_all)
        indoor_sd = _std(indoor_all)
        delta = (indoor_m - outdoor_m) if (indoor_m is not None and outdoor_m is not None) else None

        rows.append({
            "phase": "4_5",
            "group": g_id,
            "label": label,
            "n_scenes": sum(1 for scene in V2_SCENES_4 if per_scene[scene]),
            "bicycle": col_vals["bicycle"],
            "garden": col_vals["garden"],
            "bonsai": col_vals["bonsai"],
            "room": col_vals["room"],
            "outdoor_mean": f"{outdoor_m:.4f}" if outdoor_m else "n/a",
            "outdoor_std": f"{outdoor_sd:.4f}" if outdoor_sd else "n/a",
            "indoor_mean": f"{indoor_m:.4f}" if indoor_m else "n/a",
            "indoor_std": f"{indoor_sd:.4f}" if indoor_sd else "n/a",
            "delta_indoor_minus_outdoor": f"{delta:.4f}" if delta is not None else "n/a",
        })

    out_csv = stats_root / "cross_scene_4_5_only.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[v2 cross-scene] 输出: {out_csv}")
    print(f"  {len(rows)} 行 (§4.5 全 7 组)")


if __name__ == "__main__":
    main()
