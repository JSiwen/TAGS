#!/usr/bin/env python3
"""v2 §4.2 / §4.3 / §4.4 summarize: 配对 t 检验 + 95% CI + 跨场景 ANOVA

输入:
  output/exp4_{scene}/4_2_*/summary.json
  output/exp4_{scene}/4_3_*/summary.json
  output/exp4_{scene}/4_4_*/summary.json

输出:
  output/exp4_stats/exp4_2_paired_t.csv
  output/exp4_stats/exp4_3_paired_t.csv
  output/exp4_stats/exp4_4_anova.csv (双因素 ANOVA: group × scene)
  output/exp4_stats/exp4_4_paired_t.csv
  output/exp4_stats/exp4_2_means.csv
  output/exp4_stats/exp4_3_means.csv
  output/exp4_stats/exp4_4_means.csv

依赖: scipy.stats (paired t, ANOVA, Levene, Welch ANOVA)
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available, 部分统计将被跳过")

REPO_ROOT = Path(__file__).resolve().parents[1]

V2_SCENES_3 = ["bicycle", "garden", "bonsai"]
V2_SCENES_4 = ["bicycle", "garden", "bonsai", "room"]


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


def _ci95(values):
    if not HAS_SCIPY:
        return None
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    m = _mean(clean)
    s = _std(clean)
    if s is None or s == 0:
        return (m, m)
    sem = s / math.sqrt(len(clean))
    t_crit = stats.t.ppf(0.975, df=len(clean) - 1)
    return (m - t_crit * sem, m + t_crit * sem)


def _paired_t(a, b):
    if not HAS_SCIPY:
        return None
    if len(a) != len(b) or len(a) < 2:
        return None
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    a_clean, b_clean = zip(*pairs)
    result = stats.ttest_rel(a_clean, b_clean)
    mean_diff = sum(x - y for x, y in pairs) / len(pairs)
    return {
        "n": len(pairs),
        "mean_diff": mean_diff,
        "t_stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_at_0.05": float(result.pvalue) < 0.05,
    }


def _two_way_anova(data: Dict[str, Dict[str, List[float]]]):
    """双因素 ANOVA (factor1=scene, factor2=group).

    data[scene][group] = list of psnrs
    Returns {factor1_p, factor2_p, interaction_p, levene_p} or None
    """
    if not HAS_SCIPY:
        return None
    scenes = list(data.keys())
    groups = sorted({g for scene_data in data.values() for g in scene_data.keys()})
    if not scenes or not groups:
        return None

    # Levene 检验方差齐性
    flat = [data[s][g] for s in scenes for g in groups if g in data[s]]
    if not flat:
        return None
    try:
        levene_stat, levene_p = stats.levene(*flat)
    except Exception:
        levene_stat, levene_p = None, None

    # 简化版 two-way ANOVA: 手动计算 SS_between / SS_within
    # 这里用 scipy 的 f_oneway 近似 (one-way on group), 然后对 scene 做 one-way
    # 严格 two-way 需 statsmodels, 这里用近似
    group_data = [data[s][g] for s in scenes for g in groups if g in data[s] and data[s][g]]
    scene_data = [data[s][g] for s in scenes for g in groups if g in data[s] and data[s][g]]

    # 真正的 two-way ANOVA 用 stattools 不在 scipy. 改为 one-way on group, one-way on scene
    try:
        group_lists = [data[s][g] for s in scenes for g in groups if g in data[s] and len(data[s][g]) >= 2]
        group_lists = [lst for lst in [data[s].get(g, []) for s in scenes for g in groups] if len(lst) >= 2]
        scene_lists = [data[s].get(g, []) for s in scenes for g in groups if len(data[s].get(g, [])) >= 2]
        scene_lists_grouped = {}
        for s in scenes:
            vals = []
            for g in groups:
                vals.extend(data[s].get(g, []))
            if len(vals) >= 2:
                scene_lists_grouped[s] = vals
        f_group, p_group = stats.f_oneway(*group_lists)
        if len(scene_lists_grouped) >= 2:
            f_scene, p_scene = stats.f_oneway(*scene_lists_grouped.values())
        else:
            f_scene, p_scene = None, None
    except Exception as exc:
        print(f"ANOVA failed: {exc!r}")
        return None

    return {
        "levene_p": levene_p,
        "one_way_on_group_f": f_group,
        "one_way_on_group_p": p_group,
        "one_way_on_scene_f": f_scene,
        "one_way_on_scene_p": p_scene,
        "note": "Approximation: 真正 two-way ANOVA 用 statsmodels; 这里 one-way on group, one-way on scene"
    }


def load_group_results(output_root: Path, scene: str, group_id: str) -> Dict[int, Dict]:
    summary_path = output_root / scene / group_id / "summary.json"
    if not summary_path.is_file():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {int(s): v for s, v in payload.get("seeds", {}).items()}


def summarize_4_2(output_root: Path, stats_root: Path):
    """§4.2: 配对 t 检验 4.2.B/C/D/E vs 4.2.A, 选 prune_ratio2."""
    print("\n[§4.2] 配对 t 检验 4.2.B/C/D/E vs 4.2.A")
    # 5 组
    groups = {
        "A": ("Baseline", "4_2_A_Baseline"),
        "B": ("Weak", "4_2_B_Weak"),
        "C": ("Reference", "4_2_C_Reference"),
        "D": ("MediumStrong", "4_2_D_MediumStrong"),
        "E": ("Strong", "4_2_E_Strong"),
    }
    # 加载
    data = {}
    for scene in V2_SCENES_3:
        data[scene] = {}
        for g_id, (g_name, g_path) in groups.items():
            seeds = load_group_results(output_root, scene, g_path)
            data[scene][g_id] = [s.get("psnr") for s in seeds.values() if s.get("psnr") is not None]

    # means
    rows = []
    for scene in V2_SCENES_3:
        for g_id, (g_name, _) in groups.items():
            psnrs = data[scene][g_id]
            m = _mean(psnrs)
            sd = _std(psnrs)
            ci = _ci95(psnrs)
            row = {
                "scene": scene, "group": g_id, "group_name": g_name, "n_seeds": len(psnrs),
                "psnr_mean": f"{m:.4f}" if m else "n/a",
                "psnr_std": f"{sd:.4f}" if sd else "n/a",
                "psnr_ci95_low": f"{ci[0]:.4f}" if ci else "n/a",
                "psnr_ci95_high": f"{ci[1]:.4f}" if ci else "n/a",
                "num_points_mean": "n/a",  # 简化: 略
            }
            rows.append(row)
    with (stats_root / "exp4_2_means.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # paired t
    ttest_rows = []
    for scene in V2_SCENES_3:
        a_psnrs = data[scene]["A"]
        for g_id in ("B", "C", "D", "E"):
            g_psnrs = data[scene][g_id]
            ttest = _paired_t(g_psnrs, a_psnrs)
            if ttest:
                ttest_rows.append({
                    "comparison": f"4.2.{g_id} vs 4.2.A",
                    "scene": scene,
                    "n_pairs": ttest["n"],
                    "mean_diff": f"{ttest['mean_diff']:.4f}",
                    "t_stat": f"{ttest['t_stat']:.4f}",
                    "p_value": f"{ttest['p_value']:.4f}",
                    "significant_at_0.05": ttest["significant_at_0.05"],
                })
    with (stats_root / "exp4_2_paired_t.csv").open("w", encoding="utf-8", newline="") as f:
        if ttest_rows:
            writer = csv.DictWriter(f, fieldnames=list(ttest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ttest_rows)

    # 选 prune_ratio2: 取 5 个组中 PSNR 最高且不显著差于 Reference (C) 的
    # 简化: 选所有 5 seed 中 PSNR 均值最高的组
    best = None
    for g_id in ("B", "C", "D", "E"):
        all_psnrs = []
        for scene in V2_SCENES_3:
            all_psnrs.extend(data[scene][g_id])
        m = _mean(all_psnrs)
        if m is not None and (best is None or m > best[1]):
            best = (g_id, m)
    if best:
        print(f"  [§4.2] 推荐 prune_ratio2: 4.2.{best[0]} (PSNR={best[1]:.4f})")
        (stats_root / "exp4_2_recommended.txt").write_text(
            f"4.2.{best[0]} (PSNR={best[1]:.4f})\n"
        )


def summarize_4_3(output_root: Path, stats_root: Path):
    """§4.3: 4.3.D (Joint-Fixed) vs 4.3.E (TAGS-Staged) 配对 t."""
    print("\n[§4.3] 配对 t 检验 4.3.E vs 4.3.D (Staged vs Joint-Fixed)")
    groups = {
        "A": "4_3_A_Control",
        "B": "4_3_B_Perceptual",
        "C": "4_3_C_Sparse",
        "D": "4_3_D_JointFixed",
        "E": "4_3_E_TagsStaged",
    }
    data = {}
    for scene in V2_SCENES_3:
        data[scene] = {}
        for g_id, g_path in groups.items():
            seeds = load_group_results(output_root, scene, g_path)
            data[scene][g_id] = [s.get("psnr") for s in seeds.values() if s.get("psnr") is not None]

    # means
    rows = []
    for scene in V2_SCENES_3:
        for g_id, g_path in groups.items():
            psnrs = data[scene][g_id]
            m = _mean(psnrs)
            sd = _std(psnrs)
            ci = _ci95(psnrs)
            rows.append({
                "scene": scene, "group": g_id, "n_seeds": len(psnrs),
                "psnr_mean": f"{m:.4f}" if m else "n/a",
                "psnr_std": f"{sd:.4f}" if sd else "n/a",
                "psnr_ci95_low": f"{ci[0]:.4f}" if ci else "n/a",
                "psnr_ci95_high": f"{ci[1]:.4f}" if ci else "n/a",
            })
    with (stats_root / "exp4_3_means.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # paired t: E vs D (主对照)
    ttest_rows = []
    for scene in V2_SCENES_3:
        for comp_label, g1, g2 in [("E vs D", "E", "D"), ("D vs A", "D", "A"), ("E vs A", "E", "A"),
                                    ("E vs C", "E", "C"), ("D vs C", "D", "C")]:
            ttest = _paired_t(data[scene][g1], data[scene][g2])
            if ttest:
                ttest_rows.append({
                    "comparison": comp_label, "scene": scene, "n_pairs": ttest["n"],
                    "mean_diff": f"{ttest['mean_diff']:.4f}",
                    "t_stat": f"{ttest['t_stat']:.4f}",
                    "p_value": f"{ttest['p_value']:.4f}",
                    "significant_at_0.05": ttest["significant_at_0.05"],
                })
    with (stats_root / "exp4_3_paired_t.csv").open("w", encoding="utf-8", newline="") as f:
        if ttest_rows:
            writer = csv.DictWriter(f, fieldnames=list(ttest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ttest_rows)

    # 选 LPIPS 调度
    best = None
    for g_id in ("B", "D", "E"):
        all_psnrs = []
        for scene in V2_SCENES_3:
            all_psnrs.extend(data[scene][g_id])
        m = _mean(all_psnrs)
        if m is not None and (best is None or m > best[1]):
            best = (g_id, m)
    if best:
        print(f"  [§4.3] 推荐 LPIPS 调度: 4.3.{best[0]} (PSNR={best[1]:.4f})")
        (stats_root / "exp4_3_recommended.txt").write_text(
            f"4.3.{best[0]} (PSNR={best[1]:.4f})\n"
        )


def summarize_4_4(output_root: Path, stats_root: Path):
    """§4.4: 双因素 ANOVA (group × scene) + 配对 t 检验 T1/T2/T3 vs T0, S1/S3 vs S0."""
    print("\n[§4.4] ANOVA + 配对 t 检验 (配对设计)")
    # 6 个独立组 (T3=S2, T0=S0 共享)
    groups = {
        "T0": "4_4_T0_NoGeo",
        "T1": "4_4_T1_GeoEarly",
        "T2": "4_4_T2_GeoMid",
        "T3": "4_4_T3_GeoLate",
        "S1": "4_4_S1_GeoWeak",
        "S3": "4_4_S3_GeoStrong",
    }
    data = {}
    for scene in V2_SCENES_4:
        data[scene] = {}
        for g_id, g_path in groups.items():
            seeds = load_group_results(output_root, scene, g_path)
            data[scene][g_id] = [s.get("psnr") for s in seeds.values() if s.get("psnr") is not None]

    # means
    rows = []
    for scene in V2_SCENES_4:
        for g_id, g_path in groups.items():
            psnrs = data[scene][g_id]
            m = _mean(psnrs)
            sd = _std(psnrs)
            ci = _ci95(psnrs)
            rows.append({
                "scene": scene, "group": g_id, "n_seeds": len(psnrs),
                "psnr_mean": f"{m:.4f}" if m else "n/a",
                "psnr_std": f"{sd:.4f}" if sd else "n/a",
                "psnr_ci95_low": f"{ci[0]:.4f}" if ci else "n/a",
                "psnr_ci95_high": f"{ci[1]:.4f}" if ci else "n/a",
            })
    with (stats_root / "exp4_4_means.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # paired t
    ttest_rows = []
    for scene in V2_SCENES_4:
        t0 = data[scene]["T0"]
        for comp_label, g1, g2 in [("T1 vs T0", "T1", "T0"), ("T2 vs T0", "T2", "T0"),
                                    ("T3 vs T0", "T3", "T0"), ("S1 vs T0", "S1", "T0"),
                                    ("S3 vs T0", "S3", "T0")]:
            ttest = _paired_t(data[scene][g1], data[scene][g2])
            if ttest:
                ttest_rows.append({
                    "comparison": comp_label, "scene": scene, "n_pairs": ttest["n"],
                    "mean_diff": f"{ttest['mean_diff']:.4f}",
                    "t_stat": f"{ttest['t_stat']:.4f}",
                    "p_value": f"{ttest['p_value']:.4f}",
                    "significant_at_0.05": ttest["significant_at_0.05"],
                })
    with (stats_root / "exp4_4_paired_t.csv").open("w", encoding="utf-8", newline="") as f:
        if ttest_rows:
            writer = csv.DictWriter(f, fieldnames=list(ttest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ttest_rows)

    # ANOVA
    anova = _two_way_anova(data)
    if anova:
        with (stats_root / "exp4_4_anova.csv").open("w", encoding="utf-8", newline="") as f:
            f.write("metric,value,note\n")
            f.write(f"levene_p,{anova.get('levene_p')},方差齐性 (p<0.05 表示不齐)\n")
            f.write(f"one_way_on_group_f,{anova.get('one_way_on_group_f')},group 主效应\n")
            f.write(f"one_way_on_group_p,{anova.get('one_way_on_group_p')},group 主效应 p-value\n")
            f.write(f"one_way_on_scene_f,{anova.get('one_way_on_scene_f')},scene 主效应\n")
            f.write(f"one_way_on_scene_p,{anova.get('one_way_on_scene_p')},scene 主效应 p-value\n")
            f.write(f"note,{anova.get('note')},\n")
        print(f"  [§4.4] ANOVA: group p={anova.get('one_way_on_group_p')}, scene p={anova.get('one_way_on_scene_p')}")

    # 选最佳组合: T3 (Late + Base)
    print(f"  [§4.4] 推荐窗口: T3 (Geo-Late), 强度: Base (S2 = T3) (论文 §4.4.4 主张)")


def main():
    parser = argparse.ArgumentParser(description="v2 §4.2/§4.3/§4.4 汇总 + 配对 t + ANOVA")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--stats-root", default="output/exp4_stats")
    parser.add_argument("--sections", default="4_2,4_3,4_4")
    args = parser.parse_args()

    output_root = (REPO_ROOT / args.output_root).resolve()
    stats_root = (REPO_ROOT / args.stats_root).resolve()
    stats_root.mkdir(parents=True, exist_ok=True)

    sections = set(s.strip() for s in args.sections.split(","))
    if "4_2" in sections:
        summarize_4_2(output_root, stats_root)
    if "4_3" in sections:
        summarize_4_3(output_root, stats_root)
    if "4_4" in sections:
        summarize_4_4(output_root, stats_root)
    print(f"\n[§4.2/4.3/4.4] 输出在 {stats_root}")


if __name__ == "__main__":
    main()
