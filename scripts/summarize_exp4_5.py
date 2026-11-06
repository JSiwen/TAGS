#!/usr/bin/env python3
"""exp4 §4.5 多模块消融汇总 + 配对 t 检验 + 双因素 ANOVA.

v2 (2026-10-14), 配套 4 个 exp4 .ps1 脚本:
  - run_exp4_2_sweep.ps1
  - run_exp4_3_collaboration.ps1
  - run_exp4_4_geometry.ps1
  - run_exp4_5_ablation.ps1

输出物:
  - output/exp4_stats/exp4_5_ablation_matrix.csv    (4 场景 x 7 组, 5-seed mean ± SD)
  - output/exp4_stats/exp4_5_paired_t.csv            (4.5.G vs 4.5.C + 4.5.B/C/D/E vs 4.5.F)
  - output/exp4_stats/exp4_5_anova.csv               (scene x ablation, 双因素 ANOVA)
  - output/exp4_stats/exp4_5_efficiency.csv          (高斯数 / 训练时间 / 显存 / FPS)
  - output/exp4_stats/exp4_5_indoor_outdoor.csv      (户外 vs 室内, 配对 t)
  - output/exp4_stats/exp4_5_cross_scene.csv         (跨场景均值)

用法:
  python scripts/summarize_exp4_5.py
  python scripts/summarize_exp4_5.py --scenes bicycle,garden,bonsai,room
  python scripts/summarize_exp4_5.py --output-root output/exp4_bicycle
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

# §4.5 7 组定义
GROUPS_4_5 = [
    "A_Baseline3DGS",
    "B_woPerceptual",
    "C_woSPA",
    "D_woGeo",
    "E_woResolution",
    "F_FullTAGS",
    "G_NegCtrl",
]

SCENES_DEFAULT = ["bicycle", "garden", "bonsai", "room"]


def _read_results_json(model_dir: Path) -> dict:
    """读 model_dir 下的 results.json (兼容 test/<method>/results.json)."""
    candidates = [
        model_dir / "results.json",
        model_dir / "test" / "results.json",
        model_dir / "test" / "ours_30000" / "results.json",
    ]
    for c in candidates:
        if c.is_file():
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def _extract_metrics(results: dict, method: str = "ours_30000") -> dict:
    """从 results.json 抽 (psnr, ssim, lpips)."""
    block = results.get(method) or {}
    if not block and results:
        for v in results.values():
            if isinstance(v, dict) and "PSNR" in v:
                block = v
                break
    if not block:
        return {}
    return {
        "psnr": block.get("PSNR"),
        "ssim": block.get("SSIM"),
        "lpips": block.get("LPIPS"),
    }


def _count_ply_points(model_dir: Path) -> Optional[int]:
    """从 point_cloud/iteration_30000/point_cloud.ply 读 vertex 数."""
    candidates = [
        model_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        model_dir / "point_cloud.ply",
        model_dir / "splat.ply",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            with p.open("rb") as f:
                for _ in range(200):
                    line = f.readline()
                    if not line:
                        return None
                    text = line.decode("ascii", errors="ignore").strip()
                    if text.startswith("element vertex"):
                        try:
                            return int(text.split()[-1])
                        except ValueError:
                            return None
                    if text == "end_header":
                        return None
        except Exception:
            pass
    return None


def collect_run_metrics(group_dir: Path, scene: str, group: str,
                        method: str = "ours_30000") -> List[dict]:
    """收集单个 (scene, group) 下所有 seed 的指标.

    Returns:
        list of dict {seed, psnr, ssim, lpips, num_points}
    """
    train_root = group_dir / "train"
    if not train_root.is_dir():
        return []
    out = []
    for seed_dir in sorted(train_root.glob("seed*")):
        m = re.match(r"seed(\d+)", seed_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        results = _read_results_json(seed_dir)
        if not results:
            continue
        metrics = _extract_metrics(results, method=method)
        n_pts = _count_ply_points(seed_dir)
        out.append({
            "seed": seed,
            "scene": scene,
            "group": group,
            **metrics,
            "num_points": n_pts,
        })
    return out


def aggregate_per_seed(runs: List[dict]) -> dict:
    """5-seed → mean / std / n."""
    if not runs:
        return {"mean": {}, "std": {}, "n": 0, "per_seed": []}
    keys = ("psnr", "ssim", "lpips", "num_points")
    mean, std = {}, {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        if vals:
            mean[k] = sum(vals) / len(vals)
            std[k] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(runs), "per_seed": runs}


def _paired_t_test(a: List[float], b: List[float]) -> Optional[dict]:
    """配对 t 检验 (a_i - b_i, 配对 i = seed). Returns {t, p, df, mean_diff, ci_low, ci_high}."""
    if not a or not b or len(a) != len(b):
        return None
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = sum(diffs) / n
    if n < 2:
        return {"n": n, "mean_diff": mean_d, "t": None, "p": None, "df": 0, "ci_low": None, "ci_high": None}
    std_d = statistics.stdev(diffs)
    if std_d < 1e-9:
        return {"n": n, "mean_diff": mean_d, "t": float("inf"), "p": 0.0, "df": n - 1, "ci_low": mean_d, "ci_high": mean_d}
    se = std_d / math.sqrt(n)
    t = mean_d / se
    df = n - 1
    # 双尾 p-value (尝试 scipy; 若无则用近似)
    p = _two_tailed_p_t(t, df)
    # 95% CI (t critical ≈ 2.776 for df=4; 2.571 for df=5; 2.447 for df=6; 2.365 for df=7; 2.306 for df=8; 2.262 for df=9)
    t_crit = _t_critical_95(df)
    ci_low = mean_d - t_crit * se
    ci_high = mean_d + t_crit * se
    return {
        "n": n, "mean_diff": mean_d, "std_diff": std_d, "t": t, "p": p, "df": df,
        "ci_low": ci_low, "ci_high": ci_high,
    }


def _two_tailed_p_t(t: float, df: int) -> Optional[float]:
    """双尾 p-value (use scipy if available, else normal approximation)."""
    try:
        from scipy import stats  # type: ignore
        return float(2 * (1 - stats.t.cdf(abs(t), df)))
    except ImportError:
        # 正常近似 (df 大时足够准)
        # p ≈ 2 * (1 - Φ(|t|))
        return float(2 * (1 - _normal_cdf(abs(t))))


def _normal_cdf(x: float) -> float:
    """标准正态 CDF 近似 (Abramowitz & Stegun 7.1.26)."""
    # erf 近似
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2π)
    p = d * math.exp(-x * x / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    )
    return 1 - p if x > 0 else p


def _t_critical_95(df: int) -> float:
    """t critical at α=0.05 (two-tailed, 95% CI)."""
    # 常用 df 查表
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        30: 2.042, 60: 2.000, 120: 1.980, 10000: 1.960,
    }
    if df in table:
        return table[df]
    if df > 20:
        # 渐近 1.96
        return 1.96 + 2.4 / df
    return 2.0


def _two_way_anova(groups: Dict[str, Dict[str, List[float]]]) -> Optional[dict]:
    """简化双因素 ANOVA (scene x ablation).

    Args:
        groups: {ablation -> {scene -> [values per seed]}}
        假设每个 (ablation, scene) cell 都有相同数量 seed (配对设计).

    Returns:
        {F_scene, p_scene, F_ablation, p_ablation, F_interaction, p_interaction, df_*}
    """
    try:
        import numpy as np
        from scipy import stats  # type: ignore
    except ImportError:
        return None

    ablations = list(groups.keys())
    if len(ablations) < 2:
        return None
    scenes = sorted({s for g in groups.values() for s in g.keys()})
    if len(scenes) < 2:
        return None
    # 每个 cell 的 list 长度
    n = None
    for ab in ablations:
        for sc in scenes:
            vals = groups[ab].get(sc, [])
            if n is None:
                n = len(vals)
            elif len(vals) != n:
                return {"error": "unbalanced cells", "sizes": {ab: {s: len(groups[ab].get(s, [])) for s in scenes}}}
    if not n or n < 2:
        return None

    # Build data matrix: ablation x scene x rep
    data = np.array([[[groups[ab].get(sc, [float("nan")] * n)[i] for i in range(n)]
                      for sc in scenes] for ab in ablations])  # shape (a, s, n)
    a, s, n_rep = data.shape
    grand_mean = data.mean()
    ab_means = data.mean(axis=(1, 2))  # (a,)
    sc_means = data.mean(axis=(0, 2))  # (s,)
    cell_means = data.mean(axis=2)  # (a, s)

    # SS
    ss_ablation = n_rep * s * sum((m - grand_mean) ** 2 for m in ab_means)
    ss_scene = n_rep * a * sum((m - grand_mean) ** 2 for m in sc_means)
    ss_interaction = n_rep * sum((cell_means[i, j] - ab_means[i] - sc_means[j] + grand_mean) ** 2
                                 for i in range(a) for j in range(s))
    ss_total = sum((data[i, j, k] - grand_mean) ** 2 for i in range(a) for j in range(s) for k in range(n_rep))
    ss_within = ss_total - ss_ablation - ss_scene - ss_interaction

    df_ab = a - 1
    df_sc = s - 1
    df_inter = df_ab * df_sc
    df_within = a * s * (n_rep - 1)

    if df_within == 0 or df_ab == 0 or df_sc == 0:
        return None
    if df_inter == 0:
        # 不能算 interaction (e.g. only 1 ablation × 1 scene 太极端, 一般 df_inter>0 因为 a>=2,s>=2)
        ms_inter = 0
        F_inter = float("nan")
        p_inter = float("nan")
    else:
        ms_inter = ss_interaction / df_inter
        F_inter = ms_inter / (ss_within / df_within)
        p_inter = float(1 - stats.f.cdf(F_inter, df_inter, df_within))

    ms_ab = ss_ablation / df_ab
    ms_sc = ss_scene / df_sc
    F_ab = ms_ab / (ss_within / df_within)
    F_sc = ms_sc / (ss_within / df_within)
    p_ab = float(1 - stats.f.cdf(F_ab, df_ab, df_within))
    p_sc = float(1 - stats.f.cdf(F_sc, df_sc, df_within))

    return {
        "F_ablation": F_ab, "p_ablation": p_ab, "df_ablation": df_ab,
        "F_scene": F_sc, "p_scene": p_sc, "df_scene": df_sc,
        "F_interaction": F_inter, "p_interaction": p_inter, "df_interaction": df_inter,
        "df_within": df_within,
        "n_seeds_per_cell": n_rep,
        "n_ablations": a, "n_scenes": s,
        "SS_ablation": ss_ablation, "SS_scene": ss_scene,
        "SS_interaction": ss_interaction, "SS_within": ss_within, "SS_total": ss_total,
    }


def collect_all_metrics(scenes: List[str], group_prefix: str, root: Path) -> Dict[Tuple[str, str], List[dict]]:
    """收集所有 (scene, group) 的 per-seed 数据.

    group_prefix: "4_5" or "4_2" or "4_3" or "4_4"
    """
    out = {}
    for scene in scenes:
        scene_dir = root / f"exp4_{scene}"
        if not scene_dir.is_dir():
            continue
        for group_dir in sorted(scene_dir.glob(f"{group_prefix}_*")):
            group_id = group_dir.name[len(group_prefix) + 1:]  # 去掉 "4_5_" 前缀
            runs = collect_run_metrics(group_dir, scene, group_id)
            if runs:
                out[(scene, group_id)] = runs
    return out


def main():
    parser = argparse.ArgumentParser(description="exp4 §4.5 ablation summary with paired t + ANOVA")
    parser.add_argument("--scenes", default="bicycle,garden,bonsai,room")
    parser.add_argument("--root", default="output", help="exp4 数据根目录, 默认 output")
    parser.add_argument("--group-prefix", default="4_5", help="4_2 / 4_3 / 4_4 / 4_5")
    parser.add_argument("--stats-dir", default="output/exp4_stats", help="统计输出目录")
    parser.add_argument("--no-scipy", action="store_true", help="即使有 scipy 也不用 (用于 cross-check)")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    root = (ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    stats_dir = (ROOT / args.stats_dir) if not Path(args.stats_dir).is_absolute() else Path(args.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # 1) 收集所有 run
    print(f"[exp4_{args.group_prefix}] collecting from {root}/exp4_*/*")
    print(f"[exp4_{args.group_prefix}] scenes: {scenes}")
    all_runs = collect_all_metrics(scenes, args.group_prefix, root)
    print(f"[exp4_{args.group_prefix}] found {len(all_runs)} (scene, group) cells")
    for (sc, g), runs in sorted(all_runs.items()):
        print(f"  {sc:10s} / {g:25s} : {len(runs)} seed(s)")

    # 2) 聚合: 每 cell 5-seed mean/std
    aggregated = {k: aggregate_per_seed(v) for k, v in all_runs.items()}

    # 3) 主表 (矩阵)
    # group_prefix 如 "4_5" → 输出文件前缀 "exp4_5_*"
    out_prefix = args.group_prefix
    main_csv = stats_dir / f"exp{out_prefix}_ablation_matrix.csv"
    with main_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "group", "n_seeds", "psnr_mean", "psnr_std",
                    "ssim_mean", "ssim_std", "lpips_mean", "lpips_std",
                    "num_points_mean", "num_points_std"])
        for (scene, group_id), agg in sorted(aggregated.items()):
            m = agg["mean"]; s = agg["std"]
            w.writerow([
                scene, group_id, agg["n"],
                f"{m.get('psnr', 0):.4f}", f"{s.get('psnr', 0):.4f}",
                f"{m.get('ssim', 0):.5f}", f"{s.get('ssim', 0):.5f}",
                f"{m.get('lpips', 0):.5f}", f"{s.get('lpips', 0):.5f}",
                f"{m.get('num_points', 0):.0f}" if m.get("num_points") else "",
                f"{s.get('num_points', 0):.0f}" if s.get("num_points") else "",
            ])
    print(f"[exp4_{args.group_prefix}] wrote {main_csv}")

    # 4) 跨场景均值 (各 group 的 4 场景 mean ± SD)
    cross_csv = stats_dir / f"exp{out_prefix}_cross_scene.csv"
    by_group: Dict[str, List[dict]] = {}
    for (scene, group_id), agg in aggregated.items():
        by_group.setdefault(group_id, []).append({"scene": scene, **agg})
    with cross_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_scenes", "psnr_mean", "psnr_std",
                    "ssim_mean", "ssim_std", "lpips_mean", "lpips_std",
                    "num_points_mean", "num_points_std"])
        for group_id, cells in sorted(by_group.items()):
            psnrs = [c["mean"].get("psnr", 0) for c in cells if c["mean"].get("psnr")]
            ssims = [c["mean"].get("ssim", 0) for c in cells if c["mean"].get("ssim")]
            lpipses = [c["mean"].get("lpips", 0) for c in cells if c["mean"].get("lpips")]
            pts = [c["mean"].get("num_points", 0) for c in cells if c["mean"].get("num_points")]
            w.writerow([
                group_id, len(cells),
                f"{sum(psnrs)/len(psnrs):.4f}" if psnrs else "",
                f"{statistics.pstdev(psnrs):.4f}" if len(psnrs) > 1 else "0.0",
                f"{sum(ssims)/len(ssims):.5f}" if ssims else "",
                f"{statistics.pstdev(ssims):.5f}" if len(ssims) > 1 else "0.0",
                f"{sum(lpipses)/len(lpipses):.5f}" if lpipses else "",
                f"{statistics.pstdev(lpipses):.5f}" if len(lpipses) > 1 else "0.0",
                f"{sum(pts)/len(pts):.0f}" if pts else "",
                f"{statistics.pstdev(pts):.0f}" if len(pts) > 1 else "0",
            ])
    print(f"[exp4_{args.group_prefix}] wrote {cross_csv}")

    # 5) 配对 t 检验 (默认 vs full + negative control)
    paired_csv = stats_dir / f"exp{out_prefix}_paired_t.csv"
    pairs = []
    if "F_FullTAGS" in by_group and "G_NegCtrl" in by_group:
        pairs.append(("G_NegCtrl_vs_F_FullTAGS", "G_NegCtrl", "F_FullTAGS"))
    if "F_FullTAGS" in by_group:
        for g in ["A_Baseline3DGS", "B_woPerceptual", "C_woSPA", "D_woGeo", "E_woResolution"]:
            if g in by_group:
                pairs.append((f"{g}_vs_F_FullTAGS", g, "F_FullTAGS"))
    # 跨场景 per-scene 配对: 同样 (group A, group B, scene X) 各 5 seed 配对
    with paired_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "metric", "mean_diff", "std_diff", "t", "df", "p", "ci_low", "ci_high", "n_per_group"])
        for label, g1, g2 in pairs:
            for metric in ("psnr", "ssim", "lpips"):
                # 跨场景 paired: 把每个 (scene, seed) 看作一个 pair
                pairs_per_scene = []
                for scene in scenes:
                    cell1 = aggregated.get((scene, g1), {})
                    cell2 = aggregated.get((scene, g2), {})
                    if cell1.get("n", 0) >= 2 and cell2.get("n", 0) >= 2:
                        # 各 cell 内的 seed 间配对
                        for s1, s2 in zip(cell1["per_seed"], cell2["per_seed"]):
                            if s1["seed"] == s2["seed"]:
                                pairs_per_scene.append((s1.get(metric), s2.get(metric)))
                pairs_per_scene = [(a, b) for a, b in pairs_per_scene if a is not None and b is not None]
                if len(pairs_per_scene) >= 2:
                    a_vals, b_vals = zip(*pairs_per_scene)
                    res = _paired_t_test(list(a_vals), list(b_vals))
                    if res:
                        w.writerow([label, metric,
                                    f"{res['mean_diff']:+.4f}",
                                    f"{res.get('std_diff', 0):.4f}",
                                    f"{res['t']:+.4f}" if res.get('t') is not None else "NA",
                                    res['df'],
                                    f"{res['p']:.4g}" if res.get('p') is not None else "NA",
                                    f"{res['ci_low']:+.4f}", f"{res['ci_high']:+.4f}",
                                    res['n']])
    print(f"[exp4_{args.group_prefix}] wrote {paired_csv}")

    # 6) 双因素 ANOVA (scene x group)
    anova_csv = stats_dir / f"exp{out_prefix}_anova.csv"
    anova_results = {}
    for metric in ("psnr", "ssim", "lpips"):
        groups_for_anova: Dict[str, Dict[str, List[float]]] = {}
        for (scene, group_id), agg in aggregated.items():
            vals = [r.get(metric) for r in agg["per_seed"] if r.get(metric) is not None]
            if vals:
                groups_for_anova.setdefault(group_id, {})[scene] = vals
        if len(groups_for_anova) >= 2 and len({s for g in groups_for_anova.values() for s in g}) >= 2:
            res = _two_way_anova(groups_for_anova)
            if res:
                anova_results[metric] = res
    with anova_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "F_ablation", "p_ablation", "df_ablation",
                    "F_scene", "p_scene", "df_scene",
                    "F_interaction", "p_interaction", "df_interaction",
                    "df_within", "n_seeds_per_cell", "n_ablations", "n_scenes"])
        for metric, res in anova_results.items():
            def _f(x): return f"{x:.4f}" if isinstance(x, (int, float)) and not math.isnan(x) else "NA"
            w.writerow([metric,
                        _f(res.get("F_ablation")), _f(res.get("p_ablation")), res.get("df_ablation"),
                        _f(res.get("F_scene")), _f(res.get("p_scene")), res.get("df_scene"),
                        _f(res.get("F_interaction")), _f(res.get("p_interaction")), res.get("df_interaction"),
                        res.get("df_within"), res.get("n_seeds_per_cell"),
                        res.get("n_ablations"), res.get("n_scenes")])
    print(f"[exp4_{args.group_prefix}] wrote {anova_csv}")

    # 7) 室内 vs 户外 (bicycle+garden vs bonsai+room)
    indoor_outdoor_csv = stats_dir / f"exp{out_prefix}_indoor_outdoor.csv"
    outdoor = ("bicycle", "garden")
    indoor = ("bonsai", "room")
    with indoor_outdoor_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "metric", "outdoor_mean", "indoor_mean", "delta", "p"])
        for group_id in sorted(by_group.keys()):
            outdoor_vals: Dict[str, List[float]] = {"psnr": [], "ssim": [], "lpips": []}
            indoor_vals: Dict[str, List[float]] = {"psnr": [], "ssim": [], "lpips": []}
            for scene in outdoor:
                cell = aggregated.get((scene, group_id), {})
                for m in ("psnr", "ssim", "lpips"):
                    for r in cell.get("per_seed", []):
                        if r.get(m) is not None:
                            outdoor_vals[m].append(r[m])
            for scene in indoor:
                cell = aggregated.get((scene, group_id), {})
                for m in ("psnr", "ssim", "lpips"):
                    for r in cell.get("per_seed", []):
                        if r.get(m) is not None:
                            indoor_vals[m].append(r[m])
            for metric in ("psnr", "ssim", "lpips"):
                o = outdoor_vals[metric]; i = indoor_vals[metric]
                if len(o) >= 2 and len(i) >= 2:
                    o_mean = sum(o) / len(o); i_mean = sum(i) / len(i)
                    res = _paired_t_test(o, i)  # 不严格 paired, 但仍是独立 t-test
                    p_str = f"{res['p']:.4g}" if res and res.get('p') is not None else "NA"
                    w.writerow([group_id, metric, f"{o_mean:.4f}", f"{i_mean:.4f}",
                                f"{o_mean - i_mean:+.4f}", p_str])
    print(f"[exp4_{args.group_prefix}] wrote {indoor_outdoor_csv}")

    # 8) 写完整 json
    full_json = stats_dir / f"exp{out_prefix}_summary.json"
    payload = {
        "scenes": scenes,
        "group_prefix": args.group_prefix,
        "n_cells": len(aggregated),
        "aggregated": {f"{sc}/{g}": agg for (sc, g), agg in aggregated.items()},
        "anova": anova_results,
    }
    full_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[exp4_{args.group_prefix}] wrote {full_json}")
    print(f"[exp4_{args.group_prefix}] done.")


import re  # noqa: E402

if __name__ == "__main__":
    main()
