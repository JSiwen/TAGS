#!/usr/bin/env python3
"""v2 summarize 共享模块: 5-seed 聚合 + wandb 交叉对比 + winner gate.

v2 新增 (2026-10-14, exp4 ablations 配套):
  - discover_seed_dirs(): 兼容新旧路径 (exp4_runner.py / 旧脚本)
  - collect_seed_metrics(): 5-seed per-seed 聚合 + mean/std
  - fetch_wandb_runs(): 拉 wandb run summary (Public API, 不需要重登录)
  - cross_check_ledger_vs_wandb(): 本地 ledger vs wandb 5-seed 一致性检查
  - evaluate_winner_gate(): 论文 4.5 winner gate (PSNR≥DN+0.5, LPIPS≤DN+0.05)
  - write_multiseed_csv(): 5-seed per-seed CSV (论文表 4.6 用的格式)

被引用方:
  - scripts/summarize_bicycle_e4_vs_dn.py (bicycle §4.5/4.6)
  - scripts/summarize_vr_room_e4_vs_dn.py  (vr_room §4.5/4.6, 待创建)
  - scripts/build_section_4_6.py           (§4.6 全对比表生成器)

设计:
  - 不强制依赖 wandb (拿不到时降级到本地数据)
  - 所有 wandb 拉取都 try/except, 失败不阻塞主流程
  - seed 目录路径自动 discover, 不需要用户传 seed 列表
"""

import csv
import json
import os
import re
import statistics
import warnings
from pathlib import Path
from typing import Dict, List, Optional


# 4.5 winner gate (相对 DN 增量, 论文预设)
WINNER_GATE = {
    "min_delta_psnr": 0.5,    # PSNR 至少比 DN 高 0.5 dB
    "max_delta_lpips": 0.05,  # LPIPS 不能比 DN 高 0.05 (即 TAGS 改善至少 -0.05)
}


def _try_wandb():
    """尝试加载 wandb; 不强制依赖, 拿不到就 None."""
    try:
        import wandb  # type: ignore
        return wandb
    except ImportError:
        return None


def _count_ply_vertices(path):
    """从 PLY 头读 vertex 数 (独立于 summarize_bicycle 的实现, 共享给所有 v2 caller)."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        for _ in range(200):  # PLY header 通常前 20 行, 200 行足够
            line = handle.readline()
            if not line:
                return None
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("element vertex"):
                try:
                    return int(text.split()[-1])
                except ValueError:
                    return None
            if text == "end_header":
                return None
    return None


def _find_ply(root):
    """找一个目录下的 PLY (按 priority: iteration_30000, 顶层 point_cloud.ply, splat.ply)."""
    root = Path(root)
    if not root.exists():
        return None
    preferred = [
        root / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        root / "point_cloud.ply",
        root / "splat.ply",
    ]
    for p in preferred:
        if p.is_file():
            return p
    found = sorted(root.rglob("*.ply"))
    return found[0] if found else None


def discover_seed_dirs(tags_dir: Path, sibling_glob: Optional[str] = None) -> list:
    """发现 tags_dir 下所有 seed 目录.

    支持多种 layout:
      A) <root>/<scene>/<group>/train/seed<N>/      # exp4_runner.py 标准输出
      B) <root>/<group>/train/seed<N>/             # exp4_runner.py 单 scene 直铺
      C) <root>/seed<N>/                           # 旧脚本输出
      D) <root>/                                    # 单 run, 视为 seed0
      E) <root>_seed<N>/  (sibling_glob 启用时)     # bicycle e4 winner 历史用法
         e.g. tags_dir="output/bicycle_e4_lpips_ms_w005"
              sibling_glob="output/bicycle_e4_lpips_ms_w005_seed*"
    """
    tags_dir = Path(tags_dir)
    candidates = []
    # 1) 最深 train/seed<N>/
    for d in sorted(tags_dir.rglob("train/seed*")):
        if d.is_dir() and (any(d.glob("test/*/results.json")) or any(d.rglob("point_cloud.ply"))):
            candidates.append(d)
    if candidates:
        return candidates
    # 2) <root>/seed<N>/
    for d in sorted(tags_dir.glob("seed*")):
        if d.is_dir():
            candidates.append(d)
    if candidates:
        return candidates
    # 3) 单 run
    if (tags_dir / "test").is_dir() or any(tags_dir.rglob("point_cloud.ply")):
        candidates.append(tags_dir)
    # 4) sibling glob (layout E) — 用于同一个 case 的 _seed{N} 同级目录
    # 约定: sibling_glob 是相对于 tags_dir.parent 的尾段 (如 "bicycle_e4_lpips_ms_w005_seed*"),
    # 也兼容绝对路径 (以 / 或 drive-letter 开头).
    if sibling_glob:
        seen = {c.resolve() for c in candidates}
        if any(sibling_glob.startswith(p) for p in ("/", "\\")) or (len(sibling_glob) >= 2 and sibling_glob[1] == ":"):
            base = Path(sibling_glob).parent
            patt = Path(sibling_glob).name
        else:
            base = tags_dir.parent
            patt = sibling_glob
        for d in sorted(base.glob(patt)):
            if d.is_dir() and d.resolve() not in seen \
                    and ((d / "test").is_dir() or any(d.rglob("point_cloud.ply"))):
                candidates.append(d)
    return candidates


def parse_seed_from_dir(d: Path) -> int:
    """从目录名提取 seed, 找不到默认 0."""
    m = re.search(r"seed(\d+)", d.name)
    return int(m.group(1)) if m else 0


def collect_seed_metrics(tags_dir: Path, method_name: str = "ours_30000",
                        sibling_glob: Optional[str] = None) -> dict:
    """遍历所有 seed 目录, 抽取 (psnr, ssim, lpips, num_points).

    Returns:
        {
          "per_seed": [
            {"seed": 0, "psnr": 25.03, "ssim": 0.7487, "lpips": 0.2032,
             "num_points": 2282018, "test_dir": "..."},
            ...
          ],
          "mean": {"psnr": 25.05, "ssim": 0.7488, "lpips": 0.2035, "num_points": 2280000},
          "std":  {"psnr": 0.05,  "ssim": 0.0003, "lpips": 0.0005, "num_points": 3000},
          "n_seeds": 5,
        }
    """
    seed_dirs = discover_seed_dirs(tags_dir, sibling_glob=sibling_glob)
    per_seed = []
    for d in seed_dirs:
        seed = parse_seed_from_dir(d)
        # 找 results.json
        results_path = None
        for cand in [
            d / "test" / f"{method_name}" / "results.json",
            d / "test" / "ours_30000" / "results.json",
            d / "test" / "results.json",
            d / "results.json",
        ]:
            if cand.is_file():
                results_path = cand
                break
        if results_path is None:
            continue
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # 抽 ours_30000
        block = None
        for key in (method_name, "ours_30000", "ours_1000"):
            v = results.get(key)
            if isinstance(v, dict) and "PSNR" in v:
                block = v
                break
        if block is None:
            continue
        # num_points 从 PLY 算
        ply = _find_ply(d)
        n_pts = _count_ply_vertices(ply) if ply else None
        per_seed.append({
            "seed": seed,
            "psnr": float(block.get("PSNR", float("nan"))),
            "ssim": float(block.get("SSIM", float("nan"))),
            "lpips": float(block.get("LPIPS", float("nan"))),
            "num_points": n_pts,
            "test_dir": str(results_path.parent),
        })

    if not per_seed:
        return {"per_seed": [], "mean": {}, "std": {}, "n_seeds": 0}

    def _agg(key):
        vals = [r[key] for r in per_seed
                if r.get(key) is not None
                and not (isinstance(r[key], float) and r[key] != r[key])]  # 排除 NaN
        if not vals:
            return None
        mean_v = float(sum(vals) / len(vals))
        std_v = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        return mean_v, std_v

    mean, std = {}, {}
    for k in ("psnr", "ssim", "lpips", "num_points"):
        m = _agg(k)
        if m is not None:
            mean[k], std[k] = m

    return {
        "per_seed": per_seed,
        "mean": mean,
        "std": std,
        "n_seeds": len(per_seed),
    }


def fetch_wandb_runs(project: str, entity: str, group: str, scene: str) -> list:
    """从 wandb 拉指定 (project, group, scene) 的所有 run.

    Returns:
        list of dict {name, psnr, ssim, lpips, num_points, wandb_url, config}
    """
    wandb = _try_wandb()
    if wandb is None:
        return []
    if not project or not entity:
        return []
    api = wandb.Api()
    try:
        filters = {"config.scene": scene}
        if group:
            filters["config.group"] = group
        runs = api.runs(f"{entity}/{project}", filters=filters)
    except Exception as exc:
        warnings.warn(f"[wandb] fetch failed: {exc!r}")
        return []
    out = []
    for r in runs:
        s = r.summary or {}
        out.append({
            "name": r.name,
            "url": r.url,
            "state": r.state,
            "psnr": s.get("psnr"),
            "ssim": s.get("ssim"),
            "lpips": s.get("lpips"),
            "num_points": s.get("num_points"),
            "config": dict(r.config or {}),
        })
    return out


def cross_check_ledger_vs_wandb(local: dict, wandb_runs: list,
                                 tolerance_psnr: float = 0.05,
                                 tolerance_ssim: float = 0.001,
                                 tolerance_lpips: float = 0.005) -> dict:
    """本地 ledger 5-seed 聚合 vs wandb 5-seed 聚合, 检查一致性.

    tolerance:
      PSNR: 0.05 dB
      SSIM: 0.001
      LPIPS: 0.005 (LPIPS VGG 量级 ~0.1-0.3, 0.005 是 5% 容忍)

    Returns:
        {
          "local_n": 5, "wandb_n": 5,
          "psnr_local_mean": 25.05, "psnr_wandb_mean": 25.04, "diff": 0.01,
          "consistent": True/False/None,
          "notes": [...]
        }
    """
    notes = []
    if not local.get("per_seed") or not wandb_runs:
        return {
            "local_n": len(local.get("per_seed", [])),
            "wandb_n": len(wandb_runs),
            "consistent": None,
            "notes": ["数据不足, 跳过交叉验证"] + notes,
        }

    w_by_seed = {}
    for r in wandb_runs:
        seed = (r.get("config") or {}).get("seed")
        if seed is None:
            continue
        w_by_seed[int(seed)] = r

    tol = {"psnr": tolerance_psnr, "ssim": tolerance_ssim, "lpips": tolerance_lpips}
    diffs = {"psnr": [], "ssim": [], "lpips": []}
    for ls in local["per_seed"]:
        seed = ls["seed"]
        w = w_by_seed.get(seed)
        if w is None:
            notes.append(f"seed={seed}: wandb 缺, 本地 psnr={ls['psnr']:.4f}")
            continue
        for k in ("psnr", "ssim", "lpips"):
            lv = ls.get(k)
            wv = w.get(k)
            if lv is None or wv is None:
                continue
            d = float(lv) - float(wv)
            diffs[k].append(d)
            if abs(d) > tol[k]:
                notes.append(f"seed={seed} {k}: local={lv:.4f} vs wandb={wv:.4f} (Δ={d:+.4f}, tol=±{tol[k]})")

    consistent = all(
        abs(max(diffs[k], default=0.0)) < tol[k] and abs(min(diffs[k], default=0.0)) < tol[k]
        for k in diffs
    )
    wandb_psnr_mean = None
    if w_by_seed:
        psnr_vals = [w_by_seed[s]["psnr"] for s in w_by_seed if w_by_seed[s].get("psnr") is not None]
        if psnr_vals:
            wandb_psnr_mean = sum(psnr_vals) / len(psnr_vals)
    return {
        "local_n": len(local["per_seed"]),
        "wandb_n": len(wandb_runs),
        "psnr_local_mean": local["mean"].get("psnr"),
        "psnr_wandb_mean": wandb_psnr_mean,
        "tolerance": tol,
        "consistent": consistent,
        "notes": notes,
    }


def evaluate_winner_gate(local: dict, dn: dict) -> dict:
    """检查 4.5 winner 是否过 gate (PSNR≥DN+0.5, LPIPS≤DN+0.05)."""
    if not local.get("mean") or not dn:
        return {"passed": None, "reason": "数据不足"}
    psnr_delta = (local["mean"].get("psnr") or 0) - (dn.get("psnr") or 0)
    lpips_delta = (local["mean"].get("lpips") or 0) - (dn.get("lpips") or 0)
    ssim_delta = (local["mean"].get("ssim") or 0) - (dn.get("ssim") or 0)
    psnr_ok = psnr_delta >= WINNER_GATE["min_delta_psnr"]
    lpips_ok = lpips_delta <= WINNER_GATE["max_delta_lpips"]
    return {
        "psnr_delta": psnr_delta,
        "ssim_delta": ssim_delta,
        "lpips_delta": lpips_delta,
        "psnr_pass": psnr_ok,
        "lpips_pass": lpips_ok,
        "passed": psnr_ok and lpips_ok,
        "threshold_psnr": WINNER_GATE["min_delta_psnr"],
        "threshold_lpips": WINNER_GATE["max_delta_lpips"],
    }


def write_multiseed_csv(path: Path, scene: str, tags_group: str,
                        tags_local: dict, dn_row: dict,
                        gate: dict, wandb_runs: list) -> None:
    """5-seed per-seed CSV, 论文表 4.6 用的格式.

    Layout:
      scene, group, seed, method, psnr, ssim, lpips, num_points, psnr_local_vs_wandb, wandb_url
    """
    fields = ["scene", "group", "seed", "method",
              "psnr", "ssim", "lpips", "num_points",
              "psnr_local_vs_wandb", "wandb_url"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        # 1) per-seed TAGS
        w_by_seed = {
            r.get("config", {}).get("seed"): r
            for r in wandb_runs
            if r.get("config", {}).get("seed") is not None
        }
        for ls in tags_local.get("per_seed", []):
            wb = w_by_seed.get(ls["seed"], {})
            delta = None
            if ls.get("psnr") and wb.get("psnr"):
                delta = ls["psnr"] - wb["psnr"]
            w.writerow({
                "scene": scene,
                "group": (wb.get("config") or {}).get("group", tags_group),
                "seed": ls["seed"],
                "method": f"TAGS_{tags_group}",
                "psnr": ls.get("psnr"),
                "ssim": ls.get("ssim"),
                "lpips": ls.get("lpips"),
                "num_points": ls.get("num_points"),
                "psnr_local_vs_wandb": delta,
                "wandb_url": wb.get("url", ""),
            })
        # 2) DN row
        w.writerow({
            "scene": scene,
            "group": "DN-Splatter",
            "seed": 0,
            "method": "DN-Splatter",
            "psnr": dn_row.get("psnr"),
            "ssim": dn_row.get("ssim"),
            "lpips": dn_row.get("lpips"),
            "num_points": dn_row.get("num_points"),
            "psnr_local_vs_wandb": None,
            "wandb_url": "",
        })
        # 3) summary row (mean)
        mean = tags_local.get("mean", {})
        std = tags_local.get("std", {})

        def _fmt(k, prec):
            if not mean.get(k):
                return None
            return f"{mean[k]:.{prec}f} ± {std.get(k, 0):.{prec}f}"

        w.writerow({
            "scene": scene,
            "group": f"TAGS_{tags_group}_MEAN",
            "seed": -1,
            "method": f"TAGS_{tags_group}",
            "psnr": _fmt("psnr", 4),
            "ssim": _fmt("ssim", 5),
            "lpips": _fmt("lpips", 5),
            "num_points": mean.get("num_points"),
            "psnr_local_vs_wandb": None,
            "wandb_url": "",
        })
        # 4) gate row
        if gate.get("passed") is not None:
            w.writerow({
                "scene": scene,
                "group": "GATE",
                "seed": -1,
                "method": "winner_gate",
                "psnr": f"Δpsnr={gate['psnr_delta']:+.3f} (pass={gate['psnr_pass']})",
                "ssim": f"Δssim={gate['ssim_delta']:+.4f}",
                "lpips": f"Δlpips={gate['lpips_delta']:+.4f} (pass={gate['lpips_pass']})",
                "num_points": None,
                "psnr_local_vs_wandb": None,
                "wandb_url": "",
            })


def multiseed_summary_text(scene: str, tags_group: str,
                           tags_local: dict, dn: dict,
                           gate: dict, cross: dict) -> str:
    """生成给人读的多 seed 摘要 (markdown 表格风格)."""
    if not tags_local.get("per_seed"):
        return f"[v2] {scene}/{tags_group}: 无 seed 数据"
    lines = [f"\n## {scene} / {tags_group} — 5-seed 摘要"]
    lines.append("")
    lines.append("| seed | PSNR↑ | SSIM↑ | LPIPS↓ | num_points |")
    lines.append("|---:|---:|---:|---:|---:|")
    for ls in tags_local["per_seed"]:
        lines.append(
            f"| {ls['seed']} | {ls['psnr']:.4f} | {ls['ssim']:.5f} "
            f"| {ls['lpips']:.5f} | {ls.get('num_points') or '-'} |"
        )
    m = tags_local["mean"]; s = tags_local["std"]
    lines.append(
        f"| **MEAN** | **{m.get('psnr', 0):.4f} ± {s.get('psnr', 0):.4f}** "
        f"| **{m.get('ssim', 0):.5f} ± {s.get('ssim', 0):.5f}** "
        f"| **{m.get('lpips', 0):.5f} ± {s.get('lpips', 0):.5f}** "
        f"| **{m.get('num_points', 0):.0f}** |"
    )
    if dn:
        lines.append(
            f"| DN-Splatter | {dn.get('psnr', 0):.4f} | {dn.get('ssim', 0):.5f} "
            f"| {dn.get('lpips', 0):.5f} | {dn.get('num_points') or '-'} |"
        )
    if gate.get("passed") is not None:
        lines.append("")
        lines.append(
            f"**winner_gate**: Δpsnr={gate['psnr_delta']:+.3f} (need ≥+{gate['threshold_psnr']}, "
            f"pass={gate['psnr_pass']}), "
            f"Δlpips={gate['lpips_delta']:+.4f} (need ≤+{gate['threshold_lpips']}, "
            f"pass={gate['lpips_pass']}), "
            f"**passed={gate['passed']}**"
        )
    if cross.get("consistent") is not None:
        lines.append("")
        lines.append(
            f"**wandb cross-check**: local_n={cross['local_n']}, wandb_n={cross['wandb_n']}, "
            f"consistent={cross['consistent']}"
        )
        for note in cross.get("notes", [])[:5]:
            lines.append(f"  - {note}")
        if len(cross.get("notes", [])) > 5:
            lines.append(f"  - ... ({len(cross['notes']) - 5} more)")
    return "\n".join(lines)
