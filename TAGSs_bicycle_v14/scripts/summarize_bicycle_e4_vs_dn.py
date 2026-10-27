#!/usr/bin/env python3
"""Score bicycle E4 winner and DN-Splatter with the repo metrics.py.

v2 新增 (2026-10-14):
  - 读 wandb (Public API, 不需要重登录, 拉 5-seed mean/std 与本地 ledger 交叉)
  - 5-seed 聚合 (output/<scene>/<group>/train/seed{0..4}/) + 旧路径兼容
  - 4.5 winner 推荐 + gate check (PSNR≥DN+0.5, LPIPS≤DN-0.05)
  - 不依赖 wandb 也不阻塞: wandb 拿不到时降级到本地 ledger
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def count_ply_vertices(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                return None
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("element vertex"):
                return int(text.split()[-1])
            if text == "end_header":
                return None


def find_ply(root):
    root = Path(root)
    if not root.exists():
        return None
    preferred = [
        root / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        root / "point_cloud.ply",
        root / "splat.ply",
    ]
    for path in preferred:
        if path.is_file():
            return path
    found = sorted(root.rglob("*.ply"))
    return found[0] if found else None


def stage_dn_renders(src_eval, dest_method):
    """Copy DN pred/gt rgb into the metrics.py test/<method>/{renders,gt} layout."""
    src_eval = Path(src_eval)
    dest_method = Path(dest_method)
    pred = src_eval / "pred" / "rgb"
    gt = src_eval / "gt" / "rgb"
    if not pred.is_dir() or not gt.is_dir():
        return 0
    renders = dest_method / "renders"
    gts = dest_method / "gt"
    renders.mkdir(parents=True, exist_ok=True)
    gts.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(pred.iterdir()):
        if not path.is_file() or path.suffix not in IMAGE_EXTS:
            continue
        match = None
        for ext in IMAGE_EXTS:
            candidate = gt / f"{path.stem}{ext}"
            if candidate.is_file():
                match = candidate
                break
        if match is None:
            continue
        shutil.copy2(path, renders / path.name)
        shutil.copy2(match, gts / path.name)
        copied += 1
    return copied


def pick_method(results):
    if not isinstance(results, dict):
        return None, None
    for key in ("ours_30000", "dn-splatter", "ours_1000"):
        metrics = results.get(key)
        if isinstance(metrics, dict) and "PSNR" in metrics:
            return key, metrics
    for key, metrics in results.items():
        if isinstance(metrics, dict) and "PSNR" in metrics:
            return key, metrics
    return None, None


def _results_candidates(model_dir):
    model_dir = Path(model_dir)
    return [
        model_dir / "test" / "results.json",
        model_dir / "test" / "ours_30000" / "results.json",
        model_dir / "test" / "dn-splatter" / "results.json",
        model_dir / "results.json",
    ]


def read_metrics(model_dir, ply_path=None):
    candidates = _results_candidates(model_dir)
    if ply_path is not None and Path(ply_path).is_file():
        candidates = [p for p in candidates if p.is_file() and p.stat().st_mtime >= Path(ply_path).stat().st_mtime]
    chosen = _pick_newest(candidates)
    if chosen is None:
        return None, None
    results = json.loads(chosen.read_text(encoding="utf-8"))
    return pick_method(results)


def _per_view_candidates(model_dir, method):
    model_dir = Path(model_dir)
    return [
        model_dir / "test" / method / "per_view.json",
        model_dir / "per_view.json",
    ]


def _pick_newest(paths):
    paths = [p for p in paths if p.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def read_per_view(model_dir, method, ply_path=None):
    if not method:
        return {}
    candidates = _per_view_candidates(model_dir, method)
    if ply_path is not None and Path(ply_path).is_file():
        candidates = [p for p in candidates if p.is_file() and p.stat().st_mtime >= Path(ply_path).stat().st_mtime]
    chosen = _pick_newest(candidates)
    if chosen is None:
        return {}
    payload = json.loads(chosen.read_text(encoding="utf-8"))
    block = payload.get(method) or {}
    return block


def stem_map(metric_dict):
    if not isinstance(metric_dict, dict):
        return {}
    return {Path(name).stem: float(value) for name, value in metric_dict.items()}


def per_view_wins(tags_block, dn_block):
    """Pair by stem intersection; if empty, fall back to sorted-index pairing
    (TAGS uses images_4 numbering, DN uses original _DSC#### names)."""
    tags_psnr = stem_map((tags_block or {}).get("PSNR"))
    dn_psnr = stem_map((dn_block or {}).get("PSNR"))
    tags_ssim = stem_map((tags_block or {}).get("SSIM"))
    dn_ssim = stem_map((dn_block or {}).get("SSIM"))
    tags_lpips = stem_map((tags_block or {}).get("LPIPS"))
    dn_lpips = stem_map((dn_block or {}).get("LPIPS"))

    common = set(tags_psnr) & set(dn_psnr)
    if common:
        names = sorted(common)
        alignment = "by_stem"
    else:
        dn_names = sorted(dn_psnr)
        tags_names = sorted(tags_psnr)
        n = min(len(dn_names), len(tags_names))
        if n == 0:
            return None
        names = [f"rank_{i:02d}" for i in range(n)]
        alignment = "by_sorted_index"

    def per_metric(tags_map, dn_map, op):
        out = 0
        for name in names:
            if alignment == "by_stem":
                t, d = tags_map.get(name), dn_map.get(name)
            else:
                idx = int(name.split("_")[1])
                t = tags_map.get(tags_names[idx]) if idx < len(tags_names) else None
                d = dn_map.get(dn_names[idx]) if idx < len(dn_names) else None
            if t is None or d is None:
                continue
            if op == "gt" and t > d:
                out += 1
            elif op == "lt" and t < d:
                out += 1
        return out

    return {
        "num_views": len(names),
        "psnr_tags_better": per_metric(tags_psnr, dn_psnr, "gt"),
        "ssim_tags_better": per_metric(tags_ssim, dn_ssim, "gt"),
        "lpips_tags_better": per_metric(tags_lpips, dn_lpips, "lt"),
        "view_alignment": alignment,
    }


def compare_metrics(tags, dn):
    if not tags or not dn:
        return None
    out = {
        "psnr": float(tags["PSNR"]) - float(dn["PSNR"]),
        "ssim": float(tags["SSIM"]) - float(dn["SSIM"]),
        "lpips": float(tags["LPIPS"]) - float(dn["LPIPS"]),
    }
    if tags.get("num_points") and dn.get("num_points"):
        out["num_points_ratio"] = float(tags["num_points"]) / float(dn["num_points"])
    return out


def pack_side(method, metrics, ply_path):
    if metrics is None:
        return {"method": method, "status": "missing"}
    points = count_ply_vertices(ply_path) if ply_path else None
    return {
        "method": method,
        "status": "scored",
        "psnr": metrics.get("PSNR"),
        "ssim": metrics.get("SSIM"),
        "lpips": metrics.get("LPIPS"),
        "num_points": points,
        "ply": str(ply_path) if ply_path else None,
    }


def run_metrics(python_bin, model_dirs):
    cmd = [python_bin, str(ROOT / "metrics.py"), "-m", *[str(path) for path in model_dirs]]
    completed = subprocess.run(cmd, check=False, cwd=str(ROOT))
    return completed.returncode


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def write_csv(path, tags, dn, delta):
    fields = ["method", "psnr", "ssim", "lpips", "num_points", "delta_psnr_vs_dn", "delta_ssim_vs_dn", "delta_lpips_vs_dn"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "method": "TAGS lpips_ms_w005",
                "psnr": tags.get("psnr"),
                "ssim": tags.get("ssim"),
                "lpips": tags.get("lpips"),
                "num_points": tags.get("num_points"),
                "delta_psnr_vs_dn": None if not delta else delta.get("psnr"),
                "delta_ssim_vs_dn": None if not delta else delta.get("ssim"),
                "delta_lpips_vs_dn": None if not delta else delta.get("lpips"),
            }
        )
        writer.writerow(
            {
                "method": "DN-Splatter",
                "psnr": dn.get("psnr"),
                "ssim": dn.get("ssim"),
                "lpips": dn.get("lpips"),
                "num_points": dn.get("num_points"),
            }
        )


# =====================================================================
# v2 增强: 5-seed 聚合 + wandb 交叉 (从 scripts.exp4_summarize_v2 共享导入)
# =====================================================================
import sys as _sys  # noqa: E402
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from scripts.exp4_summarize_v2 import (  # noqa: E402
    WINNER_GATE,
    _try_wandb,
    discover_seed_dirs,
    parse_seed_from_dir,
    collect_seed_metrics,
    fetch_wandb_runs,
    cross_check_ledger_vs_wandb,
    evaluate_winner_gate,
    write_multiseed_csv,
    multiseed_summary_text,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags-dir", default="output/bicycle_e4_lpips_ms_w005")
    parser.add_argument("--dn-dir", default="output/bicycle_dn_splatter")
    parser.add_argument("--dn-renders", default="output/bicycle_dn_renders/eval")
    parser.add_argument("--json-out", default="output/bicycle_e4_vs_dn_summary.json")
    parser.add_argument("--csv-out", default="output/bicycle_e4_vs_dn_summary.csv")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--require-both", action="store_true")
    # v2 新增
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "tags-ablation"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP", ""))
    parser.add_argument("--scene", default="bicycle", help="v2 5-seed 聚合用, bicycle/vr_room")
    parser.add_argument("--tags-group", default="lpips_ms_w005", help="v2 5-seed: TAGS 实验组 ID")
    parser.add_argument("--method-name", default="ours_30000", help="results.json 里的 method key")
    parser.add_argument("--no-wandb", action="store_true", help="跳过 wandb 拉取")
    parser.add_argument("--multiseed-csv", default="output/bicycle_e4_vs_dn_multiseed.csv",
                        help="v2 输出: 5-seed per-seed CSV")
    parser.add_argument("--tags-dir-sibling-glob", default=None,
                        help="v2: 额外扫描的同级 seed 目录 glob "
                             "(默认 'bicycle_e4_lpips_ms_w005_seed*'). "
                             "设为空串 '' 关闭自动 sibling 发现.")
    args = parser.parse_args()
    args.tags_dir = str(resolve_path(args.tags_dir))
    args.dn_dir = str(resolve_path(args.dn_dir))
    args.dn_renders = str(resolve_path(args.dn_renders))
    args.json_out = str(resolve_path(args.json_out))
    args.csv_out = str(resolve_path(args.csv_out))

    dn_method_dir = Path(args.dn_dir) / "test" / "dn-splatter"
    copied = stage_dn_renders(args.dn_renders, dn_method_dir)
    score_dirs = []
    tags_test = Path(args.tags_dir) / "test"
    if tags_test.is_dir():
        score_dirs.append(args.tags_dir)
    if copied or (dn_method_dir / "renders").is_dir():
        score_dirs.append(args.dn_dir)
    score_status = "skipped"
    if score_dirs and not args.no_score:
        code = run_metrics(args.python_bin, score_dirs)
        score_status = "ok" if code == 0 else f"metrics_exit_{code}"

    tags_ply = find_ply(args.tags_dir)
    dn_ply = find_ply(args.dn_dir)
    tags_method, tags_metrics = read_metrics(args.tags_dir, ply_path=tags_ply)
    dn_method, dn_metrics = read_metrics(args.dn_dir, ply_path=dn_ply)
    tags_row = pack_side(tags_method, tags_metrics, tags_ply)
    dn_row = pack_side(dn_method, dn_metrics, dn_ply)
    tags_for_delta = None if tags_metrics is None else dict(tags_metrics)
    dn_for_delta = None if dn_metrics is None else dict(dn_metrics)
    if tags_for_delta is not None:
        tags_for_delta["num_points"] = tags_row.get("num_points")
    if dn_for_delta is not None:
        dn_for_delta["num_points"] = dn_row.get("num_points")
    delta = compare_metrics(tags_for_delta, dn_for_delta)
    tags_pv = read_per_view(args.tags_dir, tags_method, ply_path=tags_ply)
    dn_pv = read_per_view(args.dn_dir, dn_method, ply_path=dn_ply)
    wins = per_view_wins(tags_pv, dn_pv)

    summary = {
        "protocol": {
            "scene": "bicycle",
            "tags_profile": "lpips_ms_w005",
            "images": "images_4",
            "tags_resolution": 1,
            "hold": 8,
            "dn_depth_loss": "PearsonDepth",
            "metrics": "metrics.py VGG LPIPS",
            "note": (
                "Delta is TAGS minus DN. Positive PSNR/SSIM favors TAGS; "
                "positive LPIPS favors DN. Both train at images_4 native size (1237x822)."
            ),
        },
        "score_status": score_status,
        "dn_views_staged": copied,
        "tags": tags_row,
        "dn": dn_row,
        "delta_tags_minus_dn": delta,
        "per_view_tags_better": wins,
    }

    # v2: 5-seed 聚合 + wandb 交叉对比
    # 单 run 模式下自动发现同级 <dir>_seed{N} (bicycle e4 winner 历史用法)
    tags_root = Path(args.tags_dir)
    if args.tags_dir_sibling_glob is None:
        if (tags_root / "test").is_dir() or any(tags_root.rglob("point_cloud.ply")):
            sibling_glob = str(tags_root.name) + "_seed*"
        # else: 保持 None (多 scene / exp4_runner 路径无需 sibling)
    else:
        sibling_glob = args.tags_dir_sibling_glob or None
    print("\n[v2] 5-seed aggregation + wandb cross-check")
    json_out = Path(args.json_out)  # 提前绑定, 给 v2 块用
    csv_out = Path(args.csv_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    tags_local = collect_seed_metrics(tags_root, method_name=args.method_name,
                                      sibling_glob=sibling_glob)
    print(f"[v2] Found {tags_local['n_seeds']} local seed(s) under {args.tags_dir}")
    if tags_local.get("mean"):
        m = tags_local["mean"]; s = tags_local["std"]
        print(f"[v2] TAGS 5-seed mean: PSNR={m.get('psnr', 0):.4f}±{s.get('psnr', 0):.4f}"
              f" SSIM={m.get('ssim', 0):.5f}±{s.get('ssim', 0):.5f}"
              f" LPIPS={m.get('lpips', 0):.5f}±{s.get('lpips', 0):.5f}")

    # 拉 wandb (best-effort)
    wandb_runs = []
    if not args.no_wandb:
        try:
            wandb_runs = fetch_wandb_runs(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group or args.tags_group,
                scene=args.scene,
            )
            print(f"[v2] wandb: {len(wandb_runs)} run(s) for {args.wandb_entity}/{args.wandb_project} group={args.wandb_group}")
        except Exception as exc:
            print(f"[v2] wandb fetch failed (degraded to local only): {exc!r}")
    else:
        print("[v2] wandb skipped (--no-wandb)")

    # 交叉对比
    cross = cross_check_ledger_vs_wandb(tags_local, wandb_runs)
    print(f"[v2] cross-check: local_n={cross['local_n']} wandb_n={cross['wandb_n']} consistent={cross['consistent']}")
    for note in cross.get("notes", []):
        print(f"[v2]   note: {note}")

    # winner gate
    gate = evaluate_winner_gate(tags_local, dn_row)
    print(f"[v2] winner gate: passed={gate.get('passed')} (Δpsnr={gate.get('psnr_delta', 0):+.3f}, Δlpips={gate.get('lpips_delta', 0):+.4f})")

    # 写 5-seed CSV
    multiseed_csv = Path(args.multiseed_csv)
    multiseed_csv.parent.mkdir(parents=True, exist_ok=True)
    write_multiseed_csv(
        multiseed_csv,
        scene=args.scene,
        tags_group=args.tags_group,
        tags_local=tags_local,
        dn_row=dn_row,
        gate=gate,
        wandb_runs=wandb_runs,
    )
    print(f"[v2] Wrote {multiseed_csv}")

    # 写交叉对比 json (与主 json 分开, 便于单独 review)
    cross_json = json_out.with_name(json_out.stem + "_v2_cross.json")
    cross_payload = {
        "tags_5seed": tags_local,
        "dn_baseline": dn_row,
        "wandb_runs": wandb_runs,
        "cross_check": cross,
        "winner_gate": gate,
    }
    cross_json.write_text(json.dumps(cross_payload, indent=2), encoding="utf-8")
    print(f"[v2] Wrote {cross_json}")

    # 把 v2 数据塞回主 summary, 一次打印完整
    summary["v2_multiseed"] = {
        "tags_5seed_mean": tags_local.get("mean"),
        "tags_5seed_std": tags_local.get("std"),
        "n_seeds": tags_local.get("n_seeds"),
        "winner_gate": gate,
        "cross_check_summary": {
            "consistent": cross.get("consistent"),
            "local_n": cross.get("local_n"),
            "wandb_n": cross.get("wandb_n"),
            "n_notes": len(cross.get("notes", [])),
        },
    }
    # 重新写主 json, 包含 v2 块
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[v2] updated {json_out} with multiseed block")

    json_out = Path(args.json_out)
    csv_out = Path(args.csv_out)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {json_out} and {csv_out}")
    if isinstance(score_status, str) and score_status.startswith("metrics_exit"):
        raise SystemExit(1)
    if args.require_both and (tags_row.get("status") != "scored" or dn_row.get("status") != "scored"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
