#!/usr/bin/env python3
"""bicycle 1237x822 三方重跑汇总 (official 3DGS / TAGS E4 / TAGS v14).

对应 ``scripts/run_bicycle_rerun_1237.ps1`` 产出的三个 run:
  A  output/bicycle_official_3dgs_1237                 official graphdeco-inria 3DGS (stock)
  B  output/bicycle_e4_lpips_ms_w005_rerun_1237        TAGS E4 winner (lpips_ms_w005)
  C  output/bicycle_v14_v8_monitor_28500_select_1237    TAGS v14 v8 monitor_28500_select

统一协议 (三者完全一致):
  scene      Mip-NeRF 360 bicycle
  images     images_4 (1237x822 原生), --resolution 1
  split      --eval, LLFF hold=8 -> 25 test views / 169 train views
  iterations 30000, seed 0
  metrics    仓库 metrics.py, VGG LPIPS (A 用 third_party 版, B/C 用仓库版, 两者 LPIPS 实现一致)

同时纳入历史 reference (只读, 便于核对可复现性):
  - bicycle_e4_lpips_ms_w005[_seedN]  : 2026-09 跑出的 E4 4-seed 结果 (25.06 dB)
  - bicycle_dn_splatter                : DN-Splatter baseline (22.90 dB)

输出:
  output/bicycle_rerun_1237_summary.json
  output/bicycle_rerun_1237_summary.csv
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    {
        "key": "official_3dgs",
        "label": "official 3DGS (graphdeco-inria, stock)",
        "dir": "bicycle_official_3dgs_1237",
        "method": "ours_30000",
        "ply": "point_cloud/iteration_30000/point_cloud.ply",
    },
    {
        "key": "tags_e4",
        "label": "TAGS E4 (lpips_ms_w005)",
        "dir": "bicycle_e4_lpips_ms_w005_rerun_1237",
        "method": "ours_30000",
        "ply": "point_cloud/iteration_30000/point_cloud.ply",
    },
    {
        "key": "tags_v14",
        "label": "TAGS v14 (v8 monitor_28500_select)",
        "dir": "bicycle_v14_v8_monitor_28500_select_1237",
        "method": "ours_30000",
        "ply": "point_cloud/iteration_30000/point_cloud.ply",
    },
]

# 历史 reference (2026-09/10 已跑完, 用于可复现性核对)
REF_E4_BASE = "bicycle_e4_lpips_ms_w005"
REF_DN_DIR = "bicycle_dn_splatter"
REF_DN_METHOD = "dn-splatter"


def read_block(model_dir: Path, method: str):
    """读 <model>/test/<method>/results.json, 回退到 <model>/test/results.json。"""
    for cand in (model_dir / "test" / method / "results.json",
                 model_dir / "test" / "results.json",
                 model_dir / "results.json"):
        if not cand.is_file():
            continue
        payload = json.loads(cand.read_text(encoding="utf-8-sig"))
        block = payload.get(method)
        if isinstance(block, dict) and "PSNR" in block:
            return block, cand
        # 有的仓库把 method 块直接放在顶层
        if isinstance(payload, dict) and "PSNR" in payload:
            return payload, cand
    return None, None


def read_per_view(model_dir: Path, method: str):
    for cand in (model_dir / "test" / method / "per_view.json",
                 model_dir / "test" / "per_view.json",
                 model_dir / "per_view.json"):
        if not cand.is_file():
            continue
        payload = json.loads(cand.read_text(encoding="utf-8-sig"))
        block = payload.get(method)
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


def stem_map(metric_dict):
    if not isinstance(metric_dict, dict):
        return {}
    out = {}
    for name, val in metric_dict.items():
        try:
            out[Path(name).stem] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def per_view_wins(a_block, b_block):
    """按 stem 配对, 统计 a 在多少视角上优于 b。"""
    a_psnr = stem_map((a_block or {}).get("PSNR"))
    b_psnr = stem_map((b_block or {}).get("PSNR"))
    a_ssim = stem_map((a_block or {}).get("SSIM"))
    b_ssim = stem_map((b_block or {}).get("SSIM"))
    a_lp = stem_map((a_block or {}).get("LPIPS"))
    b_lp = stem_map((b_block or {}).get("LPIPS"))
    common = sorted(set(a_psnr) & set(b_psnr))
    if not common:
        return None
    out = {"num_views": len(common), "view_alignment": "by_stem"}
    out["psnr"] = sum(1 for s in common if a_psnr[s] > b_psnr.get(s, -1))
    out["ssim"] = sum(1 for s in common if a_ssim.get(s, -1) > b_ssim.get(s, -1))
    out["lpips"] = sum(1 for s in common if a_lp.get(s, 9) < b_lp.get(s, 9))
    return out


def ms(values):
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    mean = statistics.mean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std


def discover_ref_e4_seeds(output_dir: Path):
    """历史 E4 4-seed: base + base_seedN。"""
    found = {}
    tail = re.compile(r"^" + re.escape(REF_E4_BASE) + r"_seed(?P<n>\d+)$")
    for entry in sorted(output_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == REF_E4_BASE:
            label = "seed0"
        else:
            m = tail.match(entry.name)
            if not m:
                continue
            label = "seed" + m.group("n")
        found[label] = entry
    return dict(sorted(found.items(),
                       key=lambda kv: 0 if kv[0] == "seed0" else int(kv[0][4:])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    output_dir = root / "output"

    json_out = Path(args.json_out) if args.json_out else output_dir / "bicycle_rerun_1237_summary.json"
    csv_out = Path(args.csv_out) if args.csv_out else output_dir / "bicycle_rerun_1237_summary.csv"

    summary = {
        "protocol": {
            "scene": "bicycle",
            "images": "images_4",
            "resolution": "native 1237x822 (--resolution 1)",
            "split": "--eval, LLFF hold=8 -> 25 test views",
            "iterations": 30000,
            "seed": 0,
            "metrics": "metrics.py, VGG LPIPS",
            "note": ("三个 run 同协议同数据同 holdout 划分; "
                     "Delta 列 = TAGS - baseline (PSNR/SSIM 越大越好, LPIPS 越小越好)."),
        },
        "runs": {},
    }

    per_view_blocks = {}

    # ---- 三个重跑 case ----
    for case in CASES:
        mdir = output_dir / case["dir"]
        key = case["key"]
        if not mdir.is_dir():
            summary["runs"][key] = {"label": case["label"], "status": "missing_dir",
                                    "dir": str(mdir)}
            print("[warn] missing dir: " + str(mdir))
            continue
        block, src = read_block(mdir, case["method"])
        if block is None:
            summary["runs"][key] = {"label": case["label"], "status": "not_scored",
                                    "dir": str(mdir)}
            print("[warn] no results.json for " + key + " under " + str(mdir))
            continue
        points = count_ply_vertices(mdir / case["ply"])
        pv = read_per_view(mdir, case["method"])
        per_view_blocks[key] = pv
        summary["runs"][key] = {
            "label": case["label"],
            "status": "scored",
            "dir": str(mdir),
            "source": str(src),
            "psnr": block["PSNR"],
            "ssim": block["SSIM"],
            "lpips": block["LPIPS"],
            "num_points": points,
        }
        print("[ok] %-16s PSNR=%.4f SSIM=%.5f LPIPS=%.5f points=%s"
              % (key, block["PSNR"], block["SSIM"], block["LPIPS"], points))

    # ---- 历史 E4 4-seed reference ----
    ref = {}
    ref_seeds = discover_ref_e4_seeds(output_dir)
    per_seed = {}
    for label, mdir in ref_seeds.items():
        block, _ = read_block(mdir, "ours_30000")
        if block is None:
            continue
        per_seed[label] = {"psnr": block["PSNR"], "ssim": block["SSIM"], "lpips": block["LPIPS"]}
    if per_seed:
        p_m, p_s = ms([v["psnr"] for v in per_seed.values()])
        s_m, s_s = ms([v["ssim"] for v in per_seed.values()])
        l_m, l_s = ms([v["lpips"] for v in per_seed.values()])
        ref["e4_historical_4seed"] = {
            "n_seeds": len(per_seed),
            "per_seed": per_seed,
            "psnr_mean": p_m, "psnr_std": p_s,
            "ssim_mean": s_m, "ssim_std": s_s,
            "lpips_mean": l_m, "lpips_std": l_s,
        }
        print("[ref] historical E4 4-seed: PSNR=%.4f±%.4f" % (p_m, p_s))

    # ---- DN-Splatter reference ----
    ddir = output_dir / REF_DN_DIR
    dblock, dsrc = read_block(ddir, REF_DN_METHOD)
    if dblock is not None:
        ref["dn_splatter"] = {
            "psnr": dblock["PSNR"], "ssim": dblock["SSIM"], "lpips": dblock["LPIPS"],
            "num_points": count_ply_vertices(ddir / "splat.ply"),
            "source": str(dsrc),
        }

    summary["reference"] = ref

    # ---- delta: 各 TAGS run vs official 3DGS ----
    base = summary["runs"].get("official_3dgs", {})
    if base.get("status") == "scored":
        deltas = {}
        for key in ("tags_e4", "tags_v14"):
            run = summary["runs"].get(key, {})
            if run.get("status") != "scored":
                continue
            deltas[key] = {
                "psnr": run["psnr"] - base["psnr"],
                "ssim": run["ssim"] - base["ssim"],
                "lpips": run["lpips"] - base["lpips"],
                "points_ratio": (run["num_points"] / base["num_points"])
                if base.get("num_points") and run.get("num_points") else None,
            }
        summary["delta_vs_official_3dgs"] = deltas
    else:
        summary["delta_vs_official_3dgs"] = {
            "status": "official_3dgs_not_run",
            "note": ("Case A (official graphdeco-inria 3DGS) is opt-in via "
                     "RERUN_OFFICIAL=1; baseline numbers live in "
                     "scripts/summarize_bicycle_3way.py / output/bicycle_3way_summary.json."),
        }

    # ---- delta vs DN-Splatter baseline (always available) ----
    dn = ref.get("dn_splatter")
    if dn:
        dn_deltas = {}
        for key in ("tags_e4", "tags_v14"):
            run = summary["runs"].get(key, {})
            if run.get("status") != "scored":
                continue
            dn_deltas[key] = {
                "psnr": run["psnr"] - dn["psnr"],
                "ssim": run["ssim"] - dn["ssim"],
                "lpips": run["lpips"] - dn["lpips"],
            }
        summary["delta_vs_dn_splatter"] = dn_deltas

    # ---- delta vs historical E4 4-seed mean (reproducibility) ----
    hist = ref.get("e4_historical_4seed")
    if hist and hist.get("psnr_mean") is not None:
        hist_deltas = {}
        for key in ("tags_e4", "tags_v14"):
            run = summary["runs"].get(key, {})
            if run.get("status") != "scored":
                continue
            hist_deltas[key] = {
                "psnr": run["psnr"] - hist["psnr_mean"],
                "ssim": run["ssim"] - hist["ssim_mean"],
                "lpips": run["lpips"] - hist["lpips_mean"],
            }
        summary["delta_vs_historical_e4_4seed_mean"] = hist_deltas

    # ---- per-view 胜场 ----
    pv_summary = {}
    if "tags_e4" in per_view_blocks and "official_3dgs" in per_view_blocks:
        pv_summary["tags_e4_vs_official_3dgs"] = per_view_wins(
            per_view_blocks["tags_e4"], per_view_blocks["official_3dgs"])
    if "tags_v14" in per_view_blocks and "official_3dgs" in per_view_blocks:
        pv_summary["tags_v14_vs_official_3dgs"] = per_view_wins(
            per_view_blocks["tags_v14"], per_view_blocks["official_3dgs"])
    if "tags_e4" in per_view_blocks and "tags_v14" in per_view_blocks:
        pv_summary["tags_e4_vs_tags_v14"] = per_view_wins(
            per_view_blocks["tags_e4"], per_view_blocks["tags_v14"])
    if pv_summary:
        summary["per_view_wins"] = pv_summary

    # ---- 可复现性核对 (重跑 E4 vs 历史 E4 seed0) ----
    e4_run = summary["runs"].get("tags_e4", {})
    ref_seed0 = (ref.get("e4_historical_4seed") or {}).get("per_seed", {}).get("seed0")
    if e4_run.get("status") == "scored" and ref_seed0:
        summary["reproducibility_e4"] = {
            "rerun_psnr": e4_run["psnr"], "historical_seed0_psnr": ref_seed0["psnr"],
            "delta_psnr": e4_run["psnr"] - ref_seed0["psnr"],
            "rerun_ssim": e4_run["ssim"], "historical_seed0_ssim": ref_seed0["ssim"],
            "delta_ssim": e4_run["ssim"] - ref_seed0["ssim"],
            "rerun_lpips": e4_run["lpips"], "historical_seed0_lpips": ref_seed0["lpips"],
            "delta_lpips": e4_run["lpips"] - ref_seed0["lpips"],
        }

    # ---- 写盘 ----
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "status", "PSNR", "SSIM", "LPIPS", "num_points"])
        for case in CASES:
            run = summary["runs"].get(case["key"], {})
            if run.get("status") == "scored":
                w.writerow([case["label"], "scored",
                            "%.4f" % run["psnr"], "%.5f" % run["ssim"], "%.5f" % run["lpips"],
                            "%d" % (run.get("num_points") or 0)])
            else:
                w.writerow([case["label"], run.get("status", "missing"), "", "", "", ""])
        hist = ref.get("e4_historical_4seed")
        if hist:
            w.writerow(["[ref] historical E4 (4-seed)", "scored",
                        "%.4f +- %.4f" % (hist["psnr_mean"], hist["psnr_std"]),
                        "%.5f +- %.5f" % (hist["ssim_mean"], hist["ssim_std"]),
                        "%.5f +- %.5f" % (hist["lpips_mean"], hist["lpips_std"]), ""])
        dn = ref.get("dn_splatter")
        if dn:
            w.writerow(["[ref] DN-Splatter", "scored",
                        "%.4f" % dn["psnr"], "%.5f" % dn["ssim"], "%.5f" % dn["lpips"],
                        "%d" % (dn.get("num_points") or 0)])
        for key, d in summary.get("delta_vs_official_3dgs", {}).items():
            if not isinstance(d, dict) or "psnr" not in d:
                continue
            label = next((c["label"] for c in CASES if c["key"] == key), key)
            w.writerow(["Delta %s - official 3DGS" % label, "",
                        "%+.4f" % d["psnr"], "%+.5f" % d["ssim"], "%+.5f" % d["lpips"],
                        ("%.3fx" % d["points_ratio"]) if d.get("points_ratio") else ""])
        for key, d in summary.get("delta_vs_dn_splatter", {}).items():
            label = next((c["label"] for c in CASES if c["key"] == key), key)
            w.writerow(["Delta %s - DN-Splatter" % label, "",
                        "%+.4f" % d["psnr"], "%+.5f" % d["ssim"], "%+.5f" % d["lpips"], ""])
        for key, d in summary.get("delta_vs_historical_e4_4seed_mean", {}).items():
            label = next((c["label"] for c in CASES if c["key"] == key), key)
            w.writerow(["Delta %s - historical E4 4-seed mean" % label, "",
                        "%+.4f" % d["psnr"], "%+.5f" % d["ssim"], "%+.5f" % d["lpips"], ""])

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nWrote " + str(json_out))
    print("Wrote " + str(csv_out))


if __name__ == "__main__":
    main()
