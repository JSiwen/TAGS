#!/usr/bin/env python3
"""bicycle 三方对比: TAGS E4 winner (4-seed) vs vanilla 3DGS vs DN-Splatter.

三条 baseline 共用同一套 ``metrics.py`` (VGG LPIPS) 打分, 因此
本脚本只做汇总, 不重新评分。

输入 (缺一即退化输出已有部分):
  output/bicycle_e4_lpips_ms_w005[_seedN]/test/ours_30000/results.json   TAGS
  output/vanilla_3dgs_bicycle/test/ours_30000/results.json              vanilla 3DGS
  output/bicycle_dn_splatter/test/dn-splatter/results.json              DN-Splatter

输出:
  output/bicycle_3way_summary.json
  output/bicycle_3way_summary.csv
  output/bicycle_3way_multiseed.csv

协议说明:
  - 全部在 ``images_4`` (1237x822) 原生尺寸训练, 30k iter, ``--eval --llffhold 8``
    (vanilla 无 ``--llffhold``, 走它自己的 default LLFF hold=8, 实测同为 25 test views)
  - TAGS / vanilla 视角文件名同为 ``00000.png..00024.png`` (COLMAP 排序后),
    可按 stem 直接配对; DN 是原始 ``_DSC####.JPG``, 用 sorted-index 配对
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TAGS_BASE = "bicycle_e4_lpips_ms_w005"
TAGS_KEY = "ours_30000"
VANILLA_DIR = "vanilla_3dgs_bicycle"
VANILLA_KEY = "ours_30000"
DN_DIR = "bicycle_dn_splatter"
DN_KEY = "dn-splatter"


def read_block(model_dir: Path, key: str):
    """优先 ``<model>/test/<key>/results.json``, 回退顶层 results.json."""
    for cand in (model_dir / "test" / key / "results.json",
                 model_dir / "test" / "results.json",
                 model_dir / "results.json"):
        if cand.is_file():
            payload = json.loads(cand.read_text(encoding="utf-8-sig"))
            block = payload.get(key)
            if isinstance(block, dict) and "PSNR" in block:
                return block, cand
    return None, None


def read_per_view(model_dir: Path, key: str):
    for cand in (model_dir / "test" / key / "per_view.json",
                 model_dir / "test" / "per_view.json",
                 model_dir / "per_view.json"):
        if cand.is_file():
            payload = json.loads(cand.read_text(encoding="utf-8-sig"))
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


def discover_tags_seeds(output_dir: Path):
    """``{label: dir}``, label = seed0 / seedN, 按数字序。"""
    found = {}
    tail_re = re.compile(r"^" + re.escape(TAGS_BASE) + r"_seed(?P<n>\d+)$")
    for entry in sorted(output_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == TAGS_BASE:
            label = "seed0"
        else:
            m = tail_re.match(entry.name)
            if not m:
                continue
            label = "seed" + m.group("n")
        if (entry / "point_cloud" / "iteration_30000" / "point_cloud.ply").is_file():
            found[label] = entry
    return dict(sorted(found.items(),
                       key=lambda kv: 0 if kv[0] == "seed0" else int(kv[0][4:])))


def stem_metric(block, name):
    sub = block.get(name) if isinstance(block, dict) else None
    if not isinstance(sub, dict):
        return {}
    out = {}
    for fname, val in sub.items():
        try:
            out[Path(fname).stem] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def ms(values):
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    mean = statistics.mean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    output_dir = root / "output"

    json_out = Path(args.json_out) if args.json_out else output_dir / "bicycle_3way_summary.json"
    csv_out = Path(args.csv_out) if args.csv_out else output_dir / "bicycle_3way_summary.csv"

    summary = {
        "protocol": {
            "scene": "bicycle",
            "images": "images_4",
            "resolution": "native 1237x822",
            "hold": 8,
            "iterations": 30000,
            "metrics": "metrics.py VGG LPIPS",
            "note": ("TAGS = 4-seed mean +- std; vanilla 3DGS and DN-Splatter are "
                     "single runs. Delta columns are TAGS minus baseline."),
        }
    }

    # ---- TAGS 4-seed ----
    seeds = discover_tags_seeds(output_dir)
    per_seed, per_seed_points = {}, {}
    for label, mdir in seeds.items():
        block, src = read_block(mdir, TAGS_KEY)
        if block is None:
            print("[warn] no TAGS results for " + label + " at " + str(mdir))
            continue
        per_seed[label] = {"psnr": block["PSNR"], "ssim": block["SSIM"], "lpips": block["LPIPS"]}
        per_seed_points[label] = count_ply_vertices(
            mdir / "point_cloud" / "iteration_30000" / "point_cloud.ply")
    if per_seed:
        p_mean, p_std = ms([v["psnr"] for v in per_seed.values()])
        s_mean, s_std = ms([v["ssim"] for v in per_seed.values()])
        l_mean, l_std = ms([v["lpips"] for v in per_seed.values()])
        pt_mean, _ = ms(list(per_seed_points.values()))
        summary["tags"] = {
            "profile": "lpips_ms_w005",
            "n_seeds": len(per_seed),
            "per_seed": per_seed,
            "per_seed_points": per_seed_points,
            "psnr_mean": p_mean, "psnr_std": p_std,
            "ssim_mean": s_mean, "ssim_std": s_std,
            "lpips_mean": l_mean, "lpips_std": l_std,
            "num_points_mean": pt_mean,
        }
    else:
        print("[warn] no TAGS seeds scored; TAGS column will be omitted")

    # ---- vanilla 3DGS ----
    vdir = output_dir / VANILLA_DIR
    vblock, vsrc = read_block(vdir, VANILLA_KEY)
    if vblock is not None:
        summary["vanilla_3dgs"] = {
            "psnr": vblock["PSNR"], "ssim": vblock["SSIM"], "lpips": vblock["LPIPS"],
            "num_points": count_ply_vertices(vdir / "point_cloud" / "iteration_30000" / "point_cloud.ply"),
            "source": str(vsrc),
            "note": "vanilla graphdeco-inria/gaussian-splatting, defaults + images_4 + eval",
        }
        summary["vanilla_3dgs"]["per_view"] = read_per_view(vdir, VANILLA_KEY)
    else:
        print("[warn] no vanilla 3DGS results under " + str(vdir))

    # ---- DN-Splatter ----
    ddir = output_dir / DN_DIR
    dblock, dsrc = read_block(ddir, DN_KEY)
    if dblock is not None:
        summary["dn_splatter"] = {
            "psnr": dblock["PSNR"], "ssim": dblock["SSIM"], "lpips": dblock["LPIPS"],
            "num_points": count_ply_vertices(ddir / "splat.ply"),
            "source": str(dsrc),
            "depth_loss": "PearsonDepth lambda=0.2",
        }
        summary["dn_splatter"]["per_view"] = read_per_view(ddir, DN_KEY)
    else:
        print("[warn] no DN-Splatter results under " + str(ddir))

    # ---- delta ----
    if "tags" in summary:
        delta = {}
        for name in ("vanilla_3dgs", "dn_splatter"):
            if name in summary:
                delta[name] = {
                    "psnr": summary["tags"]["psnr_mean"] - summary[name]["psnr"],
                    "ssim": summary["tags"]["ssim_mean"] - summary[name]["ssim"],
                    "lpips": summary["tags"]["lpips_mean"] - summary[name]["lpips"],
                    "points_ratio": (
                        (summary["tags"]["num_points_mean"] / summary[name]["num_points"])
                        if summary[name].get("num_points") else None
                    ),
                }
        summary["delta_tags_minus_baseline"] = delta

    # ---- per-view win counts vs vanilla (same stems) ----
    if "tags" in summary and "vanilla_3dgs" in summary:
        vpv = summary["vanilla_3dgs"].pop("per_view", {})
        wins = {"num_views": 0, "psnr": 0, "ssim": 0, "lpips": 0, "view_alignment": None}
        v_psnr = stem_metric(vpv, "PSNR")
        # TAGS seed0 per-view (all seeds share stems)
        tags0_dir = output_dir / TAGS_BASE
        tpv = read_per_view(tags0_dir, TAGS_KEY)
        t_psnr = stem_metric(tpv, "PSNR")
        common = sorted(set(v_psnr) & set(t_psnr))
        if common:
            wins["view_alignment"] = "by_stem"
            wins["num_views"] = len(common)
            for stem in common:
                if t_psnr.get(stem, -1) > v_psnr[stem]:
                    wins["psnr"] += 1
            v_ssim, t_ssim = stem_metric(vpv, "SSIM"), stem_metric(tpv, "SSIM")
            for stem in common:
                if t_ssim.get(stem, -1) > v_ssim.get(stem, -1):
                    wins["ssim"] += 1
            v_lp, t_lp = stem_metric(vpv, "LPIPS"), stem_metric(tpv, "LPIPS")
            for stem in common:
                if t_lp.get(stem, 9) < v_lp.get(stem, 9):
                    wins["lpips"] += 1
            wins["per_view"] = [
                {"view": s,
                 "vanilla_psnr": v_psnr[s], "tags_psnr": t_psnr.get(s),
                 "vanilla_ssim": v_ssim.get(s), "tags_ssim": t_ssim.get(s),
                 "vanilla_lpips": v_lp.get(s), "tags_lpips": t_lp.get(s)}
                for s in common
            ]
        summary["per_view_tags_vs_vanilla"] = wins

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "seeds", "PSNR", "SSIM", "LPIPS", "num_points"])
        if "tags" in summary:
            t = summary["tags"]
            w.writerow([
                "TAGS lpips_ms_w005", str(t["n_seeds"]),
                "%.4f +- %.4f" % (t["psnr_mean"], t["psnr_std"]),
                "%.5f +- %.5f" % (t["ssim_mean"], t["ssim_std"]),
                "%.5f +- %.5f" % (t["lpips_mean"], t["lpips_std"]),
                "%.0f" % (t["num_points_mean"] or 0),
            ])
        for name, label in (("vanilla_3dgs", "vanilla 3DGS"), ("dn_splatter", "DN-Splatter")):
            if name in summary:
                b = summary[name]
                w.writerow([label, "1",
                            "%.4f" % b["psnr"], "%.5f" % b["ssim"], "%.5f" % b["lpips"],
                            "%.0f" % (b.get("num_points") or 0)])
        for name, label in (("vanilla_3dgs", "vanilla 3DGS"), ("dn_splatter", "DN-Splatter")):
            d = summary.get("delta_tags_minus_baseline", {}).get(name)
            if d:
                w.writerow(["Delta TAGS - " + label, "",
                            "%+.4f" % d["psnr"], "%+.5f" % d["ssim"], "%+.5f" % d["lpips"],
                            ("%.3fx" % d["points_ratio"]) if d.get("points_ratio") else ""])

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nWrote " + str(json_out))
    print("Wrote " + str(csv_out))


if __name__ == "__main__":
    main()