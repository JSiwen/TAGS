#!/usr/bin/env python3
"""把 A.1 已经跑完的旧数据 (无 wandb 同步) 补 log 一次.

用法:
    set WANDB_ENABLED=1
    set WANDB_PROJECT=TAGSs_exp4v3_A_method_comparison
    set WANDB_ENTITY=jsweni-neu
    python scripts/_backfill_a1_to_wandb.py
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# 模拟 exp4 runner 的路径
out = REPO / "output" / "exp4v3_A_bicycle" / "bicycle" / "A.A1_Raw3DGS"
summary_path = out / "summary.json"
if not summary_path.is_file():
    print(f"[skip] no summary at {summary_path}")
    sys.exit(0)

s = json.loads(summary_path.read_text(encoding="utf-8"))
seed_info = s.get("seeds", {}).get("0", {})
if not seed_info:
    print("[skip] no seed0 in summary")
    sys.exit(0)

print(f"A.1 seed0 metrics: {seed_info}")

# 调 log_run_result (自动 init wandb run)
from scripts.exp4_wandb_logger import log_run_result
ok = log_run_result(
    scene="bicycle",
    group="A.A1_Raw3DGS",
    seed=0,
    metrics={
        "psnr": seed_info.get("psnr"),
        "ssim": seed_info.get("ssim"),
        "lpips": seed_info.get("lpips"),
        "num_points": seed_info.get("num_points"),
        "iterations": seed_info.get("iterations"),
    },
    config={
        "scene": "bicycle",
        "group": "A.A1_Raw3DGS",
        "seed": 0,
        "iterations": seed_info.get("iterations"),
        "images_dir": "images_2",
        "resolution": 2,
        "backfill": True,
    },
)
print(f"wandb log ok: {ok}")
