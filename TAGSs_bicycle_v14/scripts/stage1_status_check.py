#!/usr/bin/env python3
"""阶段 1 状态盘点 + Stage 1 第一步可执行命令生成器 (零依赖)

不依赖 numpy/torch, 仅用标准库.
- 扫描 4 场景的预处理状态
- 输出"立即可执行"的 PowerShell 命令清单
- 写到 output/exp4_stats/stage1_status.json
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
STATS_ROOT = REPO_ROOT / "output" / "exp4_stats"

SCENES = ["bicycle", "garden", "bonsai", "room"]
EXPECTED_DIRS = ["images", "images_2", "images_4", "sparse"]
PREPROCESS_DIRS = ["depth_any_out", "normals_from_pretrain", "mono_depth"]


def count_files(p: Path) -> int:
    if not p.is_dir():
        return 0
    return sum(1 for _ in p.iterdir() if _.is_file())


def check_scene(scene: str) -> dict:
    scene_dir = DATA_ROOT / scene
    info = {"scene": scene, "scene_dir_exists": scene_dir.is_dir()}
    if not info["scene_dir_exists"]:
        return info

    for d in EXPECTED_DIRS:
        p = scene_dir / d
        info[f"has_{d}"] = p.is_dir()

    for d in PREPROCESS_DIRS:
        p = scene_dir / d
        info[f"has_{d}"] = p.is_dir()
        info[f"n_{d}"] = count_files(p) if p.is_dir() else 0

    params = scene_dir / "sparse" / "0" / "depth_params.json"
    info["has_depth_params"] = params.is_file()

    transforms = scene_dir / "transforms.json"
    if transforms.is_file():
        try:
            with transforms.open("r", encoding="utf-8") as f:
                content = f.read()
            info["transforms_rewritten"] = "mono_depth_path" in content
        except Exception:
            info["transforms_rewritten"] = False
    else:
        info["transforms_rewritten"] = False

    return info


def main():
    stats = {
        "timestamp": datetime.now().isoformat(),
        "scenes": [check_scene(s) for s in SCENES],
    }

    # 判断每个场景处于哪一阶段
    for sc in stats["scenes"]:
        if not sc.get("scene_dir_exists"):
            sc["status"] = "MISSING_DATA"
            sc["next_step"] = "下载 Mip-NeRF 360 数据"
        elif not sc.get("has_depth_any_out"):
            sc["status"] = "STAGE_1_PENDING"
            sc["next_step"] = "python scripts/prepare_extra_scenes.py --stage depth --scenes " + sc["scene"]
        elif not sc.get("has_normals_from_pretrain"):
            sc["status"] = "DEPTH_DONE_NEED_NORMAL"
            sc["next_step"] = "python scripts/prepare_extra_scenes.py --stage normal --scenes " + sc["scene"]
        elif not sc.get("has_depth_params"):
            sc["status"] = "NORMAL_DONE_NEED_PARAMS"
            sc["next_step"] = "python scripts/prepare_extra_scenes.py --stage params --scenes " + sc["scene"]
        elif not sc.get("has_mono_depth"):
            sc["status"] = "PARAMS_DONE_NEED_EXPORT"
            sc["next_step"] = "python scripts/prepare_extra_scenes.py --stage export --scenes " + sc["scene"]
        else:
            sc["status"] = "STAGE_1_DONE"
            sc["next_step"] = "(无需操作)"

    # 找需要 export 的场景
    pending = [sc["scene"] for sc in stats["scenes"]
               if sc["status"] in ("STAGE_1_PENDING", "DEPTH_DONE_NEED_NORMAL",
                                    "NORMAL_DONE_NEED_PARAMS", "PARAMS_DONE_NEED_EXPORT")]
    stats["scenes_pending_stage1"] = pending
    stats["scenes_done_stage1"] = [sc["scene"] for sc in stats["scenes"] if sc["status"] == "STAGE_1_DONE"]

    STATS_ROOT.mkdir(parents=True, exist_ok=True)
    out = STATS_ROOT / "stage1_status.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stage1_status] Wrote {out}")

    # 打印人类可读摘要
    print("\n" + "=" * 60)
    print(f"[v3 Stage 1 状态盘点] {stats['timestamp']}")
    print("=" * 60)
    for sc in stats["scenes"]:
        print(f"\n  {sc['scene']:>10s}: {sc['status']}")
        if sc.get("scene_dir_exists"):
            for d in EXPECTED_DIRS:
                mark = "OK" if sc.get(f"has_{d}") else "MISS"
                print(f"    {d:>10s}: {mark}")
            for d in PREPROCESS_DIRS:
                n = sc.get(f"n_{d}", 0)
                mark = f"OK ({n} files)" if sc.get(f"has_{d}") and n > 0 else "MISS"
                print(f"    {d:>10s}: {mark}")
            print(f"    depth_params: {'OK' if sc.get('has_depth_params') else 'MISS'}")
            print(f"    transforms.json: {'rewritten' if sc.get('transforms_rewritten') else 'not rewritten'}")
        print(f"    next_step: {sc.get('next_step', '(无)')}")

    print("\n" + "=" * 60)
    print(f"  Stage 1 已完成: {stats['scenes_done_stage1']}")
    print(f"  Stage 1 待完成: {pending}")
    print("=" * 60)

    # 输出 v3 计划第一步的"立即可执行"指令
    if pending:
        print("\n[v3 Stage 1 第一步] 立即可执行指令 (PowerShell):\n")
        print("  # 一次性跑完所有 pending 场景的 4 个子步骤")
        print(f'  $env:E4_SCENES = "{",".join(pending)}"')
        print("  python scripts/prepare_extra_scenes.py --stage all --scenes $env:E4_SCENES --hold 8")
        print("  # 或分步:")
        for stage in ["depth", "normal", "params", "export", "check"]:
            print(f"  python scripts/prepare_extra_scenes.py --stage {stage} --scenes $env:E4_SCENES --hold 8")
        print("\n  # 或用 PS1 包装:")
        print("  .\\scripts\\run_prepare_extra_scenes.ps1")
        print("\n  # 跑完后: 状态自检")
        print("  python scripts/prepare_extra_scenes.py --stage check --scenes bicycle,garden,bonsai,room --hold 8")
        print("\n  # 下一步 (Stage 2):")
        print("  .\\scripts\\run_qa_baseline.ps1")
    else:
        print("\n  4 场景 Stage 1 全部完成, 立即可进入 Stage 2:")
        print("  .\\scripts\\run_qa_baseline.ps1")


if __name__ == "__main__":
    main()
