#!/usr/bin/env python3
"""通用 Depth Anything V2 + 法线 + depth_params 预生成脚本 (v2 高可靠性版)

针对 v2 计划的 4 个代表性场景: bicycle (已有), garden, bonsai, room
- bicycle 跳过, 直接走 check 模式
- garden/bonsai/room: 走完整的 export + check
- 复用 export_bicycle_mono_depth.py 的 inverse depth 标定逻辑
- 复用 export_vr_room_mono_depth.py 的 transforms.json 改写逻辑

依赖:
  - data/{scene}/images_4/ (输入)
  - data/{scene}/depth_any_out/ (Depth Anything V2 输出 png, 单位 uint8 inverse depth)
  - data/{scene}/normals_from_pretrain/ (法线 png, 由 normal_from_depth.py 或 DA3 输出)
  - data/{scene}/sparse/0/ (COLMAP 重建)
  - utils/make_depth_scale.py (depth_params.json 标定)

用法:
  # 1) 跑 Depth Anything V2 推理 (4 场景, 需 GPU)
  python scripts/prepare_extra_scenes.py --stage depth --scenes garden,bonsai,room \\
      --depth-anything-cmd "python Depth-Anything-V2/run.py ..."

  # 2) 跑法线估计
  python scripts/prepare_extra_scenes.py --stage normal --scenes garden,bonsai,room \\
      --normal-cmd "python Depth-Anything-3/export_depths_for_tags.py ..."

  # 3) 标定 depth_params.json
  python scripts/prepare_extra_scenes.py --stage params --scenes garden,bonsai,room

  # 4) 导出 mono_depth/*_aligned.npy + 改写 transforms.json
  python scripts/prepare_extra_scenes.py --stage export --scenes garden,bonsai,room

  # 5) 全部场景快速自检 (bicycle + garden + bonsai + room)
  python scripts/prepare_extra_scenes.py --stage check --scenes bicycle,garden,bonsai,room
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

# 直接复用 export_bicycle_mono_depth 的核心函数
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_bicycle_mono_depth import (  # type: ignore  # noqa: E402
    export_scene as _bicycle_export_scene,
    check_scene as _bicycle_check_scene,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

# v2 计划的 4 个场景
V2_SCENES = ["bicycle", "garden", "bonsai", "room"]

# bicycle 已经预处理过, 其余 3 个需要 export
SCENES_NEEDING_EXPORT = {"garden", "bonsai", "room"}


def run_cmd(cmd: List[str], cwd: Path = None) -> int:
    """运行外部命令, 出错抛出."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit {result.returncode}: {' '.join(cmd)}")
    return result.returncode


def stage_depth(scenes: List[str], depth_cmd: str, hold: int):
    """Step 1: 跑 Depth Anything V2 推理 → depth_any_out/."""
    for scene in scenes:
        if scene not in SCENES_NEEDING_EXPORT:
            print(f"[depth] skip {scene} (already prepared)")
            continue
        scene_dir = DATA_ROOT / scene
        images_dir = scene_dir / "images_4"
        out_dir = scene_dir / "depth_any_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        # 替换占位符
        cmd_parts = depth_cmd.format(
            scene=scene,
            images=str(images_dir),
            output=str(out_dir),
            hold=hold,
        ).split()
        print(f"[depth] {scene} -> {out_dir}")
        run_cmd(cmd_parts, cwd=REPO_ROOT)


def stage_normal(scenes: List[str], normal_cmd: str, hold: int):
    """Step 2: 跑法线估计 → normals_from_pretrain/."""
    for scene in scenes:
        if scene not in SCENES_NEEDING_EXPORT:
            print(f"[normal] skip {scene} (already prepared)")
            continue
        scene_dir = DATA_ROOT / scene
        out_dir = scene_dir / "normals_from_pretrain"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd_parts = normal_cmd.format(
            scene=scene,
            depth=str(scene_dir / "depth_any_out"),
            images=str(scene_dir / "images_4"),
            output=str(out_dir),
            hold=hold,
        ).split()
        print(f"[normal] {scene} -> {out_dir}")
        run_cmd(cmd_parts, cwd=REPO_ROOT)


def stage_params(scenes: List[str], hold: int):
    """Step 3: 标定 depth_params.json (scale/offset 拟合)."""
    for scene in scenes:
        if scene not in SCENES_NEEDING_EXPORT:
            print(f"[params] skip {scene} (already prepared)")
            continue
        scene_dir = DATA_ROOT / scene
        # utils/make_depth_scale.py 假设 base_dir 是 scene 根目录，
        # 自己拼 sparse/0/ 和 sparse/0/depth_params.json
        cmd = [
            sys.executable, "utils/make_depth_scale.py",
            "--base_dir", str(scene_dir),
            "--depths_dir", str(scene_dir / "depth_any_out"),
        ]
        print(f"[params] {scene}")
        run_cmd(cmd, cwd=REPO_ROOT)


def stage_export(scenes: List[str], hold: int, overwrite: bool, rewrite: bool):
    """Step 4: 导出 mono_depth/*_aligned.npy + 改写 transforms.json."""
    for scene in scenes:
        scene_dir = DATA_ROOT / scene
        print(f"[export] {scene}")
        manifest = _bicycle_export_scene(
            data_root=str(scene_dir),
            images_dirname="images_4",
            depths_dirname="depth_any_out",
            depth_params_path="sparse/0/depth_params.json",
            out_dirname="mono_depth",
            hold=hold,
            overwrite=overwrite,
            rewrite=rewrite,
        )
        # 不输出 holdout_names 字段 (太大)
        summary = {k: v for k, v in manifest.items() if k != "holdout_names"}
        print(json.dumps(summary, indent=2))


def stage_check(scenes: List[str], hold: int) -> bool:
    """Step 5: 4 场景自检, 返回是否全部通过."""
    all_ok = True
    for scene in scenes:
        scene_dir = DATA_ROOT / scene
        print(f"[check] {scene}")
        try:
            report = _bicycle_check_scene(
                data_root=str(scene_dir),
                images_dirname="images_4",
                out_dirname="mono_depth",
                hold=hold,
            )
            print(json.dumps(report, indent=2))
            ok = bool(report.get("aligned"))
            if not ok:
                all_ok = False
        except Exception as exc:
            print(f"  FAILED: {exc!r}")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="v2 通用预处理 (Depth Anything V2 + 法线 + depth_params + mono_depth + transforms)"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["depth", "normal", "params", "export", "check", "all"],
        help="执行阶段; all = depth -> normal -> params -> export -> check",
    )
    parser.add_argument(
        "--scenes",
        default="garden,bonsai,room",
        help="要处理的场景, 逗号分隔. 默认 garden,bonsai,room (bicycle 已有, 跳过)",
    )
    parser.add_argument("--hold", type=int, default=8)
    parser.add_argument(
        "--depth-anything-cmd",
        default=(
            "python Depth-Anything-V2/run.py "
            "--encoder vitl --img-path {images} --outdir {output} --input-size 518"
        ),
        help="Depth Anything V2 推理命令, 用 {images} {output} {scene} 占位",
    )
    parser.add_argument(
        "--normal-cmd",
        default=(
            "python Depth-Anything-3/export_depths_for_tags.py "
            "--images {images} --depths {depth} --output {output}"
        ),
        help="法线估计命令, 用 {images} {depth} {output} 占位",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-rewrite-transforms", action="store_true")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    unknown = [s for s in scenes if s not in V2_SCENES]
    if unknown:
        raise SystemExit(f"Unknown scenes (v2 only supports {V2_SCENES}): {unknown}")

    rewrite = not args.no_rewrite_transforms

    if args.stage == "all":
        stage_depth(scenes, args.depth_anything_cmd, args.hold)
        stage_normal(scenes, args.normal_cmd, args.hold)
        stage_params(scenes, args.hold)
        stage_export(scenes, args.hold, args.overwrite, rewrite)
        ok = stage_check(scenes, args.hold)
        if not ok:
            raise SystemExit("check stage failed; preprocess incomplete")
    elif args.stage == "depth":
        stage_depth(scenes, args.depth_anything_cmd, args.hold)
    elif args.stage == "normal":
        stage_normal(scenes, args.normal_cmd, args.hold)
    elif args.stage == "params":
        stage_params(scenes, args.hold)
    elif args.stage == "export":
        stage_export(scenes, args.hold, args.overwrite, rewrite)
    elif args.stage == "check":
        # check 阶段包含所有指定场景, 包括 bicycle
        check_scenes = scenes if "bicycle" in scenes else ["bicycle"] + scenes
        ok = stage_check(check_scenes, args.hold)
        if not ok:
            raise SystemExit("check failed")
    print(f"[prepare_extra_scenes] {args.stage} done for {scenes}")


if __name__ == "__main__":
    main()
