"""Pack the bicycle v14 experiment into a portable .zip for migration.

Scope (v14 only):
  - v14 launch/summarize scripts + the shared pipeline code they call
  - v14 documentation / summary artifacts
  - data/bicycle/depth_any_out/  (194 uint8 inverse-depth PNGs, what --depths points at)
  - data/bicycle/images_4/       (194 RGB PNGs, 1237x822, the 1237 re-run protocol)
  - data/bicycle/sparse/0/       (COLMAP + depth_params.json calibration)
  - CUDA submodule sources (rebuild required on the target machine)

Deliberately excluded (see README "本包包含/不包含"):
  - Depth-Anything-2/3, third_party/, output/ checkpoints (.pth / .ply)
  - normals_from_pretrain/, mono_depth/, images/, images_2/, images_8/
  - __pycache__ / build / .git / *.pyd / *.so

Usage:
    python scripts/pack_bicycle_v14.py
    python scripts/pack_bicycle_v14.py --out D:/somewhere/TAGSs_bicycle_v14.zip
    python scripts/pack_bicycle_v14.py --keep-stage   # inspect staging tree first
"""

import argparse
import os
import shutil
import stat
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PKG_NAME = "TAGSs_bicycle_v14"

# --------------------------------------------------------------------------
# Exclusions
# --------------------------------------------------------------------------
IGNORE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".svn",
    ".vscode",
    ".idea",
    "build",
    "dist",
    ".eggs",
    ".egg-info",
    "dist-info",
    "node_modules",
    "output",
    "wandb",
    "Depth-Anything-3",
    "Depth-Anything-V2",
    "third_party",
    "packaging",
    "data",
}

IGNORE_FILE_NAMES = {
    ".ds_store",
    "thumbs.db",
    ".git",
    ".gitmodules",
    ".gitignore",
}

IGNORE_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".egg",
    ".whl",
    ".pth",
    ".ckpt",
}

# --------------------------------------------------------------------------
# Code / docs to include
# --------------------------------------------------------------------------
ROOT_FILES = [
    "train.py",
    "train_optimized.py",
    "render.py",
    "metrics.py",
    "convert.py",
    "full_eval.py",
    "export_ply.py",
    "environment.yml",
    "LICENSE.md",
    "DEPENDENCIES_README.md",
    "EXPERIMENT_SUMMARY.md",
    "VERSION_HISTORY.md",
    "results.md",
    "README.md",
]

ROOT_DIRS = [
    "arguments",
    "experiments",
    "gaussian_renderer",
    "scene",
    "utils",
    "lpipsPyTorch",
    "tests",
]

SUBMODULE_DIRS = [
    "submodules/diff-gaussian-rasterization",
    "submodules/simple-knn",
    "submodules/fused-ssim",
]

TEXT_LF_SUFFIXES = {".sh", ".env", ".bashrc", ".ps1"}

# v14 scripts (primary + the 1237 re-run driver that actually reproduces it)
V14_SCRIPTS = [
    "scripts/run_bicycle_v14_polish_checkpoint.sh",
    "scripts/run_bicycle_rerun_1237.ps1",
    "scripts/_launch_bicycle_rerun_detached.ps1",
    "scripts/summarize_bicycle_rerun_1237.py",
    "scripts/summarize_v4restore_ablation.py",
    "scripts/export_stage_metrics_to_tensorboard.py",
    "scripts/summarize_bicycle_3way.py",
    "scripts/summarize_bicycle_e4_vs_dn.py",
    "scripts/read_train_progress.py",
    "scripts/summarize_live_logs.py",
    "scripts/watch_train_progress.ps1",
    "scripts/stage1_status_check.py",
    "scripts/export_bicycle_mono_depth.py",
    "scripts/rewrite_bicycle_transforms_stdlib.py",
]

# Scripts that tests/ in this package import. They are vr_room / exp4 tooling,
# not part of the v14 path, but shipping them keeps `python -m unittest discover tests`
# green on the target machine instead of failing at collection time.
TEST_SUPPORT_SCRIPTS = [
    "scripts/exp4_summarize_v2.py",
    "scripts/summarize_vr_room_e4.py",
    "scripts/summarize_vr_room_e4b.py",
]

# --------------------------------------------------------------------------
# Data to include (bicycle v14 preprocessed inputs)
# --------------------------------------------------------------------------
DATA_FILES = [
    "data/bicycle/sparse/0/depth_params.json",
    "data/bicycle/transforms.json",
    "data/bicycle/transforms.fullres.json",
    "data/bicycle/poses_bounds.npy",
    "data/bicycle/sparse_pc.ply",
]

DATA_DIRS = [
    "data/bicycle/images_4",
    "data/bicycle/depth_any_out",
    "data/bicycle/sparse/0",
]

# --------------------------------------------------------------------------
# Docs + result artifacts
# --------------------------------------------------------------------------
V14_DOCS = [
    "output/bicycle_v14_experiment_architecture.md",
    "output/bicycle_v14_experiment_evolution_architecture_analysis.md",
    "output/bicycle_v14_polish_checkpoint_summary.md",
    "output/bicycle_v14_polish_checkpoint_summary.json",
    "docs/THESIS_SYSTEM_DESIGN.md",
    "docs/ALL_EXPERIMENTS_SUMMARY.md",
    "docs/BICYCLE_E4_VS_DN.md",
    "docs/VR_ROOM_EXPERIMENT_PAPER.md",
]

# Small run artifacts worth keeping for provenance (no ckpt / ply).
V14_RUN_ARTIFACTS = [
    ("output/bicycle_v14_v8_monitor_28500_select_1237/cfg_args", "reference/local_1237_run/cfg_args"),
    (
        "output/bicycle_v14_v8_monitor_28500_select_1237/cameras.json",
        "reference/local_1237_run/cameras.json",
    ),
    (
        "output/bicycle_v14_v8_monitor_28500_select_1237/live_logs/stage_snapshots.jsonl",
        "reference/local_1237_run/live_logs/stage_snapshots.jsonl",
    ),
    (
        "output/bicycle_v14_v8_monitor_28500_select_1237/live_logs/train_iter_metrics.jsonl",
        "reference/local_1237_run/live_logs/train_iter_metrics.jsonl",
    ),
    (
        "output/bicycle_v14_v8_monitor_28500_select_1237.log",
        "reference/local_1237_run/train.log",
    ),
    ("output/v14_rerun.out", "reference/local_1237_run/rerun.out"),
    ("output/v14_rerun.err", "reference/local_1237_run/rerun.err"),
]


def should_skip(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    if lower in IGNORE_FILE_NAMES:
        return True
    if path.suffix.lower() in IGNORE_FILE_SUFFIXES:
        return True
    if name.endswith(".egg-info") or name.endswith(".dist-info"):
        return True
    return False


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if src.suffix.lower() in TEXT_LF_SUFFIXES or src.name.endswith(".env"):
        data = dst.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        dst.write_bytes(data)
        # keep executables runnable after unzip on Linux
        if src.suffix.lower() == ".sh":
            dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    for current, dirnames, filenames in os.walk(src):
        current_path = Path(current)
        rel = current_path.relative_to(src)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in IGNORE_DIR_NAMES
            and not d.endswith(".egg-info")
            and not d.endswith(".dist-info")
        ]
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            src_file = current_path / filename
            if should_skip(src_file):
                continue
            copy_file(src_file, target_dir / filename)


def copy_existing(pairs) -> tuple:
    copied, missing = [], []
    for rel in pairs:
        src = ROOT / rel
        dst = PKG_ROOT / rel
        if src.is_file():
            copy_file(src, dst)
            copied.append(rel)
        elif src.is_dir():
            copy_tree(src, PKG_ROOT / rel)
            copied.append(rel + "/ (tree)")
        else:
            missing.append(rel)
    return copied, missing


def copy_artifacts(pairs) -> tuple:
    copied, missing = [], []
    for src_rel, dst_rel in pairs:
        src = ROOT / src_rel
        if src.is_file():
            copy_file(src, PKG_ROOT / dst_rel)
            copied.append(dst_rel)
        else:
            missing.append(src_rel)
    return copied, missing


def write_text_lf(path: Path, text: str) -> None:
    """Write text with LF newlines (py3.7 Path.write_text has no newline kwarg)."""
    with open(str(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def build_data_readme(pkg_root: Path, copied_data, missing_data) -> None:
    depth_dir = pkg_root / "data/bicycle/depth_any_out"
    depth_n = len(list(depth_dir.glob("*.png"))) if depth_dir.is_dir() else 0
    img_dir = pkg_root / "data/bicycle/images_4"
    img_n = len([p for p in img_dir.glob("*") if p.is_file()]) if img_dir.is_dir() else 0

    text = f"""# bicycle v14 数据说明（随包携带）

本目录是 **v14 复现实验已经用过的预处理结果**，目标机无需重跑 Depth-Anything。

## 目录

| 路径 | 内容 | 数量 |
|---|---|---|
| `images_4/` | 原图下采样到 1237x822（当前 1237 复现协议） | {img_n} 张 |
| `depth_any_out/` | Depth-Anything 单目深度，uint8 逆深度 PNG | {depth_n} 张 |
| `sparse/0/` | COLMAP 重建（`cameras.bin` / `images.bin` / `points3D.bin`） + `depth_params.json` | — |

## 命名对齐

- 三者文件主名一一对应：`_DSC8679` ...
- `depth_params.json` 是 `utils/make_depth_scale.py` 标定出来的 median scale
  （`med_scale = 0.4677118023188449`），换机器后**不要**重新生成，会改变实验结果。
- v14 训练脚本通过 `--depths <bicycle>/depth_any_out` 读取深度。

## 未包含（体积原因）

- `images/`、`images_2/`、`images_8/`：其它分辨率协议，本包只带 1237 协议用的 `images_4`
- `normals_from_pretrain/`（2.4 GB）、`mono_depth/`（752 MB）：v14 的法线先验是由深度
  现推的（`experiments/geo_prior.py` 的 `depth_to_normals`），不需要外挂法线图
- `Depth-Anything-*`、`third_party/`：预处理工具链，训练不依赖

## 目标机用法

```bash
export SOURCE=/path/to/{PKG_NAME}/data/bicycle
export DEPTHS=$SOURCE/depth_any_out
```

`-s $SOURCE --images images_4 --depths $DEPTHS --resolution -1`
"""
    write_text_lf(pkg_root / "data/bicycle/README.md", text)


def build_manifest(pkg_root: Path) -> None:
    lines = [
        f"package: {PKG_NAME}",
        f"built: {date.today().isoformat()}",
        "scope: bicycle v14 (v8 monitor_28500_select) reproduction bundle",
        "entry_script: scripts/run_bicycle_rerun_1237.ps1  (env RERUN_CASE=v14)",
        "alt_script: scripts/run_bicycle_v14_polish_checkpoint.sh  (A100 multi-GPU, 5 cases)",
        "data: data/bicycle/{images_4,depth_any_out,sparse/0}",
        "",
    ]
    for path in sorted(pkg_root.rglob("*")):
        if path.is_file():
            lines.append(path.relative_to(pkg_root).as_posix())
    write_text_lf(pkg_root / "MANIFEST.txt", "\n".join(lines) + "\n")


def pack(out_zip: Path, keep_stage: bool) -> Path:
    global PKG_ROOT
    stage_root = ROOT / "packaging" / f"_stage_{PKG_NAME}"
    pkg_root = stage_root / PKG_NAME
    PKG_ROOT = pkg_root

    if stage_root.exists():
        shutil.rmtree(stage_root)
    pkg_root.mkdir(parents=True)

    missing: list = []

    # 1. root entry points + docs
    for name in ROOT_FILES:
        src = ROOT / name
        if src.exists():
            copy_file(src, pkg_root / name)
        else:
            missing.append(name)

    # 2. core code dirs
    for dirname in ROOT_DIRS:
        src = ROOT / dirname
        if src.exists():
            copy_tree(src, pkg_root / dirname)
        else:
            missing.append(dirname + "/")

    # 3. v14 scripts (explicit list, keeps the package small and v14-focused)
    copied_scripts, miss = copy_existing(V14_SCRIPTS + TEST_SUPPORT_SCRIPTS)
    missing += miss

    # 4. CUDA submodule sources
    for dirname in SUBMODULE_DIRS:
        src = ROOT / dirname
        if src.exists():
            copy_tree(src, pkg_root / dirname)
        else:
            missing.append(dirname + "/")

    glm_dst = pkg_root / "submodules/diff-gaussian-rasterization/third_party/glm"
    if not (glm_dst / "glm" / "glm.hpp").is_file():
        for glm_src in (
            ROOT / "submodules/diff-gaussian-rasterization/third_party/glm",
            ROOT / "packaging/TAGSs_experiment_portable/submodules/diff-gaussian-rasterization/third_party/glm",
        ):
            if (glm_src / "glm" / "glm.hpp").is_file():
                if glm_dst.exists():
                    shutil.rmtree(glm_dst)
                copy_tree(glm_src, glm_dst)
                break
        if not (glm_dst / "glm" / "glm.hpp").is_file():
            raise FileNotFoundError("Missing glm headers (submodules/.../third_party/glm/glm/glm.hpp)")

    # 5. bicycle data
    copied_data, miss = copy_existing(DATA_DIRS + DATA_FILES)
    missing += miss
    build_data_readme(pkg_root, copied_data, missing)

    # 6. docs + result artifacts
    copied_docs, miss = copy_existing(V14_DOCS)
    missing += miss
    copied_artifacts, miss = copy_artifacts(V14_RUN_ARTIFACTS)
    missing += miss

    # 7. README (hand-written migration guide)
    readme_src = ROOT / "packaging/bicycle_v14_README.md"
    if readme_src.is_file():
        copy_file(readme_src, pkg_root / "README.md")

    # 8. MANIFEST
    build_manifest(pkg_root)

    # zip it
    if out_zip.exists():
        out_zip.unlink()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(pkg_root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path(PKG_NAME) / path.relative_to(pkg_root)))

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_zip} ({size_mb:.1f} MB)")
    print(f"  scripts copied : {len(copied_scripts)}")
    print(f"  docs copied    : {len(copied_docs) + len(copied_artifacts)}")
    print(f"  data copied    : {len(copied_data)} entries")
    if missing:
        print(f"  MISSING ({len(missing)}):")
        for m in missing:
            print(f"    - {m}")

    if not keep_stage:
        try:
            shutil.rmtree(stage_root)
        except OSError:
            pass
    return out_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack bicycle v14 into a portable zip.")
    parser.add_argument(
        "--out",
        default=str(ROOT / "packaging" / f"{PKG_NAME}.zip"),
        help="Output .zip path",
    )
    parser.add_argument("--keep-stage", action="store_true", help="Keep staging directory for inspection")
    args = parser.parse_args()
    pack(Path(args.out), keep_stage=args.keep_stage)


if __name__ == "__main__":
    main()