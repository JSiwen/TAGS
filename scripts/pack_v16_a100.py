import argparse
import os
import shutil
import stat
import tarfile
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

IGNORE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".svn",
    "build",
    "dist",
    ".eggs",
    "output",
    "Depth-Anything-3",
    "Depth-Anything-V2",
    "packaging",
    "assets",
    "images",
    "data",
}

IGNORE_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".egg",
    ".whl",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".exr",
    ".ply",
    ".bin",
}

IGNORE_FILE_NAMES = {
    ".ds_store",
    "thumbs.db",
    ".git",
    ".gitmodules",
}

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
]

ROOT_DIRS = [
    "arguments",
    "experiments",
    "gaussian_renderer",
    "scene",
    "utils",
    "lpipsPyTorch",
    "scripts",
    "docs",
    "tests",
]

SUBMODULE_DIRS = [
    "submodules/diff-gaussian-rasterization",
    "submodules/simple-knn",
    "submodules/fused-ssim",
]

TEXT_LF_SUFFIXES = {".sh", ".env", ".bashrc"}

REFERENCE_V16 = [
    (
        ROOT / "output/v16_local_3080/bicycle_v16_3080_resolution_aware/cfg_args",
        "reference/3080_v16/cfg_args",
    ),
    (
        ROOT / "output/v16_local_3080/bicycle_v16_3080_resolution_aware/schedule_plan.json",
        "reference/3080_v16/schedule_plan.json",
    ),
    (
        ROOT / "output/v16_local_3080/bicycle_v16_3080_resolution_aware/results.json",
        "reference/3080_v16/results.json",
    ),
    (
        ROOT / "output/v16_local_3080/bicycle_v16_3080_resolution_aware/per_view.json",
        "reference/3080_v16/per_view.json",
    ),
    (
        ROOT / "output/v16_local_3080/bicycle_v16_3080_stage_selection_summary.md",
        "reference/3080_v16/stage_selection_summary.md",
    ),
    (
        ROOT / "output/v16_local_3080/bicycle_v16_3080_stage_selection_summary.json",
        "reference/3080_v16/stage_selection_summary.json",
    ),
    (
        ROOT / "output/v16_local_3080/bicycle_v16_3080_checkpoint_summary.md",
        "reference/3080_v16/checkpoint_summary.md",
    ),
    (
        ROOT / "output/v16_local_3080/bicycle_v16_3080_checkpoint_summary.json",
        "reference/3080_v16/checkpoint_summary.json",
    ),
]

REFERENCE_E1B = [
    (
        ROOT / "output/v15_30k_fix_e1b_densify7k/bicycle_v15_3080_e1b_densify7k/cfg_args",
        "reference/3080_e1b/cfg_args",
    ),
    (
        ROOT / "output/v15_30k_fix_e1b_densify7k/bicycle_v15_3080_e1b_densify7k/results.json",
        "reference/3080_e1b/results.json",
    ),
    (
        ROOT / "output/v15_30k_fix_e1b_densify7k/bicycle_v15_3080_e1b_densify7k_stage_selection_summary.md",
        "reference/3080_e1b/stage_selection_summary.md",
    ),
    (
        ROOT / "output/v15_30k_fix_e1b_densify7k/bicycle_v15_3080_e1b_densify7k_checkpoint_summary.md",
        "reference/3080_e1b/checkpoint_summary.md",
    ),
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


def pack(out_tar: Path, keep_stage: bool) -> Path:
    stamp = date.today().isoformat()
    pkg_name = "TAGSs_v16_a100"
    stage_root = ROOT / "packaging" / f"_stage_{pkg_name}"
    pkg_root = stage_root / pkg_name

    if stage_root.exists():
        shutil.rmtree(stage_root)
    pkg_root.mkdir(parents=True)

    for name in ROOT_FILES:
        src = ROOT / name
        if src.exists():
            copy_file(src, pkg_root / name)

    copy_file(ROOT / "packaging/v16_a100_README.md", pkg_root / "README.md")

    for dirname in ROOT_DIRS:
        src = ROOT / dirname
        if src.exists():
            copy_tree(src, pkg_root / dirname)

    for dirname in SUBMODULE_DIRS:
        copy_tree(ROOT / dirname, pkg_root / dirname)

    glm_dst = pkg_root / "submodules/diff-gaussian-rasterization/third_party/glm"
    glm_src_candidates = [
        ROOT / "submodules/diff-gaussian-rasterization/third_party/glm",
        ROOT / "packaging/TAGSs_experiment_portable/submodules/diff-gaussian-rasterization/third_party/glm",
    ]
    glm_ready = (glm_dst / "glm" / "glm.hpp").is_file()
    if not glm_ready:
        for glm_src in glm_src_candidates:
            if (glm_src / "glm" / "glm.hpp").is_file():
                if glm_dst.exists():
                    shutil.rmtree(glm_dst)
                copy_tree(glm_src, glm_dst)
                break
        if not (glm_dst / "glm" / "glm.hpp").is_file():
            raise FileNotFoundError(
                "Missing glm headers. Expected them under "
                "submodules/diff-gaussian-rasterization/third_party/glm"
            )

    copy_file(
        ROOT / "packaging/v16_a100_reference_README.md",
        pkg_root / "reference/3080_v16/README.md",
    )
    copy_file(
        ROOT / "packaging/v15_e1b_a100_reference_README.md",
        pkg_root / "reference/3080_e1b/README.md",
    )
    for src, rel in REFERENCE_V16 + REFERENCE_E1B:
        if src.exists():
            copy_file(src, pkg_root / rel)

    manifest_lines = [
        f"package: {pkg_name}",
        f"built: {stamp}",
        "recipe: v16 resolution-aware schedule + adaptive densify extension",
        "a100_default: images + resolution=-1, 30k, eval/llffhold 8",
        "3080_reference: ours_28500 PSNR 26.9117 SSIM 0.837794 LPIPS 0.135433",
        "prev_3080_best: v15 E1b ours_28500 PSNR 26.6930 SSIM 0.828843 LPIPS 0.149942",
        "",
    ]
    for path in sorted(pkg_root.rglob("*")):
        if path.is_file():
            manifest_lines.append(path.relative_to(pkg_root).as_posix())
    (pkg_root / "MANIFEST.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8", newline="\n"
    )

    out_tar.parent.mkdir(parents=True, exist_ok=True)
    if out_tar.exists():
        out_tar.unlink()
    with tarfile.open(out_tar, "w:gz") as tar:
        tar.add(pkg_root, arcname=pkg_name)

    if not keep_stage:
        try:
            shutil.rmtree(stage_root)
        except OSError:
            pass
    return out_tar


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack v16 resolution-aware code for A100 migration.")
    parser.add_argument(
        "--out",
        default=str(ROOT / "packaging" / "TAGSs_v16_a100.tar.gz"),
        help="Output .tar.gz path",
    )
    parser.add_argument("--keep-stage", action="store_true", help="Keep staging directory")
    args = parser.parse_args()
    out_tar = pack(Path(args.out), keep_stage=args.keep_stage)
    size_mb = out_tar.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_tar} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
