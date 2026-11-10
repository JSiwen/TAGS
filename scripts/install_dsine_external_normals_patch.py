#!/usr/bin/env python3
"""Install the DSINE external-normal source patch from its upload bundle.

Run this script from an extracted bundle and pass the TAGS checkout directory.
Every destination file is backed up before replacement; on a copy failure the
files already replaced by this invocation are restored from that backup.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys


PATCH_FILES = (
    "arguments/__init__.py",
    "experiments/config.py",
    "experiments/geo_prior.py",
    "experiments/training_engine.py",
    "gaussian_renderer/__init__.py",
    "scene/__init__.py",
    "scene/cameras.py",
    "scene/dataset_readers.py",
    "train_optimized.py",
    "utils/camera_utils.py",
    "utils/normal_utils.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="TAGS_github checkout root")
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parent
    patch_root = bundle_root / "files"
    target_root = Path(args.target).expanduser().resolve()
    if not (target_root / "train_optimized.py").is_file():
        parser.error(f"not a TAGS checkout (train_optimized.py missing): {target_root}")

    missing_patch = [str(patch_root / rel) for rel in PATCH_FILES if not (patch_root / rel).is_file()]
    missing_target = [str(target_root / rel) for rel in PATCH_FILES if not (target_root / rel).is_file()]
    if missing_patch:
        parser.error("bundle is incomplete; missing patch file(s):\n  " + "\n  ".join(missing_patch))
    if missing_target:
        parser.error("target checkout is missing expected file(s):\n  " + "\n  ".join(missing_target))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = target_root / ("backup_dsine_external_normals_" + stamp)
    replaced = []
    try:
        for rel in PATCH_FILES:
            old_path = target_root / rel
            backup_path = backup_root / rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, backup_path)
        for rel in PATCH_FILES:
            dst = target_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            temp_dst = dst.with_name(dst.name + ".dsine_patch_tmp")
            shutil.copy2(patch_root / rel, temp_dst)
            os.replace(str(temp_dst), str(dst))
            replaced.append(rel)
    except Exception:
        for rel in reversed(replaced):
            shutil.copy2(backup_root / rel, target_root / rel)
        raise

    print("Installed DSINE external-normal source files:")
    for rel in PATCH_FILES:
        print("  " + rel)
    print("Backup: " + str(backup_root))
    print("No training was started.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("INSTALL FAILED: " + str(exc), file=sys.stderr)
        raise
