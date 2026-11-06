#!/usr/bin/env python3
"""Print COLMAP-registered image basenames (one per line) for a TAGS scene.

Reads data/<scene>/sparse/0/{cameras,images,points3D}.bin via the same
read_write_model helper as export_vr_room_mono_depth.py, and prints the
basenames (e.g. 'frame_00112.jpg') of every image that has a pose. This
is the source of truth for "which frames belong to the training set";
images/ on disk may contain extra jpgs that were dropped because
COLMAP could not register them.

Usage:
  python scripts/list_vr_room_registered_images.py --data data/vr_room
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "utils"))

from read_write_model import read_model  # type: ignore  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--sparse", default="sparse/0")
    args = parser.parse_args()

    sparse_dir = Path(args.data) / args.sparse
    if not sparse_dir.is_dir():
        raise SystemExit(f"sparse dir not found: {sparse_dir}")
    cams, imgs, _ = read_model(str(sparse_dir), ext=".bin")
    for name in sorted(im.name for im in imgs.values()):
        if Path(name).suffix in IMAGE_EXTS:
            print(name)


if __name__ == "__main__":
    main()