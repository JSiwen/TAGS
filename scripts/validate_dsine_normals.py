#!/usr/bin/env python3
"""Validate DSINE normal maps against Bicycle images_4.

The normal directory can be supplied explicitly with --normal-dir. If omitted,
the script searches data/bicycle for directories named normals_dsine* and looks
for .npy files recursively, so both a flat extraction and a nested ZIP layout
are supported.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "bicycle",
        help="Bicycle dataset directory containing images_4 (default: repo data/bicycle)",
    )
    parser.add_argument(
        "--normal-dir",
        type=Path,
        default=None,
        help="Directory containing DSINE .npy maps; searched recursively. "
        "If omitted, searches normals_dsine* under --source.",
    )
    parser.add_argument(
        "--expected-count", type=int, default=194,
        help="Expected number of Bicycle images/maps (default: 194; use 0 to disable)",
    )
    return parser.parse_args()


def collect_normal_files(source: Path, normal_dir: Path | None) -> list[Path]:
    if normal_dir is not None:
        if not normal_dir.is_dir():
            raise FileNotFoundError(f"Normal directory does not exist: {normal_dir}")
        roots = [normal_dir]
    else:
        roots = sorted(
            p for p in source.iterdir()
            if p.is_dir() and p.name.lower().startswith("normals_dsine")
        )
        if not roots:
            archives = sorted(source.glob("*normal*.zip"))
            extra = "\nFound ZIP archive(s):\n  " + "\n  ".join(map(str, archives)) if archives else ""
            raise FileNotFoundError(
                f"No normals_dsine* directory found under {source}. "
                "Extract the uploaded ZIP or pass --normal-dir." + extra
            )

    files = sorted({p.resolve() for root in roots for p in root.rglob("*.npy")})
    if not files:
        roots_text = "\n  ".join(map(str, roots))
        raise FileNotFoundError(
            "No .npy normal maps found. Searched recursively in:\n  " + roots_text
        )
    return files


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    image_dir = source / "images_4"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"images_4 directory does not exist: {image_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    normal_files = collect_normal_files(source, args.normal_dir)
    duplicates = [stem for stem, count in Counter(p.stem for p in normal_files).items() if count > 1]
    if duplicates:
        raise ValueError(
            "Multiple .npy files share these image stems; pass --normal-dir to select "
            f"one directory: {duplicates[:10]}"
        )

    images = {p.stem: p for p in image_paths}
    normals = {p.stem: p for p in normal_files}
    missing = sorted(images.keys() - normals.keys())
    extra = sorted(normals.keys() - images.keys())
    if missing or extra:
        print(f"images_4: {len(images)}; normal maps found: {len(normals)}")
        if missing:
            print(f"Missing normals ({len(missing)}): {missing[:20]}")
        if extra:
            print(f"Unmatched normal maps ({len(extra)}): {extra[:20]}")
        raise AssertionError("Image and normal-map stems do not match")

    if args.expected_count and len(images) != args.expected_count:
        raise AssertionError(
            f"Expected {args.expected_count} images/maps, found {len(images)}"
        )

    norm_min = float("inf")
    norm_max = 0.0
    for stem, image_path in images.items():
        normal_path = normals[stem]
        normal = np.load(normal_path, allow_pickle=False)
        if normal.dtype != np.float32:
            raise TypeError(f"{normal_path}: expected float32, got {normal.dtype}")
        if normal.ndim != 3 or normal.shape[2] != 3:
            raise ValueError(f"{normal_path}: expected HxWx3, got {normal.shape}")
        if not np.isfinite(normal).all():
            raise ValueError(f"{normal_path}: contains NaN or Inf")

        with Image.open(image_path) as image:
            expected_shape = (image.height, image.width, 3)
        if normal.shape != expected_shape:
            raise ValueError(
                f"{stem}: normal shape {normal.shape}, image requires {expected_shape}"
            )

        lengths = np.linalg.norm(normal, axis=2)
        norm_min = min(norm_min, float(lengths.min()))
        norm_max = max(norm_max, float(lengths.max()))
        if np.max(np.abs(lengths - 1.0)) > 0.02:
            raise ValueError(
                f"{normal_path}: normals are not unit length within 0.02 tolerance "
                f"(length range {lengths.min():.5f}..{lengths.max():.5f})"
            )

    print(f"OK: {len(images)} images matched with {len(normals)} DSINE normal maps")
    print(f"images: {image_dir}")
    print("normal map directories:")
    for parent, count in Counter(str(p.parent) for p in normal_files).items():
        print(f"  {parent}: {count} .npy")
    print(f"normal lengths: {norm_min:.6f}..{norm_max:.6f}")
    print("format: float32 HxWx3, finite, unit length")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise
