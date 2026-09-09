#!/usr/bin/env python3
"""
Generate sparse/0/test.txt (and optional train.txt) for 3DGS COLMAP scenes.

Training with held-out test views:
  python train_optimized.py -s <scene> -m <out> --eval --llffhold 0 ...

After training / render:
  python render.py -m <out> --skip_train
  python metrics.py -m <out> --split test

When --llffhold 0, scene/dataset_readers.py reads sparse/0/test.txt (exact COLMAP image names, one per line).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text


def _load_image_names(sparse0: Path) -> list[str]:
    images_bin = sparse0 / "images.bin"
    images_txt = sparse0 / "images.txt"
    if images_bin.is_file():
        extr = read_extrinsics_binary(str(images_bin))
    elif images_txt.is_file():
        extr = read_extrinsics_text(str(images_txt))
    else:
        raise FileNotFoundError(f"Need {images_bin} or {images_txt}")
    return sorted({extr[k].name for k in extr})


def _verify_exist(images_dir: Path, names: list[str]) -> None:
    missing = [n for n in names if not (images_dir / n).is_file()]
    if missing:
        sample = missing[:5]
        raise SystemExit(
            f"{len(missing)} image(s) missing under {images_dir}. Examples: {sample}"
        )


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Write COLMAP train/test split for 3DGS --eval --llffhold 0")
    p.add_argument("-s", "--source", required=True, type=str, help="COLMAP scene root (contains sparse/0, images)")
    p.add_argument("--images", default="images", type=str, help="RGB folder name under source (default: images)")
    p.add_argument(
        "--strategy",
        choices=("every", "random"),
        default="random",
        help="every: LLFF-style every k-th in sorted filename order; random: shuffle then take ratio (default: random)",
    )
    p.add_argument("--every", type=int, default=8, help="For strategy=every: test every k-th image (e.g. 8)")
    p.add_argument("--test_ratio", type=float, default=0.1, help="For strategy=random: fraction of views as test (0,1)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for strategy=random")
    p.add_argument("--min_test", type=int, default=1, help="Ensure at least this many test images")
    p.add_argument("--dry_run", action="store_true", help="Print counts only; do not write files")
    args = p.parse_args()

    source = Path(os.path.abspath(args.source))
    sparse0 = source / "sparse" / "0"
    if not sparse0.is_dir():
        raise SystemExit(f"Missing COLMAP reconstruction: {sparse0}")

    names = _load_image_names(sparse0)
    if not names:
        raise SystemExit("No registered images found in COLMAP model.")
    if len(names) < 2:
        raise SystemExit("Need at least 2 registered images to split train/test.")
    images_dir = source / args.images
    _verify_exist(images_dir, names)

    n = len(names)
    if args.strategy == "every":
        k = max(1, int(args.every))
        test_names = sorted({names[i] for i in range(n) if i % k == 0})
    else:
        import random
        if not (0.0 < args.test_ratio < 1.0):
            raise SystemExit("--test_ratio must be between 0 and 1 (exclusive)")
        rng = random.Random(args.seed)
        shuffled = names[:]
        rng.shuffle(shuffled)
        n_test = max(args.min_test, int(round(n * args.test_ratio)))
        # keep at least one training view
        n_test = min(max(1, n - 1), n_test)
        test_names = sorted(set(shuffled[:n_test]))

    train_names = sorted(set(names) - set(test_names))
    if not train_names:
        raise SystemExit("Split left no training images; lower test_ratio or change seed.")

    print(f"Total registered images: {n}")
    print(f"Train: {len(train_names)}  |  Test: {len(test_names)}")

    out_test = sparse0 / "test.txt"
    out_train = sparse0 / "train.txt"

    if args.dry_run:
        print(f"Would write: {out_test} ({len(test_names)} lines)")
        print(f"Would write: {out_train} ({len(train_names)} lines)")
        return

    _write_lines(out_test, test_names)
    _write_lines(out_train, train_names)
    print(f"Wrote {out_test}")
    print(f"Wrote {out_train} (reference only; 3DGS uses test.txt when --eval --llffhold 0)")
    print()
    print("Next:")
    print(f"  python train_optimized.py -s {source} -m <output> --eval --llffhold 0 ...")
    print(f"  python render.py -m <output> --skip_train")
    print(f"  python metrics.py -m <output> --split test")


if __name__ == "__main__":
    main()
