#!/usr/bin/env python3
"""Export vr_room depth maps into DN-Splatter coolermap depth files.

The vr_room images under data/vr_room/images/ are copied verbatim from
dn-splatter's own upstream dataset (frame_00001.jpg .. frame_00422.jpg,
738x994). COLMAP SfM has been re-run on this copy, so data/vr_room/sparse/0
contains cameras.bin, images.bin, points3D.bin in the usual TAGS layout.

depth_params.json currently has scale=1.0, offset=0.0 for every frame (it is
a placeholder written by prepare_vr_room_colmap.py, not a real fit). The
mono depth files are therefore produced with a fixed identity scale/offset:
  aligned_inverse = inverse / 255
  metric          = 1 / aligned_inverse

The script also writes data/vr_room/transforms.json from the COLMAP sparse
model. If the existing transforms.json already points at images/ with the
right resolution, it is left in place and a backup is created once under
transforms.fullres.json.

Usage:
  python scripts/export_vr_room_mono_depth.py --data data/vr_room --overwrite --rewrite-transforms
  python scripts/export_vr_room_mono_depth.py --data data/vr_room --check-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "utils"))
from read_write_model import Camera, read_model  # type: ignore  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def normalize_inverse_depth(raw: np.ndarray) -> np.ndarray:
    """Match utils/make_depth_scale.py inverse-depth normalization."""
    array = raw
    if array.ndim != 2:
        array = array[..., 0]
    values = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer):
        if array.dtype == np.uint8:
            return values / np.float32(255.0)
        return values / np.float32(2**16)
    peak = float(values.max()) if values.size else 1.0
    if peak > 1.0:
        return values / np.float32(peak or 1.0)
    return values


def apply_scale_offset(inverse_depth: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return inverse_depth * np.float32(scale) + np.float32(offset)


def inverse_to_metric(aligned_inverse: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    metric = np.zeros_like(aligned_inverse, dtype=np.float32)
    valid = aligned_inverse > eps
    metric[valid] = np.float32(1.0) / aligned_inverse[valid]
    return metric


def holdout_names(names, hold: int):
    ordered = sorted(names)
    if hold <= 0:
        return []
    return [name for index, name in enumerate(ordered) if index % hold == 0]


def first_image_size(images_dir: Path) -> tuple[int, int]:
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        height, width = image.shape[:2]
        return width, height
    raise FileNotFoundError(f"No readable image under {images_dir}")


def _colmap_to_nerfstudio(camera, image, point_threshold: float = 0.0):
    """Build a nerfstudio-style frame dict from COLMAP Camera + Image entries.

    COLMAP image.qvec / qvec2rotmat yields camera-to-world rotation as a
    3x3. We expand it to a homogeneous 4x4 transform_matrix. The
    translation is the image.tvec field (camera center in world coords,
    not the camera origin in world coords -- COLMAP convention is
    t_world_from_cam = -R^T t, but their Image.tvec is already
    -R^T t, so we use it directly).
    """
    R = image.qvec2rotmat()
    t = image.tvec.reshape(3, 1)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3, 3:4] = t
    if camera.model in ("SIMPLE_PINHOLE",):
        fx, fy, cx, cy = float(camera.params[0]), float(camera.params[0]), float(camera.params[1]), float(camera.params[2])
    elif camera.model in ("PINHOLE",):
        fx, fy, cx, cy = (float(v) for v in camera.params[:4])
    else:
        raise ValueError(f"Unsupported camera model for transforms.json: {camera.model}")
    return {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": int(camera.width),
        "h": int(camera.height),
        "file_path": f"./images/{image.name}",
        "transform_matrix": c2w.tolist(),
    }


def build_transforms_from_colmap(sparse_dir: Path, image_paths) -> dict:
    """Read COLMAP sparse model and return a nerfstudio transforms.json payload."""
    cameras, images, _ = read_model(str(sparse_dir), ext=".bin")
    by_name = {Path(im.name).name: im for im in images.values()}
    frames = []
    seen = set()
    for path in image_paths:
        name = path.name
        if name not in by_name:
            raise FileNotFoundError(f"{name} not in COLMAP images.bin")
        im = by_name[name]
        if im.camera_id not in cameras:
            raise KeyError(f"COLMAP image {name} references missing camera {im.camera_id}")
        frames.append(_colmap_to_nerfstudio(cameras[im.camera_id], im))
        seen.add(name)
    if len(seen) != len(by_name):
        extra = sorted(set(by_name) - seen)
        if extra:
            print(f"[export] WARNING: {len(extra)} COLMAP images are not in images/; ignoring: {extra[:3]}{'...' if len(extra) > 3 else ''}")
    return {
        "camera_model": "PINHOLE",
        "frames": frames,
    }


def rewrite_transforms(payload: dict, image_dirname: str, target_size: tuple[int, int]):
    """Point frames at image_dirname and scale intrinsics onto target_size."""
    target_w, target_h = int(target_size[0]), int(target_size[1])
    frames = payload.get("frames") or []
    current_w = int(payload.get("w") or 0)
    current_h = int(payload.get("h") or 0)
    if current_w <= 0 or current_h <= 0:
        if frames:
            current_w = int(frames[0].get("w") or 0)
            current_h = int(frames[0].get("h") or 0)
    if current_w <= 0 or current_h <= 0:
        raise ValueError("transforms.json is missing top-level w/h and per-frame w/h")
    already = (
        current_w == target_w
        and current_h == target_h
        and frames
        and all(
            str(frame.get("file_path", "")).replace("\\", "/").startswith(image_dirname + "/")
            for frame in frames
        )
    )
    if already:
        return payload, False

    scale_x = target_w / float(current_w)
    scale_y = target_h / float(current_h)
    rewritten = dict(payload)
    rewritten["w"] = target_w
    rewritten["h"] = target_h
    for key, scale in (("fl_x", scale_x), ("fl_y", scale_y), ("cx", scale_x), ("cy", scale_y)):
        if key in rewritten:
            rewritten[key] = float(rewritten[key]) * scale

    new_frames = []
    for frame in frames:
        item = dict(frame)
        name = Path(str(item.get("file_path", ""))).name
        if not name:
            raise ValueError("transforms frame is missing file_path")
        item["file_path"] = str(Path(image_dirname) / name).replace("\\", "/")
        for key, scale in (("fl_x", scale_x), ("fl_y", scale_y), ("cx", scale_x), ("cy", scale_y)):
            if key in item:
                item[key] = float(item[key]) * scale
        item["w"] = target_w
        item["h"] = target_h
        new_frames.append(item)
    rewritten["frames"] = new_frames
    return rewritten, True


def export_scene(data_root, images_dirname, depths_dirname, depth_params_path, out_dirname, hold, overwrite, rewrite):
    data_root = Path(data_root)
    images_dir = data_root / images_dirname
    depths_dir = data_root / depths_dirname
    params_path = data_root / depth_params_path if not Path(depth_params_path).is_absolute() else Path(depth_params_path)
    out_dir = data_root / out_dirname
    sparse_dir = data_root / "sparse" / "0"
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    if not depths_dir.is_dir():
        raise FileNotFoundError(depths_dir)
    if not sparse_dir.is_dir():
        raise FileNotFoundError(sparse_dir)
    if not params_path.is_file():
        raise FileNotFoundError(params_path)

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS],
        key=lambda p: p.name,
    )
    if not image_paths:
        raise FileNotFoundError(f"no images in {images_dir}")

    target_size = first_image_size(images_dir)

    transforms_path = data_root / "transforms.json"
    transforms_fullres_path = data_root / "transforms.fullres.json"
    # Filter out images that have no COLMAP pose. Otherwise we'd write
    # mono_depth files for frames the training pipeline has no camera
    # intrinsics+extrinsics for.
    cameras, images, _ = read_model(str(sparse_dir), ext=".bin")
    registered = {Path(im.name).name for im in images.values()}
    skipped_unregistered = [p.name for p in image_paths if p.name not in registered]
    if skipped_unregistered:
        print(f"[export] dropping {len(skipped_unregistered)} images without COLMAP pose: {skipped_unregistered}")
        image_paths = [p for p in image_paths if p.name in registered]
        if not image_paths:
            raise RuntimeError("no images have COLMAP pose; cannot proceed")

    payload = None
    if transforms_path.is_file():
        payload = json.loads(transforms_path.read_text(encoding="utf-8"))
        existing_paths = {Path(str(f.get("file_path", ""))).name for f in payload.get("frames", [])}
        image_set = {p.name for p in image_paths}
        if existing_paths != image_set:
            print(f"[export] existing transforms.json has {len(existing_paths)} frames, expected {len(image_set)}; rebuilding from COLMAP")
            payload = build_transforms_from_colmap(sparse_dir, image_paths)
            transforms_source = "colmap_sparse_rebuild"
        else:
            transforms_source = "existing"
    else:
        payload = build_transforms_from_colmap(sparse_dir, image_paths)
        transforms_source = "colmap_sparse"

    out_dir.mkdir(parents=True, exist_ok=True)

    params = json.loads(params_path.read_text(encoding="utf-8"))

    written = 0
    skipped = 0
    missing_depth = []
    missing_params = []
    for image_path in image_paths:
        stem = image_path.stem
        # depth files are .png regardless of the image extension; some
        # scenes (vr_room) use .jpg for images and .png for depth.
        depth_path = depths_dir / f"{stem}.png"
        dest = out_dir / f"{stem}_aligned.npy"
        record = params.get(stem)
        if not depth_path.is_file():
            missing_depth.append(image_path.name)
            continue
        if record is None:
            missing_params.append(stem)
            continue
        scale = float(record["scale"])
        offset = float(record["offset"])
        if dest.is_file() and not overwrite:
            skipped += 1
            continue
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            missing_depth.append(image_path.name)
            continue
        inverse = normalize_inverse_depth(raw)
        aligned = apply_scale_offset(inverse, scale, offset)
        metric = inverse_to_metric(aligned)
        if (metric.shape[1], metric.shape[0]) != target_size:
            metric = cv2.resize(metric, target_size, interpolation=cv2.INTER_LINEAR)
        np.save(dest, metric.astype(np.float32))
        written += 1

    names = [p.name for p in image_paths]
    holdout = holdout_names(names, hold)
    manifest = {
        "scene": data_root.name,
        "images_dirname": images_dirname,
        "target_size_wh": list(target_size),
        "num_images": len(image_paths),
        "num_holdout": len(holdout),
        "hold": hold,
        "holdout_names": holdout,
        "written": written,
        "skipped_existing": skipped,
        "missing_depth": missing_depth,
        "missing_params": missing_params,
        "transforms_source": transforms_source,
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    transforms_changed = False
    if rewrite:
        updated, transforms_changed = rewrite_transforms(payload, images_dirname, target_size)
        if transforms_changed:
            if transforms_path.is_file() and not transforms_fullres_path.is_file():
                shutil.copy2(transforms_path, transforms_fullres_path)
            transforms_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        manifest["transforms_rewritten"] = transforms_changed
        (out_dir / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    return manifest


def check_scene(data_root, images_dirname, out_dirname, hold, sparse_dirname="sparse/0"):
    data_root = Path(data_root)
    images_dir = data_root / images_dirname
    out_dir = data_root / out_dirname
    image_names_all = sorted(
        p.name for p in images_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS
    )

    # Drop frames that have no COLMAP pose, matching export_scene.
    try:
        cameras, images, _ = read_model(str(data_root / sparse_dirname), ext=".bin")
        registered = {Path(im.name).name for im in images.values()}
    except FileNotFoundError:
        registered = set(image_names_all)
    dropped = sorted(set(image_names_all) - registered)

    image_names = sorted(n for n in image_names_all if n in registered)
    npy_stems = sorted(p.name[: -len("_aligned.npy")] for p in out_dir.glob("*_aligned.npy"))
    image_stems = sorted(Path(n).stem for n in image_names)
    missing = sorted(set(image_stems) - set(npy_stems))
    extra = sorted(set(npy_stems) - set(image_stems))
    holdout = holdout_names(image_names, hold)
    transforms_path = data_root / "transforms.json"
    normals_path = data_root / "normals_from_pretrain"
    normal_count = 0
    if normals_path.is_dir():
        normal_count = sum(1 for p in normals_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    report = {
        "num_images_in_dir": len(image_names_all),
        "num_images_with_colmap_pose": len(image_names),
        "dropped_without_colmap_pose": dropped,
        "num_aligned_npy": len(npy_stems),
        "num_holdout": len(holdout),
        "holdout_names": holdout,
        "missing_npy_stems": missing,
        "extra_npy_stems": extra,
        "aligned": not missing and not extra and len(npy_stems) == len(image_stems),
        "transforms_present": transforms_path.is_file(),
        "normals_count": normal_count,
        "normals_aligned": normal_count == len(image_names),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/vr_room")
    parser.add_argument("--images", default="images")
    parser.add_argument("--depths", default="depth")
    parser.add_argument("--depth-params", default="sparse/0/depth_params.json")
    parser.add_argument("--out", default="mono_depth")
    parser.add_argument("--hold", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rewrite-transforms", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        report = check_scene(args.data, args.images, args.out, args.hold)
        print(json.dumps(report, indent=2))
        # Report but do not fail on missing normals; the user may be
        # running this between -Stage export and -Stage normals. The
        # PowerShell wrapper decides whether the overall state is
        # "ready to train".
        if not report["aligned"] or not report["transforms_present"]:
            raise SystemExit(1)
        return

    manifest = export_scene(
        args.data,
        args.images,
        args.depths,
        args.depth_params,
        args.out,
        args.hold,
        args.overwrite,
        args.rewrite_transforms,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "holdout_names"}, indent=2))
    if manifest["missing_depth"] or manifest["missing_params"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
