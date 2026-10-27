#!/usr/bin/env python3
"""Export bicycle Depth Anything maps into DN-Splatter coolermap depth files.

Normalization matches utils/make_depth_scale.py (the fit that wrote
depth_params.json): uint8 inverse depth is divided by 255, then
scale/offset are applied. Metric z = 1 / aligned_inverse is what
DN stores in mono_depth/*_aligned.npy.

The script also rewrites data/bicycle/transforms.json so coolermap reads
images_4 with intrinsics scaled to that resolution. The original file is
kept as transforms.fullres.json.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def normalize_inverse_depth(raw):
    """Match utils/make_depth_scale.py, not the runtime millimeter branch."""
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


def apply_scale_offset(inverse_depth, scale, offset):
    return inverse_depth * np.float32(scale) + np.float32(offset)


def inverse_to_metric(aligned_inverse, eps=1e-6):
    metric = np.zeros_like(aligned_inverse, dtype=np.float32)
    valid = aligned_inverse > eps
    metric[valid] = np.float32(1.0) / aligned_inverse[valid]
    return metric


def holdout_names(names, hold=8):
    ordered = sorted(names)
    if hold <= 0:
        return []
    return [name for index, name in enumerate(ordered) if index % hold == 0]


def tags_train_resolution(width, height, resolution=800, resolution_scale=1.0):
    """Same rule as utils/camera_utils.loadCam for a non-integer resolution."""
    if resolution in (1, 2, 4, 8):
        return (
            round(width / (resolution_scale * resolution)),
            round(height / (resolution_scale * resolution)),
        )
    if resolution == -1:
        global_down = width / 1600.0 if width > 1600 else 1.0
    else:
        global_down = width / float(resolution)
    scale = float(global_down) * float(resolution_scale)
    return int(width / scale), int(height / scale)


def _stem(path):
    return Path(path).stem


def rewrite_transforms(payload, image_dirname, target_size):
    """Point frames at image_dirname and scale intrinsics onto target_size.

    target_size is (width, height). Idempotent when frames already use that
    directory and the top-level size already matches.
    """
    target_w, target_h = int(target_size[0]), int(target_size[1])
    frames = payload.get("frames") or []
    current_w = int(payload.get("w") or 0)
    current_h = int(payload.get("h") or 0)
    already = (
        current_w == target_w
        and current_h == target_h
        and frames
        and all(
            Path(str(frame.get("file_path", ""))).parts[:1] == (image_dirname,)
            or str(frame.get("file_path", "")).replace("\\", "/").startswith(image_dirname + "/")
            for frame in frames
        )
    )
    if already:
        return payload, False

    if current_w <= 0 or current_h <= 0:
        raise ValueError("transforms.json is missing top-level w/h")
    scale_x = target_w / float(current_w)
    scale_y = target_h / float(current_h)
    rewritten = dict(payload)
    rewritten["w"] = target_w
    rewritten["h"] = target_h
    if "fl_x" in rewritten:
        rewritten["fl_x"] = float(rewritten["fl_x"]) * scale_x
    if "fl_y" in rewritten:
        rewritten["fl_y"] = float(rewritten["fl_y"]) * scale_y
    if "cx" in rewritten:
        rewritten["cx"] = float(rewritten["cx"]) * scale_x
    if "cy" in rewritten:
        rewritten["cy"] = float(rewritten["cy"]) * scale_y

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
            elif key in payload:
                item[key] = float(payload[key]) * scale
        item["w"] = target_w
        item["h"] = target_h
        new_frames.append(item)
    rewritten["frames"] = new_frames
    return rewritten, True


def _read_size(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    return width, height


def _load_depth(path):
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    return raw


def export_scene(data_root, images_dirname, depths_dirname, depth_params_path, out_dirname, hold, overwrite, rewrite):
    data_root = Path(data_root)
    images_dir = data_root / images_dirname
    depths_dir = data_root / depths_dirname
    params_path = data_root / depth_params_path if not Path(depth_params_path).is_absolute() else Path(depth_params_path)
    out_dir = data_root / out_dirname
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    if not depths_dir.is_dir():
        raise FileNotFoundError(depths_dir)
    params = json.loads(params_path.read_text(encoding="utf-8"))

    image_paths = sorted(
        [path for path in images_dir.iterdir() if path.is_file() and path.suffix in IMAGE_EXTS],
        key=lambda path: path.name,
    )
    if not image_paths:
        raise FileNotFoundError(f"no images in {images_dir}")

    target_size = _read_size(image_paths[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    missing_depth = []
    missing_params = []
    unreliable = []
    scales = [float(item["scale"]) for item in params.values() if float(item.get("scale", 0)) > 0]
    med_scale = float(np.median(scales)) if scales else 0.0

    import cv2

    for image_path in image_paths:
        stem = _stem(image_path)
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
        if scale <= 0 or (med_scale > 0 and (scale < 0.2 * med_scale or scale > 5 * med_scale)):
            unreliable.append(stem)
        if dest.is_file() and not overwrite:
            skipped += 1
            continue
        raw = _load_depth(depth_path)
        inverse = normalize_inverse_depth(raw)
        aligned = apply_scale_offset(inverse, scale, offset)
        metric = inverse_to_metric(aligned)
        width, height = _read_size(image_path)
        if (metric.shape[1], metric.shape[0]) != (width, height):
            metric = cv2.resize(metric, (width, height), interpolation=cv2.INTER_LINEAR)
        np.save(dest, metric.astype(np.float32))
        written += 1

    names = [path.name for path in image_paths]
    holdout = holdout_names(names, hold)
    manifest = {
        "images_dirname": images_dirname,
        "target_size_wh": list(target_size),
        "tags_train_resolution_wh": list(tags_train_resolution(target_size[0], target_size[1], 1)),
        "num_images": len(image_paths),
        "num_holdout": len(holdout),
        "hold": hold,
        "holdout_names": holdout,
        "written": written,
        "skipped_existing": skipped,
        "missing_depth": missing_depth,
        "missing_params": missing_params,
        "unreliable_stems": unreliable,
        "med_scale": med_scale,
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    transforms_changed = False
    if rewrite:
        transforms_path = data_root / "transforms.json"
        if not transforms_path.is_file():
            raise FileNotFoundError(transforms_path)
        payload = json.loads(transforms_path.read_text(encoding="utf-8"))
        updated, transforms_changed = rewrite_transforms(payload, images_dirname, target_size)
        if transforms_changed:
            backup = data_root / "transforms.fullres.json"
            if not backup.is_file():
                shutil.copy2(transforms_path, backup)
            transforms_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        manifest["transforms_rewritten"] = transforms_changed
        (out_dir / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    return manifest


def check_scene(data_root, images_dirname, out_dirname, hold):
    data_root = Path(data_root)
    images_dir = data_root / images_dirname
    out_dir = data_root / out_dirname
    image_names = sorted(
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix in IMAGE_EXTS
    )
    npy_stems = sorted(path.name[: -len("_aligned.npy")] for path in out_dir.glob("*_aligned.npy"))
    image_stems = sorted(_stem(name) for name in image_names)
    missing = sorted(set(image_stems) - set(npy_stems))
    extra = sorted(set(npy_stems) - set(image_stems))
    holdout = holdout_names(image_names, hold)
    report = {
        "num_images": len(image_names),
        "num_aligned_npy": len(npy_stems),
        "num_holdout": len(holdout),
        "holdout_names": holdout,
        "missing_npy_stems": missing,
        "extra_npy_stems": extra,
        "aligned": not missing and not extra and len(npy_stems) == len(image_stems),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/bicycle")
    parser.add_argument("--images", default="images_4")
    parser.add_argument("--depths", default="depth_any_out")
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
        if not report["aligned"]:
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
    print(json.dumps({key: manifest[key] for key in manifest if key != "holdout_names"}, indent=2))
    if manifest["missing_depth"] or manifest["missing_params"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
