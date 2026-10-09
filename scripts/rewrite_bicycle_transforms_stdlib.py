#!/usr/bin/env python3
"""Pure-stdlib shim: rewrite data/bicycle/transforms.json so coolermap loads images_4.

Replicates `rewrite_transforms` from scripts/export_bicycle_mono_depth.py without
numpy/cv2. Does not regenerate mono_depth, only updates transforms.json and
keeps transforms.fullres.json as the original backup (if a 4946x3286 backup is
still present).
"""

import argparse
import json
import shutil
from pathlib import Path


def first_image_size(images_dir: Path) -> tuple[int, int]:
    """Return (width, height) of the first jpg/png in ``images_dir``."""
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with path.open("rb") as fh:
            data = fh.read()
        # PNG: width/height live in IHDR at offset 16 (big endian uint32).
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 24:
            w = int.from_bytes(data[16:20], "big")
            h = int.from_bytes(data[20:24], "big")
            return w, h
        # JPEG: scan markers SOFn (no leading data needed beyond 0xFFC0..0xFFCF, skip 0xFFC4,0xFFC8,0xFFCC).
        if data.startswith(b"\xff\xd8"):
            i = 2
            while i < len(data):
                if data[i] != 0xFF:
                    return 0, 0
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    seg_len = int.from_bytes(data[i + 2:i + 4], "big")
                    h = int.from_bytes(data[i + 5:i + 7], "big")
                    w = int.from_bytes(data[i + 7:i + 9], "big")
                    return w, h
                seg_len = int.from_bytes(data[i + 2:i + 4], "big")
                i += 2 + seg_len
            return 0, 0
        return 0, 0
    raise FileNotFoundError(f"No jpg/png under {images_dir}")


def rewrite(payload: dict, image_dirname: str, target_size: tuple[int, int]) -> tuple[dict, bool]:
    target_w, target_h = int(target_size[0]), int(target_size[1])
    frames = payload.get("frames") or []
    current_w = int(payload.get("w") or 0)
    current_h = int(payload.get("h") or 0)
    already = (
        current_w == target_w
        and current_h == target_h
        and bool(frames)
        and all(
            str(frame.get("file_path", "")).replace("\\", "/").startswith(image_dirname + "/")
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
        new_frames.append(item)
    rewritten["frames"] = new_frames
    return rewritten, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/bicycle")
    parser.add_argument("--images", default="images_4")
    parser.add_argument("--backup", default="transforms.fullres.json")
    args = parser.parse_args()

    data_root = Path(args.data)
    transforms_path = data_root / "transforms.json"
    images_dir = data_root / args.images
    backup_path = data_root / args.backup

    target = first_image_size(images_dir)
    print(f"[rewrite] target_size={target}")

    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    current = (int(payload.get("w") or 0), int(payload.get("h") or 0))
    print(f"[rewrite] current_size={current}")

    updated, changed = rewrite(payload, args.images, target)
    if not changed:
        print("[rewrite] transforms.json already points at images_4 with target size; nothing to do.")
        return

    if backup_path.is_file():
        backup_size = json.loads(backup_path.read_text(encoding="utf-8"))
        bw = int(backup_size.get("w") or 0)
        bh = int(backup_size.get("h") or 0)
        if (bw, bh) != current:
            print(f"[rewrite] WARNING: existing backup at {backup_path} has size {(bw, bh)} != current {current}; not overwriting")
        else:
            print(f"[rewrite] backup already present at {backup_path}")
    else:
        print(f"[rewrite] backing up original transforms.json -> {backup_path}")
        shutil.copy2(transforms_path, backup_path)

    transforms_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    print(f"[rewrite] wrote {transforms_path} ({len(updated.get('frames', []))} frames)")


if __name__ == "__main__":
    main()
