#!/usr/bin/env python3
"""Build an upload-ready ZIP for the DSINE external-normal training patch."""

from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


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

README = """TAGS DSINE external-normal patch

1. Upload this ZIP to the AutoDL machine and extract it to a temporary directory.
2. Backups are created automatically by the installer.
3. Run:
   python install_dsine_external_normals_patch.py --target /root/autodl-tmp/TAGS_github
4. Verify the CLI:
   cd /root/autodl-tmp/TAGS_github
   python train_optimized.py --help 2>&1 | grep -E 'normal_prior_source|normal_maps_dir'
5. Only after the CLI check, run the short smoke command described in
   docs/DSINE_EXTERNAL_NORMALS_UPLOAD.md.

The bundle replaces only the 11 listed TAGS source files. It does not contain
datasets, model weights, or generated normal maps, and it does not start training.
"""


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "dsine_external_normals_patch.zip"

    entries = [(repo_root / rel, "files/" + rel) for rel in PATCH_FILES]
    entries.extend([
        (repo_root / "scripts" / "install_dsine_external_normals_patch.py",
         "install_dsine_external_normals_patch.py"),
        (repo_root / "docs" / "DSINE_EXTERNAL_NORMALS_UPLOAD.md", "DSINE_EXTERNAL_NORMALS_UPLOAD.md"),
    ])
    for path, _ in entries:
        if not path.is_file():
            raise FileNotFoundError(path)

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", README)
        for path, arcname in entries:
            zf.write(path, arcname)

    digest = sha256(archive_path.read_bytes()).hexdigest()
    print("Created:", archive_path)
    print("Size:", archive_path.stat().st_size, "bytes")
    print("SHA256:", digest)


if __name__ == "__main__":
    main()
