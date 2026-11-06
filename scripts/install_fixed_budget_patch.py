"""Offline zipapp installer: python tags_fixed_budget_offline.zip --target PROJECT_ROOT."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import uuid
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.target.resolve()
    if not (root / "train_optimized.py").is_file():
        raise SystemExit("Target must be an existing TAGS project containing train_optimized.py")
    bundle = Path(sys.argv[0]).resolve()
    with zipfile.ZipFile(str(bundle)) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        pending = []
        for name, digest in manifest["sha256"].items():
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts or "\\" in name:
                raise SystemExit("Invalid patch path: " + name)
            target = root.joinpath(*rel.parts).resolve()
            if root not in target.parents:
                raise SystemExit("Patch path escapes project: " + name)
            data = archive.read("payload/" + name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise SystemExit("SHA256 mismatch: " + name)
            pending.append((target, name, data))
    backup = root / ("fixed_budget_backup_" + uuid.uuid4().hex[:12] + ".zip")
    with zipfile.ZipFile(str(backup), "w", zipfile.ZIP_DEFLATED) as archive:
        for target, name, data in pending:
            if target.is_file():
                archive.write(str(target), name)
    for target, name, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if hashlib.sha256(target.read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise SystemExit("Installed hash mismatch: " + name)
    print("Installed and verified {} files".format(len(pending)))
    print("Code base: " + manifest["commit"])
    print("Backup: " + str(backup))
    print("No training was started. Run scripts/run_fixed_budget_sweep.py next.")


if __name__ == "__main__":
    main()
