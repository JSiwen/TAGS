"""Build a self-contained offline patch zipapp from a published Git snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "TAGSs_bicycle_v14/"

def git(*args):
    return subprocess.check_output(["git"] + list(args), cwd=str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="codex/fixed-budget-experiments-20261029")
    parser.add_argument("--output", type=Path, default=ROOT / "output/tags_fixed_budget_offline.zip")
    args = parser.parse_args()
    commit = git("rev-parse", args.revision).decode().strip()
    names = git("ls-tree", "-r", "--name-only", commit, "--", PREFIX + "experiments").decode().splitlines()
    names = [name[len(PREFIX):] for name in names if name.endswith(".py")]
    names += ["arguments/__init__.py", "train_optimized.py", "AGENTS.md",
              "scripts/run_fixed_budget_sweep.py", "tests/test_fixed_budget_preflight.py",
              "tests/test_fixed_budget_adaptive_resume.py", "tests/test_fixed_budget_saves.py", "tests/test_fixed_budget_checkpoint_io.py", "tests/test_fixed_budget_lpips.py", "tests/test_fixed_budget_group_lr_resume.py", "tests/test_fixed_budget_floor.py", "tests/test_fixed_budget_sparse_resume.py", "docs/TAGS_FIXED_BUDGET_RUNBOOK.md",
              "docs/TAGS_FIXED_BUDGET_EXPERIMENT_PLAN.md"]
    payload = {name: git("show", commit + ":" + PREFIX + name) for name in names}
    payload["run_fixed_budget_sweep.py"] = (
        '"""Compatibility launcher for the canonical scripts driver."""\n'
        'from pathlib import Path\nimport runpy\n'
        'if __name__ == "__main__":\n'
        '    runpy.run_path(str(Path(__file__).resolve().parent / "scripts/run_fixed_budget_sweep.py"), run_name="__main__")\n'
    ).encode()
    manifest = {"commit": commit, "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    installer = git("show", commit + ":" + PREFIX + "scripts/install_fixed_budget_patch.py")
    with zipfile.ZipFile(str(args.output), "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("__main__.py", installer)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        for name, data in payload.items():
            archive.writestr("payload/" + name, data)
    print(str(args.output.resolve()))
    print("Files: {}  Bytes: {}".format(len(payload), args.output.stat().st_size))
    print("SHA256: " + hashlib.sha256(args.output.read_bytes()).hexdigest())

if __name__ == "__main__":
    main()
