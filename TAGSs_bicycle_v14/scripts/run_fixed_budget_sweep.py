"""Foreground fixed-budget driver. Dry run by default; never overwrites an output."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = {"R0": (1,1), "P1": (1.25,1), "P2": (1.5,1), "G1": (1,.75), "G2": (1,.5),
         "S0": (1,1), "S1": (1,1), "S2": (1,1)}

def read_config(path):
    node = ast.parse(path.read_text(), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Namespace":
        raise ValueError("Expected cfg_args Namespace")
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords}

def ply_count(path):
    with path.open("rb") as f:
        for _ in range(200):
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError("PLY vertex count missing")

def hash_file(f):
    digest = hashlib.sha256()
    for block in iter(lambda: f.read(1024*1024), b""):
        digest.update(block)
    return digest.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True, help="30k baseline model directory")
    ap.add_argument("--source", required=True)
    ap.add_argument("--depths", required=True)
    ap.add_argument("--case", choices=CASES, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; select a fresh directory")
    cfg = read_config(args.baseline / "cfg_args")
    n0 = ply_count(args.baseline / "point_cloud/iteration_30000/point_cloud.ply")
    if args.case.startswith("S") and args.checkpoint:
        raise ValueError("SPA cases must train from scratch")
    if args.checkpoint and not args.checkpoint.is_file():
        raise ValueError("Checkpoint does not exist")
    if cfg.get("enable_adaptive_stage_controller") or cfg.get("distributed"):
        raise ValueError("Use single GPU with fixed schedules")
    # Reuse only declared CLI fields; never evaluate cfg_args as Python code.
    tree = ast.parse((ROOT / "arguments/__init__.py").read_text(encoding="utf-8"))
    known = {node.attr.lstrip("_") for node in ast.walk(tree)
             if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
             and isinstance(node.value, ast.Name) and node.value.id == "self"}
    cfg = {k:v for k,v in cfg.items() if k in known}
    pm, gs = CASES[args.case]
    cfg.update(source_path=args.source, depths=args.depths, model_path=str(args.output),
               iterations=30000, eval=True, llffhold=8, psnr_final_multiplier=pm, geo_scale=gs,
               final_target_gaussians=n0)
    if args.case.startswith("S"):
        cfg["spa_target_gaussians"] = n0
        cfg["spa_protect_top_fraction"] = {"S0":.05,"S1":.08,"S2":.10}[args.case]
    cmd = [sys.executable, str(ROOT / "train_optimized.py")]
    for key, value in cfg.items():
        if isinstance(value, bool):
            if value: cmd.append("--"+key)
        elif value is not None:
            cmd.extend(["--"+key, str(value)])
    cmd += ["--seed", str(args.seed), "--disable_viewer", "--checkpoint_iterations", "25000", "30000",
            "--save_iterations", "30000", "--test_iterations", "30000"]
    if args.checkpoint: cmd += ["--start_checkpoint", str(args.checkpoint)]
    record = dict(case=args.case, N0=n0, command=cmd, baseline=str(args.baseline),
                  valid_resume="unverified", official_test_used=True)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if not args.execute: return
    args.output.mkdir(parents=True)
    if args.checkpoint:
        with args.checkpoint.open("rb") as f:
            record["checkpoint_sha256"] = hash_file(f)
    (args.output / "experiment_manifest.json").write_text(json.dumps(record,indent=2))
    subprocess.run(cmd, cwd=ROOT, check=True)

if __name__ == "__main__": main()
