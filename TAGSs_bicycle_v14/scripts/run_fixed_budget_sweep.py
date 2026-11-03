"""Foreground fixed-budget driver. Dry run by default; never overwrites an output."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
CASES = {"R0": (1,1), "P1": (1.25,1), "P2": (1.5,1), "G1": (1,.75), "G2": (1,.5), "PG1": (1.5,.5), "PG1L1": (1.5,.5), "PG1D1": (1.5,.5),
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

def preflight(root, cfg):
    required = {
        "arguments/__init__.py": ("self.psnr_final_multiplier", "self.lpips_final_multiplier", "self.geo_scale", "self.final_target_gaussians", "self.spa_target_gaussians"),
        "experiments/config.py": ("psnr_final_multiplier", "spa_target_gaussians"),
        "experiments/balanced_late_controller.py": ("budget_psnr_multiplier", "budget_lpips_multiplier"),
        "experiments/geo_prior.py": ('getattr(self.cfg, "geo_scale"',),
        "experiments/loss_builder.py": ('train/psnr_loss_weighted',),
        "experiments/sparse_controller.py": ("spa_target_gaussians",),
        "experiments/training_engine.py": ("def capture_checkpoint", "def restore_checkpoint", 'adaptive_stage_controller.load_state_dict'),
        "experiments/adaptive_stage_controller.py": ("def state_dict", "def load_state_dict"),
        "train_optimized.py": ("engine.restore_checkpoint", "initial_group_lrs"),
        "experiments/group_lr_scheduler.py": ("def capture_base_lrs", "def restore_base_lrs"),
    }
    stale = []
    for name, tokens in required.items():
        path = root / name
        if not path.is_file() or any(token not in path.read_text(encoding="utf-8") for token in tokens):
            stale.append(name)
    if stale:
        raise ValueError("Incomplete fixed-budget code update. Update ALL experiment files, not only this driver: " + ", ".join(stale))
    keys = ("position_lr_init", "densify_until_iter", "enable_geo_prior",
            "enable_sparse_optimization", "sparse_mode", "spa_end_iter",
            "lpips_loss_weight_init", "psnr_loss_weight_final", "geo_prior_start_iter")
    missing = [key for key in keys if key not in cfg]
    if missing:
        raise ValueError("Baseline cfg_args lacks TAGS training configuration: " + ", ".join(missing)
                         + ". Use the actual TAGS 30k baseline, not an original 3DGS/model-only cfg_args.")


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
    preflight(ROOT, cfg)
    n0 = ply_count(args.baseline / "point_cloud/iteration_30000/point_cloud.ply")
    if args.case.startswith("S") and args.checkpoint:
        raise ValueError("SPA cases must train from scratch")
    if args.checkpoint and not args.checkpoint.is_file():
        raise ValueError("Checkpoint does not exist")
    if cfg.get("distributed"):
        raise ValueError("Use a single-GPU TAGS baseline (distributed=False)")
    test_iterations = cfg.get("test_iterations", [7000, 30000])
    if not isinstance(test_iterations, (list, tuple)) or not all(isinstance(i, int) for i in test_iterations):
        raise ValueError("Invalid baseline test_iterations")
    test_iterations = sorted(set(test_iterations) | {30000})
    # Reuse only declared CLI fields; never evaluate cfg_args as Python code.
    tree = ast.parse((ROOT / "arguments/__init__.py").read_text(encoding="utf-8"))
    known = {node.attr.lstrip("_") for node in ast.walk(tree)
             if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
             and isinstance(node.value, ast.Name) and node.value.id == "self"}
    cfg = {k:v for k,v in cfg.items() if k in known}
    pm, gs = CASES[args.case]
    cfg.update(source_path=args.source, depths=args.depths, model_path=str(args.output),
               iterations=30000, eval=True, llffhold=8, psnr_final_multiplier=pm, geo_scale=gs,
               final_target_gaussians=n0, lpips_final_multiplier=.5 if args.case == "PG1L1" else 1.0)
    if args.case == "PG1D1":
        if float(cfg.get("detail_loss_weight_init", 0)) or float(cfg.get("detail_loss_weight_final", 0)):
            raise ValueError("PG1D1 requires a baseline with detail loss disabled")
        if float(cfg.get("detail_loss_max_weight", .05)) < .01:
            raise ValueError("PG1D1 needs detail_loss_max_weight >= 0.01")
        cfg.update(detail_loss_weight_init=0.0, detail_loss_weight_final=.01,
                   detail_loss_start_iter=27000)
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
            "--save_iterations", "30000", "--test_iterations"] + [str(i) for i in test_iterations]
    if args.checkpoint: cmd += ["--start_checkpoint", str(args.checkpoint)]
    record = dict(case=args.case, N0=n0, command=cmd, baseline=str(args.baseline),
                  valid_resume="unverified", official_test_used=True)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if not args.execute: return
    # Check the exact generated command before creating any experiment output.
    subprocess.run(cmd + ["--help"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    args.output.mkdir(parents=True)
    if args.checkpoint:
        with args.checkpoint.open("rb") as f:
            record["checkpoint_sha256"] = hash_file(f)
    (args.output / "experiment_manifest.json").write_text(json.dumps(record,indent=2))
    subprocess.run(cmd, cwd=ROOT, check=True)

if __name__ == "__main__": main()
