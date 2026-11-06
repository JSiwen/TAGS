import os
import sys
import uuid
import torch
import torch.distributed as dist
from argparse import ArgumentParser, Namespace

from arguments import ModelParams, PipelineParams, OptimizationParams, ExperimentParams
from gaussian_renderer import network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from experiments import ExperimentConfig, TrainingEngine
from experiments.checkpoint_io import load_training_checkpoint
from experiments.resolution_schedule import ResolutionSchedule
from experiments.resolution_aware_schedule import (
    ResolutionProfiler,
    SchedulePlanner,
    apply_stage_plan,
    dump_schedule_plan,
    print_stage_plan,
    snapshot_schedule,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def prepare_output_and_logger(args, is_main_process=True):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") if os.getenv("OAR_JOB_ID") else str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    if not is_main_process:
        return None

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available: not logging progress")
    return None


def setup_distributed(args):
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_distributed = bool(args.distributed or env_world_size > 1)
    if not use_distributed:
        return {"enabled": False, "rank": 0, "local_rank": 0, "world_size": 1}

    if not args.model_path:
        raise RuntimeError("Distributed mode requires an explicit --model_path so all ranks write to the same output.")

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        raise RuntimeError("Distributed mode requested but WORLD_SIZE<=1. Please launch with torchrun --nproc_per_node=8.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method="env://")
    return {"enabled": True, "rank": rank, "local_rank": local_rank, "world_size": world_size}


def main():
    parser = ArgumentParser(description="3DGS optimized training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ep = ExperimentParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument(
        "--no_save_schedule_point_clouds",
        action="store_true",
        default=False,
        help="Do not save PLYs at resolution-aware schedule checkpoints; explicit --save_iterations still apply.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    # ===== v3 Group A: 记录训练元数据 (B 改动, 2026-10-19) =====
    # 在 "Training complete." 前由 TrainingEngine.run() 写 runtime.json
    # runner 后续从 runtime.json 解析 wall_time + peak_vram, 写 summary.json

    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument(
        "--ddp_allow_sparse_geometry",
        action="store_true",
        default=False,
        help="Allow sparse optimization and geometry controller in DDP (unsafe; may cause instability).",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    dist_info = setup_distributed(args)
    is_main_process = dist_info["rank"] == 0

    if not SPARSE_ADAM_AVAILABLE and args.optimizer_type == "sparse_adam":
        sys.exit("Trying to use sparse adam but it is not installed, please install the correct rasterizer with 3dgs_accel.")

    if is_main_process:
        print("Optimizing " + args.model_path)
    safe_state(args.quiet, cuda_device=dist_info["local_rank"], seed=int(getattr(args, "seed", 0)))
    if is_main_process:
        print(f"[RNG] seed={int(getattr(args, 'seed', 0))}")
    if (not dist_info["enabled"]) and (not args.disable_viewer):
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    cfg = ExperimentConfig.from_args(ep.extract(args))

    if dist_info["enabled"] and not args.ddp_allow_sparse_geometry:
        # DDP safety mode: topology-changing prune controllers can diverge across
        # ranks and cause collapse. Keep core training while disabling risky parts.
        if is_main_process:
            print("[DDP-SAFE] Disabling sparse optimization and geometry controller for stability.")
        cfg.enable_sparse_optimization = False
        cfg.enable_geometry_controller = False

    tb_writer = prepare_output_and_logger(args, is_main_process=is_main_process)
    if dist_info["enabled"]:
        dist.barrier(device_ids=[dist_info["local_rank"]])
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)

    resolution_scales = [1.0]
    if cfg.enable_resolution_adaptive or cfg.enable_resolution_aware_schedule:
        if cfg.enable_resolution_adaptive:
            schedule = ResolutionSchedule(cfg.resolution_schedule)
            resolution_scales = sorted(set(schedule.scales() + [1.0]))
        if cfg.enable_resolution_aware_schedule:
            resolution_scales = sorted(set(resolution_scales + [1.0, 2.0, 4.0]))
        if is_main_process:
            print(f"[Resolution] Preloading scales: {resolution_scales}")

    scene = Scene(dataset, gaussians, resolution_scales=resolution_scales)
    gaussians.training_setup(opt)
    initial_group_lrs = {g["name"]: float(g["lr"]) for g in gaussians.optimizer.param_groups}
    first_iter = 0
    resume_state = None
    if args.start_checkpoint:
        map_location = torch.device(f"cuda:{dist_info['local_rank']}")
        loaded = load_training_checkpoint(torch, args.start_checkpoint, map_location=map_location)
        if isinstance(loaded, dict) and loaded.get("version") == 2:
            resume_state = loaded
            model_params, first_iter = loaded["model"], loaded["iteration"]
        else:
            if cfg.final_target_gaussians > 0:
                raise ValueError("Fixed-budget resume requires a version 2 full-state checkpoint; retrain from scratch")
            print("[Resume] Legacy model-only checkpoint: RNG/camera/controller continuity unavailable")
            model_params, first_iter = loaded
        gaussians.restore(model_params, opt)
        if opt.max_gaussians > 0 and gaussians.get_xyz.shape[0] > opt.max_gaussians:
            raise ValueError(
                f"Checkpoint contains {gaussians.get_xyz.shape[0]} Gaussians, exceeding "
                f"--max_gaussians {opt.max_gaussians}. E2 budget runs must start from "
                "scratch (or from a checkpoint already within the same budget)."
            )

    if cfg.enable_resolution_aware_schedule:
        before = snapshot_schedule(opt, cfg)
        profile = ResolutionProfiler.profile_scene(
            scene,
            scale=1.0,
            ref_pixels=int(getattr(cfg, "ras_ref_pixels", 0)),
        )
        profile.resolution_scale = ResolutionProfiler.compute_scale(
            profile.cur_pixels,
            ref_pixels=int(getattr(cfg, "ras_ref_pixels", 0)),
            scale_min=float(getattr(cfg, "ras_scale_min", 1.0)),
            scale_max=float(getattr(cfg, "ras_scale_max", 2.5)),
        )
        stage_plan = SchedulePlanner.plan(
            profile,
            int(opt.iterations),
            densify_base=int(getattr(cfg, "ras_densify_base", 7200)),
            densify_min=int(getattr(cfg, "ras_densify_min", 10_000)),
            densify_max=int(getattr(cfg, "ras_densify_max", 18_000)),
            spa_delay=int(getattr(cfg, "ras_spa_delay", 2000)),
            scale_min=float(getattr(cfg, "ras_scale_min", 1.0)),
            scale_max=float(getattr(cfg, "ras_scale_max", 2.5)),
            enable_balanced_late_controller=bool(getattr(cfg, "enable_balanced_late_controller", False)),
            before=before,
        )
        apply_stage_plan(stage_plan, opt, cfg, args)
        if is_main_process:
            print_stage_plan(stage_plan)
            plan_path = dump_schedule_plan(args.model_path, stage_plan)
            print(f"[RAS] Wrote schedule plan: {plan_path}")
            with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
                cfg_log_f.write(str(Namespace(**vars(args))))
        if dist_info["enabled"]:
            dist.barrier(device_ids=[dist_info["local_rank"]])

    try:
        if is_main_process:
            save_steps = sorted(set(int(i) for i in args.save_iterations))
            print(
                f"[PLY] save_iterations={save_steps}; "
                f"no_save_schedule_point_clouds={bool(args.no_save_schedule_point_clouds)}"
            )
        engine = TrainingEngine(
            dataset=dataset,
            opt=opt,
            pipe=pipe,
            cfg=cfg,
            scene=scene,
            gaussians=gaussians,
            tb_writer=tb_writer,
            sparse_adam_available=SPARSE_ADAM_AVAILABLE,
            distributed=dist_info["enabled"],
            rank=dist_info["rank"],
            world_size=dist_info["world_size"],
        )
        # ===== v3 Group A: 记录训练元数据 (B 改动, 2026-10-19) =====
        # 在 run 前后打点, 训练结束后写 runtime.json (含 wall_time_sec + peak_vram_mb)
        if is_main_process and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            import time as _time
            _t0 = _time.time()
        if resume_state is not None:
            engine.restore_checkpoint(resume_state, fallback_group_lrs=initial_group_lrs)
        engine.run(first_iter, args.test_iterations, args.save_iterations, args.checkpoint_iterations)
        if is_main_process:
            try:
                import time as _time
                import json as _json
                _elapsed = _time.time() - _t0
                _peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                _runtime = {
                    "wall_time_sec": round(_elapsed, 1),
                    "peak_vram_mb": round(_peak_mb, 1),
                    "iterations": int(args.iterations),
                    "seed": int(args.seed),
                }
                _rt_path = os.path.join(args.model_path, "runtime.json")
                with open(_rt_path, "w", encoding="utf-8") as _rt_f:
                    _json.dump(_runtime, _rt_f, indent=2)
                print(f"[B] runtime.json written: wall={_elapsed:.1f}s peak_vram={_peak_mb:.1f}MB")
            except Exception as _exc:
                print(f"[B] runtime.json failed (non-fatal): {_exc!r}")
        if dist_info["enabled"]:
            dist.barrier(device_ids=[dist_info["local_rank"]])
        if is_main_process:
            print("\nTraining complete.")
    finally:
        if dist_info["enabled"] and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
