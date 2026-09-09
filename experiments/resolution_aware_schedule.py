"""Resolution-aware stage schedule planner for TAGSs v16."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# Median pixels for bicycle images_4 @ resolution=800 (3080 E1b reference protocol).
DEFAULT_REF_PIXELS = 426_400


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _round_iter(value: float) -> int:
    return max(1, int(round(value)))


@dataclass
class ResolutionProfile:
    cur_pixels: int
    ref_pixels: int
    resolution_scale: float
    median_width: int = 0
    median_height: int = 0


@dataclass
class StagePlan:
    cur_pixels: int
    ref_pixels: int
    resolution_scale: float
    total_iterations: int
    densify_until_iter: int
    geo_prior_warmup_until: int
    geo_prior_decay_start: int
    geo_prior_decay_until: int
    spa_start_iter: int
    spa_end_iter: int
    lpips_loss_start_iter: int
    psnr_loss_start_iter: int
    color_align_start_iter: int
    lr_stage1_end: int
    lr_stage2_end: int
    geometry_start_iter: int
    resolution_schedule: str
    checkpoint_iterations: List[int] = field(default_factory=list)
    late_controller_structure_end_iter: Optional[int] = None
    late_controller_handoff_end_iter: Optional[int] = None
    late_controller_refine_end_iter: Optional[int] = None
    late_controller_lpips_guard_start_iter: Optional[int] = None
    before: Dict[str, Any] = field(default_factory=dict)
    runtime_overrides: List[Dict[str, Any]] = field(default_factory=list)


class ResolutionProfiler:
    @staticmethod
    def median_pixels_from_cameras(cameras: Sequence) -> ResolutionProfile:
        if not cameras:
            raise ValueError("Cannot profile resolution from an empty camera list.")
        widths = sorted(int(getattr(cam, "image_width", 0)) for cam in cameras)
        heights = sorted(int(getattr(cam, "image_height", 0)) for cam in cameras)
        pixels = sorted(int(w * h) for w, h in zip(widths, heights))
        mid = len(pixels) // 2
        return ResolutionProfile(
            cur_pixels=int(pixels[mid]),
            ref_pixels=DEFAULT_REF_PIXELS,
            resolution_scale=1.0,
            median_width=int(widths[mid]),
            median_height=int(heights[mid]),
        )

    @staticmethod
    def profile_scene(scene, scale: float = 1.0, ref_pixels: int = DEFAULT_REF_PIXELS) -> ResolutionProfile:
        cameras = scene.getTrainCameras(scale)
        profile = ResolutionProfiler.median_pixels_from_cameras(cameras)
        profile.ref_pixels = int(ref_pixels) if int(ref_pixels) > 0 else DEFAULT_REF_PIXELS
        ratio = float(profile.cur_pixels) / float(profile.ref_pixels)
        profile.resolution_scale = _clamp(math.sqrt(max(ratio, 1e-6)), 1.0, 2.5)
        return profile

    @staticmethod
    def compute_scale(
        cur_pixels: int,
        ref_pixels: int = DEFAULT_REF_PIXELS,
        scale_min: float = 1.0,
        scale_max: float = 2.5,
    ) -> float:
        ref = int(ref_pixels) if int(ref_pixels) > 0 else DEFAULT_REF_PIXELS
        ratio = float(cur_pixels) / float(ref)
        return _clamp(math.sqrt(max(ratio, 1e-6)), scale_min, scale_max)


class SchedulePlanner:
    @staticmethod
    def plan(
        profile: ResolutionProfile,
        total_iterations: int,
        *,
        densify_base: int = 7200,
        densify_min: int = 10_000,
        densify_max: int = 18_000,
        spa_delay: int = 2000,
        scale_min: float = 1.0,
        scale_max: float = 2.5,
        enable_balanced_late_controller: bool = False,
        before: Optional[Dict[str, Any]] = None,
    ) -> StagePlan:
        scale = _clamp(profile.resolution_scale, scale_min, scale_max)
        total = int(total_iterations)
        tail_margin = 4000

        if scale <= 1.0 + 1e-6:
            densify_until = int(densify_base)
        else:
            densify_until = _round_iter(densify_base * scale)
            densify_until = _round_iter(_clamp(densify_until, densify_min, densify_max))

        geo_warmup_until = min(_round_iter(0.4 * densify_until), 7000)
        geo_decay_start = max(densify_until, 12_000)
        spa_start = densify_until + int(spa_delay)
        spa_end = min(total - tail_margin, spa_start + 8000)
        spa_end = max(spa_start + 1000, spa_end)
        geo_decay_until = min(total - tail_margin, spa_end)

        psnr_start = max(_round_iter(0.70 * total), spa_start + 3000)
        lpips_start = max(_round_iter(0.75 * total), spa_end - 1000)
        color_align_start = spa_end
        lr_stage1_end = 6000
        lr_stage2_end = max(densify_until, 18_000)
        geometry_start = max(9000, _round_iter(0.55 * densify_until))

        resolution_schedule = SchedulePlanner._scale_resolution_schedule(total, scale)
        checkpoints = SchedulePlanner._build_checkpoints(
            total=total,
            densify_until=densify_until,
            spa_start=spa_start,
            spa_end=spa_end,
        )

        late_structure = late_handoff = late_refine = late_lpips_guard = None
        if enable_balanced_late_controller:
            late_structure = min(total - 6000, max(spa_end + 500, _round_iter(24_000 * min(scale, 1.5))))
            late_handoff = min(total - 4500, late_structure + 1500)
            late_refine = min(total - 1500, max(_round_iter(0.95 * total), late_handoff + 3000))
            late_lpips_guard = max(spa_end, min(late_refine - 4000, _round_iter(24_500 * min(scale, 1.2))))

        return StagePlan(
            cur_pixels=int(profile.cur_pixels),
            ref_pixels=int(profile.ref_pixels),
            resolution_scale=float(scale),
            total_iterations=total,
            densify_until_iter=densify_until,
            geo_prior_warmup_until=geo_warmup_until,
            geo_prior_decay_start=geo_decay_start,
            geo_prior_decay_until=geo_decay_until,
            spa_start_iter=spa_start,
            spa_end_iter=spa_end,
            lpips_loss_start_iter=lpips_start,
            psnr_loss_start_iter=psnr_start,
            color_align_start_iter=color_align_start,
            lr_stage1_end=lr_stage1_end,
            lr_stage2_end=lr_stage2_end,
            geometry_start_iter=geometry_start,
            resolution_schedule=resolution_schedule,
            checkpoint_iterations=checkpoints,
            late_controller_structure_end_iter=late_structure,
            late_controller_handoff_end_iter=late_handoff,
            late_controller_refine_end_iter=late_refine,
            late_controller_lpips_guard_start_iter=late_lpips_guard,
            before=dict(before or {}),
        )

    @staticmethod
    def _scale_resolution_schedule(total_iterations: int, scale: float) -> str:
        first = max(1, _round_iter(1200 * scale))
        second = max(first + 1, _round_iter(4000 * scale))
        if second >= total_iterations:
            second = max(first + 1, total_iterations // 2)
        return f"{first}:4,{second}:2,{total_iterations}:1"

    @staticmethod
    def _build_checkpoints(
        *,
        total: int,
        densify_until: int,
        spa_start: int,
        spa_end: int,
    ) -> List[int]:
        anchors = [
            7000,
            densify_until,
            8000,
            10_000,
            12_000,
            14_000,
            15_000,
            16_000,
            17_000,
            18_000,
            20_000,
            24_000,
            27_000,
            28_500,
            spa_start,
            spa_end,
            total,
        ]
        return sorted({int(i) for i in anchors if 1 <= int(i) <= total})


def snapshot_schedule(opt, cfg) -> Dict[str, Any]:
    fields = [
        "densify_until_iter",
        "geo_prior_warmup_until",
        "geo_prior_decay_start",
        "geo_prior_decay_until",
        "spa_start_iter",
        "spa_end_iter",
        "lpips_loss_start_iter",
        "psnr_loss_start_iter",
        "color_align_start_iter",
        "lr_stage1_end",
        "lr_stage2_end",
        "geometry_start_iter",
        "resolution_schedule",
        "late_controller_structure_end_iter",
        "late_controller_handoff_end_iter",
        "late_controller_refine_end_iter",
        "late_controller_lpips_guard_start_iter",
        "live_stage_iterations",
    ]
    snap: Dict[str, Any] = {"densify_until_iter": int(getattr(opt, "densify_until_iter", 0))}
    for name in fields[1:]:
        if hasattr(cfg, name):
            snap[name] = getattr(cfg, name)
    return snap


def apply_stage_plan(plan: StagePlan, opt, cfg, args=None) -> None:
    opt.densify_until_iter = int(plan.densify_until_iter)
    cfg.geo_prior_warmup_until = int(plan.geo_prior_warmup_until)
    cfg.geo_prior_decay_start = int(plan.geo_prior_decay_start)
    cfg.geo_prior_decay_until = int(plan.geo_prior_decay_until)
    cfg.spa_start_iter = int(plan.spa_start_iter)
    cfg.spa_end_iter = int(plan.spa_end_iter)
    cfg.lpips_loss_start_iter = int(plan.lpips_loss_start_iter)
    cfg.psnr_loss_start_iter = int(plan.psnr_loss_start_iter)
    cfg.color_align_start_iter = int(plan.color_align_start_iter)
    cfg.lr_stage1_end = int(plan.lr_stage1_end)
    cfg.lr_stage2_end = int(plan.lr_stage2_end)
    cfg.geometry_start_iter = int(plan.geometry_start_iter)
    cfg.resolution_schedule = str(plan.resolution_schedule)

    if plan.late_controller_structure_end_iter is not None:
        cfg.late_controller_structure_end_iter = int(plan.late_controller_structure_end_iter)
        cfg.late_controller_handoff_end_iter = int(plan.late_controller_handoff_end_iter)
        cfg.late_controller_refine_end_iter = int(plan.late_controller_refine_end_iter)
        cfg.late_controller_lpips_guard_start_iter = int(plan.late_controller_lpips_guard_start_iter)

    live_spec = ",".join(str(i) for i in plan.checkpoint_iterations)
    cfg.live_stage_iterations = live_spec

    if args is not None:
        merged_test = sorted(set(getattr(args, "test_iterations", [])) | set(plan.checkpoint_iterations))
        merged_save = sorted(set(getattr(args, "save_iterations", [])) | set(plan.checkpoint_iterations))
        args.test_iterations = merged_test
        args.save_iterations = merged_save
        args.live_stage_iterations = live_spec
        args.densify_until_iter = int(plan.densify_until_iter)
        args.geo_prior_warmup_until = int(plan.geo_prior_warmup_until)
        args.geo_prior_decay_start = int(plan.geo_prior_decay_start)
        args.geo_prior_decay_until = int(plan.geo_prior_decay_until)
        args.spa_start_iter = int(plan.spa_start_iter)
        args.spa_end_iter = int(plan.spa_end_iter)
        args.lpips_loss_start_iter = int(plan.lpips_loss_start_iter)
        args.psnr_loss_start_iter = int(plan.psnr_loss_start_iter)
        args.color_align_start_iter = int(plan.color_align_start_iter)
        args.lr_stage1_end = int(plan.lr_stage1_end)
        args.lr_stage2_end = int(plan.lr_stage2_end)
        args.geometry_start_iter = int(plan.geometry_start_iter)
        args.resolution_schedule = str(plan.resolution_schedule)
        if plan.late_controller_structure_end_iter is not None:
            args.late_controller_structure_end_iter = int(plan.late_controller_structure_end_iter)
            args.late_controller_handoff_end_iter = int(plan.late_controller_handoff_end_iter)
            args.late_controller_refine_end_iter = int(plan.late_controller_refine_end_iter)
            args.late_controller_lpips_guard_start_iter = int(plan.late_controller_lpips_guard_start_iter)


def dump_schedule_plan(model_path: str, plan: StagePlan) -> str:
    os.makedirs(model_path, exist_ok=True)
    out_path = os.path.join(model_path, "schedule_plan.json")
    payload = asdict(plan)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return out_path


def append_runtime_override(model_path: str, override: Dict[str, Any]) -> None:
    out_path = os.path.join(model_path, "schedule_plan.json")
    if not os.path.isfile(out_path):
        return
    with open(out_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    runtime = payload.setdefault("runtime_overrides", [])
    runtime.append(override)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def print_stage_plan(plan: StagePlan) -> None:
    print("[RAS] Resolution-aware schedule plan:")
    print(f"  pixels: cur={plan.cur_pixels} ref={plan.ref_pixels} scale={plan.resolution_scale:.3f}")
    print(f"  densify_until={plan.densify_until_iter} spa={plan.spa_start_iter}-{plan.spa_end_iter}")
    print(
        f"  geo: warmup<={plan.geo_prior_warmup_until} decay {plan.geo_prior_decay_start}->{plan.geo_prior_decay_until}"
    )
    print(
        "  late losses: "
        f"psnr@{plan.psnr_loss_start_iter} lpips@{plan.lpips_loss_start_iter} "
        f"color@{plan.color_align_start_iter}"
    )
    print(f"  resolution_schedule={plan.resolution_schedule}")
    print(f"  checkpoints={plan.checkpoint_iterations}")
