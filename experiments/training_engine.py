import random
import os
import json
from random import randint

import torch
import torch.distributed as dist
from tqdm import tqdm

from gaussian_renderer import render
from .evaluator import EvaluationReporter
from .geo_prior import GeometricPriorLoss
from .geometry_controller import GeometryController
from .group_lr_scheduler import GroupLRScheduler
from .loss_builder import LossBuilder
from .balanced_late_controller import BalancedLateController
from .adaptive_stage_controller import AdaptiveStageController
from .resolution_aware_schedule import append_runtime_override
from .resolution_schedule import ResolutionSchedule
from .sparse_controller import SparseController


class TrainingEngine:
    def __init__(
        self,
        dataset,
        opt,
        pipe,
        cfg,
        scene,
        gaussians,
        tb_writer,
        sparse_adam_available,
        distributed=False,
        rank=0,
        world_size=1,
    ):
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe
        self.cfg = cfg
        self.scene = scene
        self.gaussians = gaussians
        self.tb_writer = tb_writer
        self.sparse_adam_available = sparse_adam_available
        self.distributed = bool(distributed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.is_main_process = self.rank == 0

        self.geo_prior = GeometricPriorLoss(cfg)
        self.res_schedule = ResolutionSchedule(cfg.resolution_schedule)
        self.sparse_controller = SparseController(cfg, self.res_schedule)
        self.loss_builder = LossBuilder(opt, cfg, self.geo_prior, self.sparse_controller)
        self.group_lr_scheduler = GroupLRScheduler(cfg)
        self.geometry_controller = GeometryController(cfg)
        self.balanced_late_controller = BalancedLateController(cfg, opt.iterations)
        self.adaptive_stage_controller = AdaptiveStageController(cfg)
        self.fullres_ready_iter = self._fullres_ready_iteration()
        self.effective_densify_until_iter = max(self.opt.densify_until_iter, self.fullres_ready_iter)
        self._last_test_psnr = None
        self._prev_test_psnr = None
        self._ema_photo_loss = None
        self._prev_ema_photo_loss = None

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.evaluator = EvaluationReporter(
            scene=scene,
            render_func=render,
            render_args=(pipe, self.background, 1.0, self.sparse_adam_available, None, dataset.train_test_exp),
            tb_writer=tb_writer,
            train_test_exp=dataset.train_test_exp,
            cfg=cfg,
        )
        self.curr_scale = 1.0
        self.viewpoint_stack = []
        self.viewpoint_indices = []
        self.view_difficulty_ema = {}
        self.view_lpips_proxy_ema = {}
        self.global_lpips_proxy_ema = None
        if cfg.enable_resolution_adaptive:
            print(f"[Resolution] Using schedule: {self.res_schedule}")
            print(f"[Resolution] Full-resolution recovery until iter {self.fullres_ready_iter}")
        if cfg.enable_geo_prior and int(getattr(cfg, "geo_prior_start_iter", 0)) > 0:
            print(f"[GeoPrior] Enabled from iter {int(cfg.geo_prior_start_iter)}")
        if self.distributed and self.is_main_process:
            print(f"[Distributed] Enabled with world_size={self.world_size}")
        if self.effective_densify_until_iter > self.opt.densify_until_iter:
            print(f"[Densify] Extending densification until iter {self.effective_densify_until_iter}")
        if int(getattr(self.opt, "max_gaussians", 0)) > 0:
            print(f"[Densify] Hard Gaussian budget: {int(self.opt.max_gaussians):,}")
        if self.adaptive_stage_controller.enabled():
            print(
                "[AdaptiveStage] Enabled: "
                f"extend_step={self.adaptive_stage_controller.extend_step}, "
                f"max_densify={self.adaptive_stage_controller.max_densify_iter}, "
                f"point_budget={self.adaptive_stage_controller.point_budget}"
            )
        if cfg.enable_hard_view_mining:
            print(
                f"[HardView] Enabled from iter {cfg.hard_view_start_iter}, "
                f"alpha={cfg.hard_view_sample_alpha}, max_boost={cfg.hard_view_boost_max}"
            )
        elif cfg.enable_hard_view_loss_reweight:
            print(
                f"[HardViewLoss] Enabled from iter {cfg.hard_view_loss_reweight_start_iter}, "
                f"max={cfg.hard_view_loss_reweight_max}"
            )
        if self.balanced_late_controller.enabled():
            print(
                "[LateController] Enabled with stages: "
                f"structure<={int(getattr(cfg, 'late_controller_structure_end_iter', 24000))}, "
                f"handoff<={int(getattr(cfg, 'late_controller_handoff_end_iter', 25500))}, "
                f"refine<={int(getattr(cfg, 'late_controller_refine_end_iter', 28500))}"
            )
        self.enable_live_loss_logging = bool(getattr(cfg, "enable_live_loss_logging", False))
        self.live_loss_log_interval = max(1, int(getattr(cfg, "live_loss_log_interval", 1)))
        self.live_stage_iterations = self._parse_live_stage_iterations(
            getattr(cfg, "live_stage_iterations", "")
        )
        self._iter_log_path = None
        self._stage_log_path = None
        self._iter_log_file = None
        self._stage_log_file = None
        self._init_live_logs()

    def _fullres_ready_iteration(self):
        if not self.cfg.enable_resolution_adaptive:
            return 0
        fullres_start = self.res_schedule.full_resolution_start_iter
        if fullres_start is None:
            return 0
        return fullres_start + max(0, int(self.cfg.spa_after_fullres_delay))

    @staticmethod
    def _parse_live_stage_iterations(spec):
        text = str(spec or "").strip()
        if not text:
            return []
        parsed = []
        for token in text.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                parsed.append(int(token))
            except ValueError:
                continue
        return sorted(set(parsed))

    def _init_live_logs(self):
        if not (self.is_main_process and self.enable_live_loss_logging):
            return
        model_path = str(getattr(self.scene, "model_path", "") or "")
        if not model_path:
            return
        log_dir = os.path.join(model_path, "live_logs")
        os.makedirs(log_dir, exist_ok=True)
        self._iter_log_path = os.path.join(log_dir, "train_iter_metrics.jsonl")
        self._stage_log_path = os.path.join(log_dir, "stage_snapshots.jsonl")
        self._iter_log_file = open(self._iter_log_path, "a", encoding="utf-8", buffering=1)
        self._stage_log_file = open(self._stage_log_path, "a", encoding="utf-8", buffering=1)
        print(
            f"[LiveLog] Enabled: interval={self.live_loss_log_interval}, "
            f"iter_log={self._iter_log_path}, stage_log={self._stage_log_path}"
        )

    def _close_live_logs(self):
        if self._iter_log_file is not None:
            self._iter_log_file.close()
            self._iter_log_file = None
        if self._stage_log_file is not None:
            self._stage_log_file.close()
            self._stage_log_file = None

    def _write_jsonl(self, file_obj, payload):
        if file_obj is None:
            return
        file_obj.write(json.dumps(payload, ensure_ascii=True) + "\n")

    @staticmethod
    def _extract_operation_terms(stats):
        loss_terms = {}
        weight_terms = {}
        for key, value in stats.items():
            if key.endswith("_loss") or "penalty" in key:
                loss_terms[key] = float(value)
            if key.endswith("_weight") or key.endswith("_multiplier"):
                weight_terms[key] = float(value)
        return loss_terms, weight_terms

    def _log_iteration_snapshot(self, iteration, viewpoint_cam, stats, train_state):
        if self._iter_log_file is None:
            return
        if iteration % self.live_loss_log_interval != 0:
            return
        late_state = train_state.get("late_controller", {}) if train_state else {}
        loss_terms, weight_terms = self._extract_operation_terms(stats)
        payload = {
            "iteration": int(iteration),
            "view": str(getattr(viewpoint_cam, "image_name", "")),
            "resolution_scale": float(train_state.get("scale", 1.0)) if train_state else 1.0,
            "fullres_ready": bool(train_state.get("fullres_ready", False)) if train_state else False,
            "late_stage": str(late_state.get("stage", "disabled")),
            "total_loss": float(stats.get("train/total_loss", 0.0)),
            "operation_losses": loss_terms,
            "operation_weights": weight_terms,
            "stats": {key: float(value) for key, value in stats.items()},
        }
        self._write_jsonl(self._iter_log_file, payload)

    def _log_stage_snapshot(self, iteration, stats, eval_results):
        if self._stage_log_file is None:
            return
        payload = {
            "iteration": int(iteration),
            "train_snapshot": {key: float(value) for key, value in stats.items()},
            "eval": eval_results or {},
            "num_points": int(self.gaussians.get_xyz.shape[0]),
        }
        self._write_jsonl(self._stage_log_file, payload)

    def _refresh_camera_stack(self, scale):
        self.curr_scale = scale
        self.viewpoint_stack = self.scene.getTrainCameras(scale).copy()
        self.viewpoint_indices = list(range(len(self.viewpoint_stack)))

    def _pick_camera(self, iteration):
        if self.cfg.enable_resolution_adaptive:
            scale = self.res_schedule.get_scale(iteration)
        else:
            scale = 1.0
        if (not self.viewpoint_stack) or scale != self.curr_scale:
            if scale != self.curr_scale:
                print(f"\n[ITER {iteration}] Switching training resolution scale to {scale}")
            self._refresh_camera_stack(scale)
        sample_weight = 1.0
        hard_start = int(getattr(self.cfg, "hard_view_start_iter", 0))
        if bool(getattr(self.cfg, "hard_view_require_fullres", True)):
            hard_start = max(hard_start, int(self.fullres_ready_iter))
        use_hard_view = (
            bool(getattr(self.cfg, "enable_hard_view_mining", False))
            and iteration >= hard_start
            and len(self.view_difficulty_ema) > 0
        )
        if use_hard_view:
            alpha = max(0.0, float(getattr(self.cfg, "hard_view_sample_alpha", 1.5)))
            max_boost = max(1.0, float(getattr(self.cfg, "hard_view_boost_max", 3.0)))
            uniform_mix = min(1.0, max(0.0, float(getattr(self.cfg, "hard_view_uniform_mix", 0.5))))
            ramp_iters = max(1, int(getattr(self.cfg, "hard_view_ramp_iters", 6000)))
            ramp = min(1.0, max(0.0, float(iteration - hard_start) / float(ramp_iters)))
            weights = []
            raw_scores = []
            for cam in self.viewpoint_stack:
                raw_scores.append(float(self.view_difficulty_ema.get(cam.image_name, 1.0)))
            sorted_scores = sorted(raw_scores)
            median = sorted_scores[len(sorted_scores) // 2] if sorted_scores else 1.0
            median = max(1e-4, float(median))
            for base in raw_scores:
                relative = max(1e-4, base / median)
                weighted = min(max_boost, relative ** alpha)
                blended = (1.0 - uniform_mix * ramp) * weighted + (uniform_mix * ramp)
                weights.append(max(1e-4, blended))
            idx = random.choices(range(len(self.viewpoint_indices)), weights=weights, k=1)[0]
            sample_weight = float(weights[idx])
        else:
            idx = randint(0, len(self.viewpoint_indices) - 1)
        viewpoint_cam = self.viewpoint_stack.pop(idx)
        self.viewpoint_indices.pop(idx)
        return viewpoint_cam, sample_weight

    def _update_view_difficulty(self, viewpoint_cam, stats, iteration):
        if not (
            bool(getattr(self.cfg, "enable_hard_view_mining", False))
            or bool(getattr(self.cfg, "enable_hard_view_loss_reweight", False))
        ):
            difficulty_enabled = False
        else:
            difficulty_enabled = True

        if difficulty_enabled:
            l1_term = float(stats.get("train/l1", 0.0))
            ssim_term = 1.0 - float(stats.get("train/ssim", 0.0))
            ssim_w = max(0.0, float(getattr(self.cfg, "hard_view_ssim_weight", 0.5)))
            psnr_value = float(stats.get("train/psnr", 0.0))
            psnr_term = 10.0 ** (-psnr_value / 20.0) if psnr_value > 0 else 1.0
            psnr_w = max(0.0, float(getattr(self.cfg, "hard_view_psnr_weight", 0.10)))
            difficulty = max(1e-4, l1_term + ssim_w * ssim_term + psnr_w * psnr_term)
            momentum = min(0.999, max(0.0, float(getattr(self.cfg, "hard_view_ema_momentum", 0.90))))
            name = viewpoint_cam.image_name
            prev = self.view_difficulty_ema.get(name, difficulty)
            self.view_difficulty_ema[name] = momentum * prev + (1.0 - momentum) * difficulty

        # Keep a lightweight per-view LPIPS proxy EMA for late-stage guarding.
        proxy_interval = max(1, int(getattr(self.cfg, "late_controller_lpips_proxy_interval", 16)))
        if (not self.balanced_late_controller.enabled()) or (iteration % proxy_interval != 0):
            return
        lpips_proxy = float(stats.get("train/lpips_proxy", 0.0))
        lpips_momentum = min(
            0.999,
            max(0.0, float(getattr(self.cfg, "late_controller_lpips_guard_ema_momentum", 0.95))),
        )
        name = viewpoint_cam.image_name
        prev_proxy = self.view_lpips_proxy_ema.get(name, lpips_proxy)
        current_proxy = lpips_momentum * prev_proxy + (1.0 - lpips_momentum) * lpips_proxy
        self.view_lpips_proxy_ema[name] = current_proxy
        if self.global_lpips_proxy_ema is None:
            self.global_lpips_proxy_ema = current_proxy
        else:
            self.global_lpips_proxy_ema = (
                lpips_momentum * self.global_lpips_proxy_ema
                + (1.0 - lpips_momentum) * current_proxy
            )

    def _view_loss_weight(self, viewpoint_cam, iteration):
        if not bool(getattr(self.cfg, "enable_hard_view_loss_reweight", False)):
            return 1.0
        start_iter = int(getattr(self.cfg, "hard_view_loss_reweight_start_iter", 18000))
        if iteration < start_iter or len(self.view_difficulty_ema) < 4:
            return 1.0
        scores = sorted(float(v) for v in self.view_difficulty_ema.values())
        median = scores[len(scores) // 2]
        median = max(1e-4, median)
        curr = float(self.view_difficulty_ema.get(viewpoint_cam.image_name, median))
        relative = max(0.0, curr / median - 1.0)
        cap = max(1.0, float(getattr(self.cfg, "hard_view_loss_reweight_max", 1.12)))
        # Mild boost only; avoid destabilizing easy views.
        return min(cap, 1.0 + 0.30 * relative)

    def _sync_tensor_(self, tensor, op="sum"):
        if not self.distributed or self.world_size <= 1:
            return
        if op == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(float(self.world_size))
        elif op == "max":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unsupported sync op: {op}")

    def _sync_optimizer_grads(self):
        if not self.distributed or self.world_size <= 1:
            return
        for optimizer in (self.gaussians.optimizer, self.gaussians.exposure_optimizer):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    # NCCL all_reduce requires contiguous tensors.
                    grad = param.grad
                    if not grad.is_contiguous():
                        reduced = grad.contiguous()
                        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                        reduced.div_(float(self.world_size))
                        grad.copy_(reduced)
                    else:
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                        grad.div_(float(self.world_size))

    def run(self, first_iter, testing_iterations, saving_iterations, checkpoint_iterations):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        use_sparse_adam = self.opt.optimizer_type == "sparse_adam" and self.sparse_adam_available
        fullres_densify_mult = max(1, int(os.environ.get("TAGGS_DENSIFY_INTERVAL_MULT_FULLRES", "1")))
        fullres_densify_point_threshold = max(0, int(os.environ.get("TAGGS_DENSIFY_INTERVAL_POINT_THRESHOLD", "0")))
        if self.distributed and use_sparse_adam:
            raise RuntimeError("Distributed training currently does not support sparse_adam. Please use optimizer_type=default.")
        if self.is_main_process and fullres_densify_mult > 1:
            print(
                "[Densify] Full-res interval multiplier="
                f"{fullres_densify_mult} (point threshold={fullres_densify_point_threshold})"
            )

        ema_loss_for_log = 0.0
        progress_bar = (
            tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
            if self.is_main_process
            else None
        )
        stage_snapshot_iters = set(int(i) for i in testing_iterations)
        stage_snapshot_iters.update(int(i) for i in self.live_stage_iterations)
        if self.is_main_process and self.enable_live_loss_logging and stage_snapshot_iters:
            print(f"[LiveLog] Stage snapshots at: {sorted(stage_snapshot_iters)}")

        first_iter += 1
        try:
            for iteration in range(first_iter, self.opt.iterations + 1):
                iter_start.record()
                self.gaussians.update_learning_rate(iteration)
                lr_stats = self.group_lr_scheduler.apply(iteration, self.gaussians)
                if iteration % 1000 == 0:
                    self.gaussians.oneupSHdegree()

                viewpoint_cam, view_sampling_weight = self._pick_camera(iteration)
                bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
                render_pkg = render(
                    viewpoint_cam,
                    self.gaussians,
                    self.pipe,
                    bg,
                    use_trained_exp=self.dataset.train_test_exp,
                    separate_sh=self.sparse_adam_available,
                )

                train_state = {
                    "scale": self.curr_scale,
                    "fullres_ready": iteration >= self.fullres_ready_iter,
                    "fullres_ready_iter": self.fullres_ready_iter,
                    "view_sampling_weight": view_sampling_weight,
                    "view_loss_weight": self._view_loss_weight(viewpoint_cam, iteration),
                }
                late_state = self.balanced_late_controller.build_state(
                    iteration=iteration,
                    view_name=viewpoint_cam.image_name,
                    view_lpips_ema=self.view_lpips_proxy_ema,
                    global_lpips_ema=self.global_lpips_proxy_ema,
                )
                train_state["late_controller"] = late_state
                loss, ll1, stats = self.loss_builder(
                    render_pkg,
                    viewpoint_cam,
                    iteration,
                    gaussians=self.gaussians,
                    train_state=train_state,
                )
                self._update_view_difficulty(viewpoint_cam, stats, iteration)
                if "train/photo_loss" in stats:
                    photo = float(stats["train/photo_loss"])
                    self._prev_ema_photo_loss = self._ema_photo_loss
                    self._ema_photo_loss = (
                        photo if self._ema_photo_loss is None else 0.4 * photo + 0.6 * self._ema_photo_loss
                    )
                stats["train/resolution_scale"] = float(self.curr_scale)
                stats["train/fullres_ready"] = 1.0 if train_state["fullres_ready"] else 0.0
                stats["train/view_sampling_weight"] = float(view_sampling_weight)
                stats["train/view_loss_weight"] = float(train_state["view_loss_weight"])
                point_count = int(self.gaussians.get_xyz.shape[0])
                stats["scene/total_points"] = float(point_count)
                max_gaussians = int(getattr(self.opt, "max_gaussians", 0))
                if max_gaussians > 0:
                    stats["scene/gaussian_budget"] = float(max_gaussians)
                    stats["scene/budget_utilization"] = point_count / float(max_gaussians)
                if late_state.get("enabled", False):
                    stage_map = {"structure": 0.0, "handoff": 1.0, "refine": 2.0, "polish": 3.0}
                    stats["train/late_controller_stage"] = stage_map.get(late_state.get("stage"), -1.0)
                    stats["train/late_lpips_guard_multiplier"] = float(
                        late_state.get("lpips_guard_multiplier", 1.0)
                    )
                    stats["train/late_color_multiplier"] = float(
                        late_state.get("color_align_multiplier", 1.0)
                    )
                    stats["train/late_is_chronic_view"] = 1.0 if late_state.get("is_chronic_view", False) else 0.0
                    if self.global_lpips_proxy_ema is not None:
                        stats["train/lpips_proxy_ema_global"] = float(self.global_lpips_proxy_ema)
                    if viewpoint_cam.image_name in self.view_lpips_proxy_ema:
                        stats["train/lpips_proxy_ema_view"] = float(
                            self.view_lpips_proxy_ema[viewpoint_cam.image_name]
                        )
                stats.update(lr_stats)
                loss.backward()
                self._sync_optimizer_grads()
                iter_end.record()
                self._log_iteration_snapshot(iteration, viewpoint_cam, stats, train_state)

                with torch.no_grad():
                    ema_loss_for_log = 0.4 * float(loss.item()) + 0.6 * ema_loss_for_log
                    if self.is_main_process and iteration % 10 == 0:
                        progress = {"Loss": f"{ema_loss_for_log:.7f}"}
                        if "geo/depth_loss" in stats:
                            progress["Depth Loss"] = f"{stats['geo/depth_loss']:.7f}"
                        if "geo/normal_loss" in stats:
                            progress["Normal Loss"] = f"{stats['geo/normal_loss']:.7f}"
                        progress_bar.set_postfix(progress)
                        progress_bar.update(10)
                    if self.is_main_process and iteration == self.opt.iterations:
                        progress_bar.close()

                    eval_results = {}
                    if self.is_main_process:
                        self.evaluator.log_train_scalars(
                            iteration, ll1, loss, iter_start.elapsed_time(iter_end), stats
                        )
                        eval_results = self.evaluator.maybe_evaluate(iteration, testing_iterations)
                        if iteration in stage_snapshot_iters:
                            self._log_stage_snapshot(iteration, stats, eval_results)
                        self._maybe_apply_adaptive_stage_extension(iteration, eval_results, stats)

                if self.is_main_process and iteration in saving_iterations:
                    print(f"\n[ITER {iteration}] Saving Gaussians")
                    self.scene.save(iteration)

                if iteration < self.effective_densify_until_iter:
                    visibility_filter = render_pkg["visibility_filter"]
                    radii = render_pkg["radii"]
                    densify_grad = self.opt.densify_grad_threshold
                    edge_factor = float(getattr(self.cfg, "edge_guided_densify_factor", 1.0))
                    detail_start = int(getattr(self.cfg, "detail_loss_start_iter", 0))
                    if iteration >= detail_start:
                        densify_grad *= max(0.5, min(1.5, edge_factor))
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    self.gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)
                    if self.distributed:
                        # Keep pruning/densify signals consistent across ranks.
                        self._sync_tensor_(self.gaussians.xyz_gradient_accum, op="sum")
                        self._sync_tensor_(self.gaussians.denom, op="sum")
                        self._sync_tensor_(self.gaussians.max_radii2D, op="max")
                    densify_interval = self.opt.densification_interval
                    point_count = int(self.gaussians.get_xyz.shape[0])
                    if (
                        self.curr_scale <= 1.0
                        and fullres_densify_mult > 1
                        and point_count >= fullres_densify_point_threshold
                    ):
                        densify_interval *= fullres_densify_mult
                    if iteration > self.opt.densify_from_iter and iteration % densify_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        try:
                            self.gaussians.densify_and_prune(
                                densify_grad, 0.005, self.scene.cameras_extent, size_threshold, radii
                            )
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(
                                    f"\n[ITER {iteration}] Densify OOM; skip densify this round and continue training."
                                )
                                torch.cuda.empty_cache()
                            else:
                                raise
                    if iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and iteration == self.opt.densify_from_iter
                    ):
                        self.gaussians.reset_opacity()

                sparse_stats = self.sparse_controller.apply(
                    iteration,
                    self.gaussians,
                    self.scene,
                    self.opt,
                    render_pkg,
                    train_state=train_state,
                    sync_fn=self._sync_tensor_,
                )
                if self.tb_writer:
                    for key, value in sparse_stats.items():
                        self.tb_writer.add_scalar(key, value, iteration)
                geometry_stats = self.geometry_controller.apply(iteration, self.gaussians)
                if self.tb_writer:
                    for key, value in geometry_stats.items():
                        self.tb_writer.add_scalar(key, value, iteration)
                if "sparse/spa_pruned" in sparse_stats:
                    remaining = sparse_stats.get("sparse/points", self.gaussians.get_xyz.shape[0])
                    if self.is_main_process:
                        print(f"\n[ITER {iteration}] SPA pruned {int(sparse_stats['sparse/spa_pruned'])} points; remaining {int(remaining)}")

                if iteration < self.opt.iterations:
                    self.gaussians.exposure_optimizer.step()
                    self.gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                    if use_sparse_adam:
                        radii = render_pkg["radii"]
                        visible = radii > 0
                        self.gaussians.optimizer.step(visible, radii.shape[0])
                        self.gaussians.optimizer.zero_grad(set_to_none=True)
                    else:
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none=True)

                if self.is_main_process and iteration in checkpoint_iterations:
                    print(f"\n[ITER {iteration}] Saving Checkpoint")
                    torch.save((self.gaussians.capture(), iteration), self.scene.model_path + f"/chkpnt{iteration}.pth")
        finally:
            self._close_live_logs()

    def _maybe_apply_adaptive_stage_extension(self, iteration, eval_results, stats):
        if not self.adaptive_stage_controller.enabled() or not self.is_main_process:
            return

        test_metrics = (eval_results or {}).get("test", {})
        if test_metrics and "psnr" in test_metrics:
            self._prev_test_psnr = self._last_test_psnr
            self._last_test_psnr = float(test_metrics["psnr"])

        decision = self.adaptive_stage_controller.maybe_extend_densify(
            iteration=iteration,
            effective_densify_until=self.effective_densify_until_iter,
            point_count=int(self.gaussians.get_xyz.shape[0]),
            last_test_psnr=self._last_test_psnr,
            prev_test_psnr=self._prev_test_psnr,
            ema_photo_loss=self._ema_photo_loss,
            prev_ema_photo_loss=self._prev_ema_photo_loss,
        )
        stats["train/adaptive_densify_until"] = float(decision.densify_until_iter)
        stats["train/adaptive_spa_start"] = float(decision.spa_start_iter)
        if not decision.extended:
            return

        applied = self.adaptive_stage_controller.apply_decision(decision, self.opt, self.cfg)
        self.effective_densify_until_iter = max(
            int(decision.densify_until_iter),
            int(self.fullres_ready_iter),
        )
        print(
            f"\n[AdaptiveStage] Extended densify to {self.effective_densify_until_iter}; "
            f"SPA {decision.spa_start_iter}-{decision.spa_end_iter} ({decision.reason})"
        )
        model_path = str(getattr(self.scene, "model_path", "") or "")
        if model_path:
            append_runtime_override(
                model_path,
                {
                    "iteration": int(iteration),
                    "reason": decision.reason,
                    **applied,
                },
            )
