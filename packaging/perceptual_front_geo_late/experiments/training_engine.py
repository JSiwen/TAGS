import random
import os
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
        self.fullres_ready_iter = self._fullres_ready_iteration()
        self.effective_densify_until_iter = max(self.opt.densify_until_iter, self.fullres_ready_iter)

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
        if cfg.enable_resolution_adaptive:
            print(f"[Resolution] Using schedule: {self.res_schedule}")
            print(f"[Resolution] Full-resolution recovery until iter {self.fullres_ready_iter}")
        if cfg.enable_geo_prior and int(getattr(cfg, "geo_prior_start_iter", 0)) > 0:
            print(f"[GeoPrior] Enabled from iter {int(cfg.geo_prior_start_iter)}")
        if self.distributed and self.is_main_process:
            print(f"[Distributed] Enabled with world_size={self.world_size}")
        if self.effective_densify_until_iter > self.opt.densify_until_iter:
            print(f"[Densify] Extending densification until iter {self.effective_densify_until_iter}")
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

    def _fullres_ready_iteration(self):
        if not self.cfg.enable_resolution_adaptive:
            return 0
        fullres_start = self.res_schedule.full_resolution_start_iter
        if fullres_start is None:
            return 0
        return fullres_start + max(0, int(self.cfg.spa_after_fullres_delay))

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

    def _update_view_difficulty(self, viewpoint_cam, stats):
        if not (
            bool(getattr(self.cfg, "enable_hard_view_mining", False))
            or bool(getattr(self.cfg, "enable_hard_view_loss_reweight", False))
        ):
            return
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

        first_iter += 1
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
            loss, ll1, stats = self.loss_builder(
                render_pkg,
                viewpoint_cam,
                iteration,
                gaussians=self.gaussians,
                train_state=train_state,
            )
            self._update_view_difficulty(viewpoint_cam, stats)
            stats["train/resolution_scale"] = float(self.curr_scale)
            stats["train/fullres_ready"] = 1.0 if train_state["fullres_ready"] else 0.0
            stats["train/view_sampling_weight"] = float(view_sampling_weight)
            stats["train/view_loss_weight"] = float(train_state["view_loss_weight"])
            stats.update(lr_stats)
            loss.backward()
            self._sync_optimizer_grads()
            iter_end.record()

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

                if self.is_main_process:
                    self.evaluator.log_train_scalars(iteration, ll1, loss, iter_start.elapsed_time(iter_end), stats)
                    self.evaluator.maybe_evaluate(iteration, testing_iterations)

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
