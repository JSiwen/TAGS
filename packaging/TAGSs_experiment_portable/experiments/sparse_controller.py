import torch


class SparseController:
    def __init__(self, cfg, resolution_schedule=None):
        self.cfg = cfg
        self.resolution_schedule = resolution_schedule
        self.z = None
        self.u = None
        self.final_pruned = False
        self.contribution_ema = None

    def apply(self, iteration, gaussians, scene, opt, render_pkg, train_state=None, sync_fn=None):
        if not self.cfg.enable_sparse_optimization:
            return {}
        self._record_contribution(gaussians, render_pkg)
        if sync_fn is not None and self.contribution_ema is not None:
            sync_fn(self.contribution_ema, op="mean")
        mode = self.cfg.sparse_mode
        stats = {}
        if mode in ("dash", "dash_spa"):
            stats.update(self._apply_budget_prune(gaussians))
        if mode in ("spa", "dash_spa"):
            stats.update(self._apply_spa_update(iteration, gaussians, train_state))
            stats.update(self._apply_spa_final_prune(iteration, gaussians))
        return stats

    def penalty(self, iteration, gaussians, train_state=None):
        if not self.cfg.enable_sparse_optimization or self.cfg.sparse_mode not in ("spa", "dash_spa"):
            return torch.tensor(0.0, device=gaussians.get_opacity.device), {}
        if iteration < self.cfg.spa_start_iter or iteration > self.cfg.spa_end_iter:
            return torch.tensor(0.0, device=gaussians.get_opacity.device), {}
        if not self._spa_allowed_for_resolution(iteration, train_state):
            return torch.tensor(0.0, device=gaussians.get_opacity.device), {"sparse/spa_penalty_skipped_resolution": 1.0}

        self._ensure_spa_state(gaussians)
        diff = gaussians.get_opacity - self.z + self.u
        rho = float(self.cfg.rho_lr)
        penalty = 0.5 * rho * torch.mean(diff * diff)
        return penalty, {"sparse/spa_penalty": float(penalty.item()), "sparse/spa_rho": rho}

    def _apply_budget_prune(self, gaussians):
        upper = int(self.cfg.sparse_budget_upperbound)
        if upper <= 0:
            return {}
        n = gaussians.get_xyz.shape[0]
        if n <= upper:
            return {"sparse/points": n}
        scores = self._importance_scores(gaussians)
        keep_indices = torch.topk(scores, k=upper, largest=True).indices
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        keep_mask[keep_indices] = True
        prune_mask = ~keep_mask
        gaussians.prune_points(prune_mask)
        self._reset_spa_state()
        return {"sparse/budget_pruned": n - upper, "sparse/points": upper}

    def _spa_allowed_for_resolution(self, iteration, train_state):
        if not self.cfg.spa_require_full_resolution:
            return True
        current_scale = train_state.get("scale", 1.0) if train_state else 1.0
        if current_scale != 1.0:
            return False
        if train_state and not train_state.get("fullres_ready", True):
            return False

        if self.resolution_schedule is None:
            return True
        fullres_start = self.resolution_schedule.full_resolution_start_iter
        if fullres_start is None:
            return True
        delay = max(0, int(self.cfg.spa_after_fullres_delay))
        return iteration >= fullres_start + delay

    def _ensure_spa_state(self, gaussians):
        opacity = gaussians.get_opacity
        if self.z is None or self.u is None or self.z.shape != opacity.shape:
            self.u = torch.zeros_like(opacity)
            self.z = opacity.detach().clone()

    def _update_spa_z(self, gaussians):
        opacity = gaussians.get_opacity
        z_candidate = opacity.detach() + self.u
        ratio = self._spa_zero_ratio()
        prune_n = int(ratio * z_candidate.shape[0])
        prune_n = self._cap_prune_count(prune_n, z_candidate.shape[0])
        if prune_n <= 0:
            self.z = z_candidate.detach().clone()
            return 0
        prune_n = min(prune_n, z_candidate.shape[0] - 1)
        scores = self._importance_scores(gaussians, opacity_override=z_candidate)
        protected = self._structure_protection_mask(gaussians)
        candidate_scores = scores.masked_fill(protected, float("inf"))
        finite_candidates = torch.isfinite(candidate_scores)
        if finite_candidates.sum() <= 1:
            self.z = z_candidate.detach().clone()
            return 0
        prune_n = min(prune_n, int(finite_candidates.sum().item()) - 1)
        threshold = torch.topk(candidate_scores, k=prune_n, largest=False).values.max()
        keep_mask = ((candidate_scores > threshold) | protected).unsqueeze(-1)
        self.z = torch.where(keep_mask, z_candidate, torch.zeros_like(z_candidate)).detach()
        return prune_n

    def _apply_spa_update(self, iteration, gaussians, train_state):
        if iteration < self.cfg.spa_start_iter or iteration > self.cfg.spa_end_iter:
            return {}
        if self.cfg.spa_interval <= 0 or iteration % self.cfg.spa_interval != 0:
            return {}
        if not self._spa_allowed_for_resolution(iteration, train_state):
            return {"sparse/spa_skipped_resolution": 1.0}

        self._ensure_spa_state(gaussians)
        zeroed = self._update_spa_z(gaussians)
        protected = int(self._structure_protection_mask(gaussians).sum().item())
        with torch.no_grad():
            self.u = self.u + gaussians.get_opacity.detach() - self.z
        return {
            "sparse/spa_z_zeroed": zeroed,
            "sparse/spa_protected": protected,
            "sparse/points": gaussians.get_xyz.shape[0],
        }

    def _apply_spa_final_prune(self, iteration, gaussians):
        if self.final_pruned or iteration != self.cfg.spa_end_iter:
            return {}
        scores = self._importance_scores(gaussians)
        n = scores.shape[0]
        keep_ratio = max(0.0, min(1.0, 1.0 - self._spa_zero_ratio()))
        prune_n = self._cap_prune_count(n - int(keep_ratio * n), n)
        keep_n = n - prune_n
        if keep_n <= 0 or keep_n >= n:
            self.final_pruned = True
            return {"sparse/points": n}
        protected = self._structure_protection_mask(gaussians)
        keep_n = max(keep_n, min(n, int(protected.sum().item())))
        scores = scores.masked_fill(protected, float("inf"))
        keep_idx = torch.topk(scores, k=keep_n, largest=True).indices
        prune_mask = torch.zeros_like(scores, dtype=torch.bool)
        prune_mask[:] = True
        prune_mask[keep_idx] = False
        gaussians.prune_points(prune_mask)
        self._reset_spa_state()
        self.final_pruned = True
        return {"sparse/spa_pruned": n - keep_n, "sparse/points": keep_n}

    def _spa_zero_ratio(self):
        if float(self.cfg.prune_ratio2) > 0:
            return max(0.0, min(1.0, float(self.cfg.prune_ratio2)))
        return max(0.0, min(1.0, float(self.cfg.spa_prune_ratio)))

    def _cap_prune_count(self, prune_n, total_n):
        max_prune = int(getattr(self.cfg, "spa_max_prune_points", 0))
        if max_prune > 0:
            prune_n = min(prune_n, max_prune)
        return max(0, min(int(prune_n), max(0, int(total_n) - 1)))

    def _reset_spa_state(self):
        self.z = None
        self.u = None
        self.contribution_ema = None

    def _record_contribution(self, gaussians, render_pkg):
        radii = render_pkg.get("radii") if render_pkg else None
        viewspace_points = render_pkg.get("viewspace_points") if render_pkg else None
        if radii is None:
            return
        n = gaussians.get_xyz.shape[0]
        if self.contribution_ema is None or self.contribution_ema.shape[0] != n:
            self.contribution_ema = torch.zeros((n,), dtype=gaussians.get_opacity.dtype, device=gaussians.get_opacity.device)

        visible = radii > 0
        if visible.shape[0] != n:
            return
        current = torch.zeros_like(self.contribution_ema)
        current[visible] = self._normalize(radii.detach().float())[visible]
        grad = getattr(viewspace_points, "grad", None)
        if grad is not None and grad.shape[0] == n:
            current = current + self._normalize(torch.norm(grad.detach()[:, :2], dim=-1))

        self.contribution_ema.mul_(0.95).add_(0.05 * current)

    def _importance_scores(self, gaussians, opacity_override=None):
        opacity = (opacity_override if opacity_override is not None else gaussians.get_opacity).detach().reshape(-1)
        if getattr(self.cfg, "spa_metric", "hybrid") == "opacity":
            return opacity

        score = 0.20 * self._normalize(opacity)
        if self.contribution_ema is not None and self.contribution_ema.shape[0] == opacity.shape[0]:
            score = score + 0.40 * self._normalize(self.contribution_ema)
        if getattr(gaussians, "xyz_gradient_accum", None) is not None and getattr(gaussians, "denom", None) is not None:
            grad = gaussians.xyz_gradient_accum.reshape(-1) / gaussians.denom.reshape(-1).clamp_min(1.0)
            score = score + 0.30 * self._normalize(grad.detach())
        if getattr(gaussians, "max_radii2D", None) is not None:
            score = score + 0.10 * self._normalize(gaussians.max_radii2D.detach())
        return score

    def _structure_protection_mask(self, gaussians):
        n = gaussians.get_xyz.shape[0]
        device = gaussians.get_opacity.device
        protected = torch.zeros((n,), dtype=torch.bool, device=device)
        keep_fraction = float(getattr(self.cfg, "spa_protect_top_fraction", 0.02))
        keep_n = int(max(0, min(n - 1, round(n * keep_fraction))))
        if keep_n <= 0:
            return protected

        if self.contribution_ema is not None and self.contribution_ema.shape[0] == n:
            protected |= self._topk_mask(self.contribution_ema, keep_n)
        if getattr(gaussians, "xyz_gradient_accum", None) is not None and getattr(gaussians, "denom", None) is not None:
            grad = gaussians.xyz_gradient_accum.reshape(-1) / gaussians.denom.reshape(-1).clamp_min(1.0)
            if grad.shape[0] == n:
                protected |= self._topk_mask(grad.detach(), keep_n)
        if getattr(gaussians, "max_radii2D", None) is not None and gaussians.max_radii2D.shape[0] == n:
            protected |= self._topk_mask(gaussians.max_radii2D.detach(), max(1, keep_n // 2))
        return protected

    def _topk_mask(self, values, k):
        values = values.detach().reshape(-1)
        if values.numel() == 0 or k <= 0:
            return torch.zeros_like(values, dtype=torch.bool)
        k = min(k, values.shape[0])
        idx = torch.topk(values.float(), k=k, largest=True).indices
        mask = torch.zeros_like(values, dtype=torch.bool)
        mask[idx] = True
        return mask

    def _normalize(self, values):
        values = values.detach().float()
        if values.numel() == 0:
            return values
        v_min = values.min()
        v_max = values.max()
        denom = (v_max - v_min).clamp_min(1e-8)
        return (values - v_min) / denom
