class BalancedLateController:
    def __init__(self, cfg, total_iterations):
        self.cfg = cfg
        self.total_iterations = int(total_iterations)
        chronic = str(getattr(cfg, "late_controller_chronic_views", "")).strip()
        self.chronic_views = {
            name.strip() for name in chronic.split(",")
            if name.strip()
        }

    def enabled(self):
        return bool(getattr(self.cfg, "enable_balanced_late_controller", False))

    def stage(self, iteration):
        structure_end = int(getattr(self.cfg, "late_controller_structure_end_iter", 24000))
        handoff_end = int(getattr(self.cfg, "late_controller_handoff_end_iter", 25500))
        refine_end = int(getattr(self.cfg, "late_controller_refine_end_iter", 28500))
        if iteration <= structure_end:
            return "structure"
        if iteration <= handoff_end:
            return "handoff"
        if iteration <= refine_end:
            return "refine"
        return "polish"

    def _ramp(self, iteration, start_iter, end_iter):
        if end_iter <= start_iter:
            return 1.0 if iteration >= end_iter else 0.0
        if iteration <= start_iter:
            return 0.0
        if iteration >= end_iter:
            return 1.0
        return float(iteration - start_iter) / float(end_iter - start_iter)

    def budget_psnr_multiplier(self, iteration):
        start = int(getattr(self.cfg, "psnr_ramp_start", 25000))
        end = int(getattr(self.cfg, "psnr_ramp_end", 27000))
        x = self._ramp(iteration, start, end)
        return 1.0 + (float(getattr(self.cfg, "psnr_final_multiplier", 1.0)) - 1.0) * x*x*(3-2*x)

    def build_state(self, iteration, view_name, view_lpips_ema, global_lpips_ema):
        if not self.enabled():
            return {
                "enabled": False,
                "stage": "disabled",
                "lpips_guard_multiplier": 1.0,
                "color_align_multiplier": 1.0,
                "psnr_multiplier": self.budget_psnr_multiplier(iteration),
                "is_chronic_view": False,
            }

        stage = self.stage(iteration)
        guard_start = int(getattr(self.cfg, "late_controller_lpips_guard_start_iter", 24500))
        guard_end = int(getattr(self.cfg, "late_controller_refine_end_iter", 28500))
        guard_ramp = self._ramp(iteration, guard_start, guard_end)
        base_boost = max(1.0, float(getattr(self.cfg, "late_controller_lpips_guard_max_boost", 1.35)))
        lpips_guard_multiplier = 1.0 + (base_boost - 1.0) * guard_ramp

        view_boost = max(1.0, float(getattr(self.cfg, "late_controller_lpips_guard_view_boost", 1.20)))
        margin = max(0.0, float(getattr(self.cfg, "late_controller_lpips_guard_margin", 0.08)))
        view_lpips = float(view_lpips_ema.get(view_name, global_lpips_ema or 0.0))
        global_lpips = float(global_lpips_ema or 0.0)
        if global_lpips > 0.0 and view_lpips > global_lpips * (1.0 + margin):
            lpips_guard_multiplier *= view_boost

        is_chronic_view = view_name in self.chronic_views
        if is_chronic_view:
            chronic_boost = max(1.0, float(getattr(self.cfg, "late_controller_lpips_guard_chronic_boost", 1.10)))
            lpips_guard_multiplier *= chronic_boost

        if stage == "structure":
            color_align_multiplier = 0.0
            psnr_multiplier = 1.0
        elif stage == "handoff":
            color_align_multiplier = 1.0
            psnr_multiplier = 1.0
        elif stage == "refine":
            color_boost = max(1.0, float(getattr(self.cfg, "late_controller_color_stage_boost", 1.15)))
            color_align_multiplier = color_boost
            psnr_multiplier = 1.0
        else:
            polish_scale = max(0.0, min(1.0, float(getattr(self.cfg, "late_controller_color_polish_scale", 0.75))))
            color_align_multiplier = polish_scale
            psnr_multiplier = 1.0

        return {
            "enabled": True,
            "stage": stage,
            "lpips_guard_multiplier": float(lpips_guard_multiplier),
            "color_align_multiplier": float(color_align_multiplier),
            "psnr_multiplier": float(psnr_multiplier) * self.budget_psnr_multiplier(iteration),
            "is_chronic_view": bool(is_chronic_view),
        }
