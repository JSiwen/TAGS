"""Runtime adaptive densify/SPA timing controller for TAGSs v16."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AdaptiveDecision:
    extended: bool
    reason: str
    densify_until_iter: int
    spa_start_iter: int
    spa_end_iter: int


class AdaptiveStageController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.extend_step = max(1, int(getattr(cfg, "adaptive_densify_extend_step", 2000)))
        self.max_densify_iter = max(1, int(getattr(cfg, "adaptive_densify_max_iter", 18000)))
        self.point_budget = max(0, int(getattr(cfg, "adaptive_point_budget", 8_000_000)))
        self._last_extension_iter = -1

    def state_dict(self, opt):
        return {
            "enabled": self.enabled(),
            "last_extension_iter": self._last_extension_iter,
            "densify_until_iter": int(opt.densify_until_iter),
            "spa_start_iter": int(self.cfg.spa_start_iter),
            "spa_end_iter": int(self.cfg.spa_end_iter),
        }

    def load_state_dict(self, state, opt):
        if bool(state["enabled"]) != self.enabled():
            raise ValueError("Adaptive controller enable flag differs from checkpoint")
        self._last_extension_iter = int(state["last_extension_iter"])
        opt.densify_until_iter = int(state["densify_until_iter"])
        self.cfg.spa_start_iter = int(state["spa_start_iter"])
        self.cfg.spa_end_iter = int(state["spa_end_iter"])

    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "enable_adaptive_stage_controller", False))

    def maybe_extend_densify(
        self,
        iteration: int,
        effective_densify_until: int,
        point_count: int,
        *,
        last_test_psnr: Optional[float],
        prev_test_psnr: Optional[float],
        ema_photo_loss: Optional[float],
        prev_ema_photo_loss: Optional[float],
    ) -> AdaptiveDecision:
        if not self.enabled():
            return AdaptiveDecision(
                extended=False,
                reason="disabled",
                densify_until_iter=effective_densify_until,
                spa_start_iter=int(self.cfg.spa_start_iter),
                spa_end_iter=int(self.cfg.spa_end_iter),
            )

        if iteration != effective_densify_until:
            return AdaptiveDecision(
                extended=False,
                reason="not_at_boundary",
                densify_until_iter=effective_densify_until,
                spa_start_iter=int(self.cfg.spa_start_iter),
                spa_end_iter=int(self.cfg.spa_end_iter),
            )

        if self._last_extension_iter == iteration:
            return AdaptiveDecision(
                extended=False,
                reason="already_extended_here",
                densify_until_iter=effective_densify_until,
                spa_start_iter=int(self.cfg.spa_start_iter),
                spa_end_iter=int(self.cfg.spa_end_iter),
            )

        if point_count >= self.point_budget:
            return AdaptiveDecision(
                extended=False,
                reason="point_budget_met",
                densify_until_iter=effective_densify_until,
                spa_start_iter=int(self.cfg.spa_start_iter),
                spa_end_iter=int(self.cfg.spa_end_iter),
            )

        new_densify = min(self.max_densify_iter, effective_densify_until + self.extend_step)
        if new_densify <= effective_densify_until:
            return AdaptiveDecision(
                extended=False,
                reason="max_densify_reached",
                densify_until_iter=effective_densify_until,
                spa_start_iter=int(self.cfg.spa_start_iter),
                spa_end_iter=int(self.cfg.spa_end_iter),
            )

        quality_improving = self._quality_still_improving(
            last_test_psnr=last_test_psnr,
            prev_test_psnr=prev_test_psnr,
            ema_photo_loss=ema_photo_loss,
            prev_ema_photo_loss=prev_ema_photo_loss,
        )
        if not quality_improving:
            return AdaptiveDecision(
                extended=False,
                reason="quality_plateau",
                densify_until_iter=effective_densify_until,
                spa_start_iter=int(self.cfg.spa_start_iter),
                spa_end_iter=int(self.cfg.spa_end_iter),
            )

        spa_shift = new_densify - effective_densify_until
        new_spa_start = max(int(self.cfg.spa_start_iter) + spa_shift, new_densify + 500)
        new_spa_end = int(self.cfg.spa_end_iter) + spa_shift
        self._last_extension_iter = iteration
        return AdaptiveDecision(
            extended=True,
            reason="capacity_extension",
            densify_until_iter=new_densify,
            spa_start_iter=new_spa_start,
            spa_end_iter=new_spa_end,
        )

    @staticmethod
    def _quality_still_improving(
        *,
        last_test_psnr: Optional[float],
        prev_test_psnr: Optional[float],
        ema_photo_loss: Optional[float],
        prev_ema_photo_loss: Optional[float],
    ) -> bool:
        if last_test_psnr is not None and prev_test_psnr is not None:
            return float(last_test_psnr) > float(prev_test_psnr)
        if ema_photo_loss is not None and prev_ema_photo_loss is not None:
            return float(ema_photo_loss) < float(prev_ema_photo_loss)
        return True

    def apply_decision(self, decision: AdaptiveDecision, opt, cfg) -> Dict[str, int]:
        if not decision.extended:
            return {}
        opt.densify_until_iter = int(decision.densify_until_iter)
        cfg.spa_start_iter = int(decision.spa_start_iter)
        cfg.spa_end_iter = int(decision.spa_end_iter)
        return {
            "densify_until_iter": int(decision.densify_until_iter),
            "spa_start_iter": int(decision.spa_start_iter),
            "spa_end_iter": int(decision.spa_end_iter),
        }
