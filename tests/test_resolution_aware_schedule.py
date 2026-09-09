import importlib.util
import math
import sys
import unittest
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "resolution_aware_schedule.py"
_SPEC = importlib.util.spec_from_file_location("resolution_aware_schedule", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)

DEFAULT_REF_PIXELS = _MOD.DEFAULT_REF_PIXELS
ResolutionProfile = _MOD.ResolutionProfile
ResolutionProfiler = _MOD.ResolutionProfiler
SchedulePlanner = _MOD.SchedulePlanner


class ResolutionAwareScheduleTests(unittest.TestCase):
    def test_scale_one_keeps_e1b_densify(self):
        profile = ResolutionProfile(
            cur_pixels=DEFAULT_REF_PIXELS,
            ref_pixels=DEFAULT_REF_PIXELS,
            resolution_scale=1.0,
        )
        plan = SchedulePlanner.plan(profile, 30_000, densify_base=7200)
        self.assertEqual(plan.densify_until_iter, 7200)
        self.assertGreater(plan.spa_start_iter, plan.densify_until_iter)
        self.assertGreater(plan.geo_prior_warmup_until, 0)
        self.assertLessEqual(plan.geo_prior_warmup_until, 7000)

    def test_scale_two_targets_fullres_capacity(self):
        profile = ResolutionProfile(
            cur_pixels=DEFAULT_REF_PIXELS * 4,
            ref_pixels=DEFAULT_REF_PIXELS,
            resolution_scale=2.0,
        )
        plan = SchedulePlanner.plan(profile, 30_000, densify_base=7200, densify_min=10_000, densify_max=18_000)
        self.assertGreaterEqual(plan.densify_until_iter, 14_000)
        self.assertLessEqual(plan.densify_until_iter, 15_000)
        self.assertGreater(plan.spa_start_iter, plan.densify_until_iter)
        self.assertGreater(plan.geo_prior_warmup_until, 0)
        self.assertLess(plan.geo_prior_warmup_until, plan.densify_until_iter)

    def test_late_losses_start_after_spa_window(self):
        profile = ResolutionProfile(
            cur_pixels=DEFAULT_REF_PIXELS * 4,
            ref_pixels=DEFAULT_REF_PIXELS,
            resolution_scale=2.0,
        )
        plan = SchedulePlanner.plan(profile, 30_000)
        self.assertGreaterEqual(plan.psnr_loss_start_iter, plan.spa_start_iter + 3000)
        self.assertGreaterEqual(plan.lpips_loss_start_iter, plan.spa_end_iter - 1000)
        self.assertGreaterEqual(plan.color_align_start_iter, plan.spa_end_iter)

    def test_checkpoints_include_primary_iters(self):
        profile = ResolutionProfile(
            cur_pixels=DEFAULT_REF_PIXELS * 4,
            ref_pixels=DEFAULT_REF_PIXELS,
            resolution_scale=2.0,
        )
        plan = SchedulePlanner.plan(profile, 30_000)
        self.assertIn(28_500, plan.checkpoint_iterations)
        self.assertIn(30_000, plan.checkpoint_iterations)
        self.assertEqual(plan.checkpoint_iterations, sorted(plan.checkpoint_iterations))

    def test_resolution_schedule_scales_breakpoints(self):
        profile = ResolutionProfile(
            cur_pixels=DEFAULT_REF_PIXELS * 4,
            ref_pixels=DEFAULT_REF_PIXELS,
            resolution_scale=2.0,
        )
        plan = SchedulePlanner.plan(profile, 30_000)
        self.assertEqual(plan.resolution_schedule, "2400:4,8000:2,30000:1")

    def test_compute_scale_uses_sqrt_ratio(self):
        scale = ResolutionProfiler.compute_scale(DEFAULT_REF_PIXELS * 4, DEFAULT_REF_PIXELS)
        self.assertAlmostEqual(scale, 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
