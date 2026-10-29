import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("adaptive_budget_test", ROOT / "experiments/adaptive_stage_controller.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Controller = module.AdaptiveStageController

class AdaptiveResumeTests(unittest.TestCase):
    def make(self, enabled=True):
        cfg = SimpleNamespace(enable_adaptive_stage_controller=enabled,
                              adaptive_densify_extend_step=2000, adaptive_densify_max_iter=18000,
                              adaptive_point_budget=8000000, spa_start_iter=13000, spa_end_iter=25000)
        opt = SimpleNamespace(densify_until_iter=10000)
        return Controller(cfg), opt

    def decision(self, controller, step, end):
        return controller.maybe_extend_densify(step, end, 1000000,
            last_test_psnr=25., prev_test_psnr=24.9,
            ema_photo_loss=.1, prev_ema_photo_loss=.11)

    def test_restore_matches_next_uninterrupted_extension(self):
        original, opt = self.make()
        original.apply_decision(self.decision(original, 10000, 10000), opt, original.cfg)
        saved = original.state_dict(opt)
        restored, restored_opt = self.make()
        restored.load_state_dict(saved, restored_opt)
        self.assertEqual(restored_opt.densify_until_iter, 12000)
        self.assertEqual(restored.cfg.spa_end_iter, 27000)
        self.assertEqual(restored._last_extension_iter, 10000)
        a = self.decision(original, 12000, 12000)
        b = self.decision(restored, 12000, 12000)
        self.assertEqual(a, b)
        restored.apply_decision(b, restored_opt, restored.cfg)
        self.assertEqual(restored.cfg.spa_end_iter, 29000)

    def test_repeated_boundary_not_reapplied(self):
        original, opt = self.make()
        original.apply_decision(self.decision(original, 10000, 10000), opt, original.cfg)
        restored, restored_opt = self.make()
        restored.load_state_dict(original.state_dict(opt), restored_opt)
        self.assertEqual(self.decision(restored, 10000, 10000).reason, "already_extended_here")

    def test_enable_flag_mismatch_rejected(self):
        original, opt = self.make()
        restored, restored_opt = self.make(False)
        with self.assertRaisesRegex(ValueError, "enable flag differs"):
            restored.load_state_dict(original.state_dict(opt), restored_opt)

if __name__ == "__main__":
    unittest.main()
