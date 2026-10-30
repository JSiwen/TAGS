import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("budget_schedule_test", ROOT / "experiments/resolution_aware_schedule.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class SaveScheduleTests(unittest.TestCase):
    def apply(self, budget):
        fields = ("densify_until_iter geo_prior_warmup_until geo_prior_decay_start geo_prior_decay_until "
                  "spa_start_iter spa_end_iter lpips_loss_start_iter psnr_loss_start_iter "
                  "color_align_start_iter lr_stage1_end lr_stage2_end geometry_start_iter").split()
        plan = SimpleNamespace(**{key: 10000 for key in fields})
        plan.resolution_schedule = "1853:4,6177:2,30000:1"
        plan.checkpoint_iterations = [7000, 10000, 30000]
        plan.late_controller_structure_end_iter = None
        cfg = SimpleNamespace(final_target_gaussians=budget)
        args = SimpleNamespace(save_iterations=[30000], test_iterations=[30000], checkpoint_iterations=[25000,30000])
        module.apply_stage_plan(plan, SimpleNamespace(), cfg, args)
        return args

    def test_budget_preserves_explicit_ply_saves_and_evaluation(self):
        args = self.apply(7234487)
        self.assertEqual(args.save_iterations, [30000])
        self.assertEqual(args.test_iterations, [7000,10000,30000])
        self.assertEqual(args.checkpoint_iterations, [25000,30000])

    def test_legacy_save_schedule_unchanged(self):
        self.assertEqual(self.apply(0).save_iterations, [7000,10000,30000])

if __name__ == "__main__":
    unittest.main()
