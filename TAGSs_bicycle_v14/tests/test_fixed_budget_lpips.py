import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("late_lpips_test", ROOT / "experiments/balanced_late_controller.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class LateLPIPSTests(unittest.TestCase):
    def test_boundary_ramp_with_and_without_balanced_controller(self):
        for enabled in (False, True):
            cfg = SimpleNamespace(enable_balanced_late_controller=enabled,
                                  psnr_final_multiplier=1.5, lpips_final_multiplier=.5)
            controller = module.BalancedLateController(cfg,30000)
            for step, expected in ((24000,1.), (25000,1.), (26000,.75), (27000,.5), (30000,.5)):
                state = controller.build_state(step,"view",{},None)
                self.assertAlmostEqual(state["lpips_loss_multiplier"],expected)
            self.assertAlmostEqual(controller.build_state(30000,"view",{},None)["psnr_multiplier"],1.5)

    def test_old_config_preserves_lpips(self):
        controller = module.BalancedLateController(SimpleNamespace(),30000)
        for step in (1,25000,26000,27000,30000):
            self.assertEqual(controller.build_state(step,"view",{},None)["lpips_loss_multiplier"],1.)

    def test_loss_weight_composes_once_with_guard_and_view_weight(self):
        tree = ast.parse((ROOT / "experiments/loss_builder.py").read_text(encoding="utf-8"))
        node = next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id == "effective_lpips_w" for t in n.targets))
        expr = compile(ast.Expression(body=node),"loss-weight", "eval")
        for multiplier in (1.,.5):
            actual = eval(expr, {}, dict(lpips_w=.006, late_state={"lpips_guard_multiplier":1.2},
                                        view_loss_weight=1.1, stats={"train/lpips_multiplier":multiplier}))
            self.assertAlmostEqual(actual, .006*1.2*1.1*multiplier)

if __name__ == "__main__":
    unittest.main()
