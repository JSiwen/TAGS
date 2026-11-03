import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("floor_test",ROOT/"experiments/budget_policy.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

class BudgetFloorTests(unittest.TestCase):
    def test_exact_integer_threshold(self):
        self.assertEqual(module.matched_budget_floor(7234487),7198315)
        for n in (1,100,200,201,7234487):
            floor=module.matched_budget_floor(n)
            self.assertLessEqual((n-floor)*200,n)
            self.assertGreater((n-floor+1)*200,n)
    def test_boundary_only_allows_remaining_points(self):
        self.assertEqual(module.allowed_geometry_prunes(7198609,500,7198315),294)
        self.assertEqual(module.allowed_geometry_prunes(7198315,500,7198315),0)
        self.assertEqual(module.allowed_geometry_prunes(7202214,500,7198315),500)
    def test_underbudget_not_silently_repaired(self):
        with self.assertRaises(ValueError): module.allowed_geometry_prunes(7197180,500,7198315)
    def test_disabled_policy_preserves_pruning(self):
        self.assertEqual(module.allowed_geometry_prunes(100,30,0),30)

if __name__ == "__main__": unittest.main()
