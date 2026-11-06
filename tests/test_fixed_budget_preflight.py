import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("budget_driver", ROOT / "scripts/run_fixed_budget_sweep.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

class PreflightTests(unittest.TestCase):
    def config(self):
        return dict(position_lr_init=.00016, densify_until_iter=18000, enable_geo_prior=True,
                    enable_sparse_optimization=True, sparse_mode="spa", spa_end_iter=25000,
                    lpips_loss_weight_init=.004, psnr_loss_weight_final=.01, geo_prior_start_iter=26000)

    def test_complete_update(self):
        driver.preflight(ROOT, self.config())

    def test_original_3dgs_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "actual TAGS 30k baseline"):
            driver.preflight(ROOT, {"images": "images_4", "resolution": 1})

    def test_stale_arguments_rejected(self):
        # Missing code must fail before CUDA imports or output creation.
        with self.assertRaisesRegex(ValueError, "Update ALL experiment files"):
            driver.preflight(ROOT / "tests", self.config())

if __name__ == "__main__":
    unittest.main()
