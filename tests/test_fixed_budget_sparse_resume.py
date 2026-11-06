import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("sparse_resume_driver", ROOT/"scripts/run_fixed_budget_sweep.py")
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

class SparseResumeTests(unittest.TestCase):
    def test_inherit_source_spa_without_overriding_loss_candidate(self):
        cfg={"spa_target_gaussians":0,"geo_scale":.5,"lpips_final_multiplier":1.}
        source={"spa_target_gaussians":7234487,"spa_protect_top_fraction":.05,
                "spa_end_iter":27118,"sparse_mode":"spa","geo_scale":1.,"lpips_final_multiplier":.5}
        with patch.object(Path,"is_file",return_value=True), patch.object(Path,"read_bytes",return_value=b"cfg"), patch.object(driver,"read_config",return_value=source):
            record=driver.inherit_checkpoint_structure(cfg,Path("source/chkpnt25000.pth"))
        self.assertEqual(cfg["spa_target_gaussians"],7234487)
        self.assertEqual(cfg["spa_end_iter"],27118)
        self.assertEqual(cfg["geo_scale"],.5)
        self.assertEqual(cfg["lpips_final_multiplier"],1.)
        self.assertEqual(record["values"]["spa_protect_top_fraction"],.05)
    def test_missing_source_config_refused(self):
        with patch.object(Path,"is_file",return_value=False):
            with self.assertRaisesRegex(ValueError,"source cfg_args missing"):
                driver.inherit_checkpoint_structure({},Path("source/chkpnt25000.pth"))
    def controller_class(self):
        tree=ast.parse((ROOT/"experiments/sparse_controller.py").read_text(encoding="utf-8"))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ("capture_config","restore_config")]
        namespace={};exec(compile(ast.Module(body=[cls],type_ignores=[]),"sparse_config_test","exec"),namespace)
        return namespace["SparseController"]
    def test_structural_config_round_trip_does_not_restore_loss_weights(self):
        cls=self.controller_class();a=cls();b=cls()
        a.cfg=SimpleNamespace(spa_target_gaussians=7234487,spa_end_iter=27118,rho_lr=.0005,geo_scale=1.)
        b.cfg=SimpleNamespace(spa_target_gaussians=0,spa_end_iter=21118,geo_scale=.5)
        b.restore_config(a.capture_config())
        self.assertEqual(b.cfg.spa_target_gaussians,7234487)
        self.assertEqual(b.cfg.spa_end_iter,27118)
        self.assertEqual(b.cfg.geo_scale,.5)
    def test_checkpoint_nonstructural_override_refused(self):
        obj=self.controller_class()();obj.cfg=SimpleNamespace()
        with self.assertRaises(ValueError): obj.restore_config({"geo_scale":1.})

if __name__ == "__main__": unittest.main()
